import os
from typing import Callable, Dict, Optional, Any
import torch

from sparse_frontier.modelling.attention.registry import (
    get_attention_handler,
    ensure_attention_initialized_from_env,
)
from .abstract_model import AbstractModel
from sparse_frontier.modelling.tokenizer import Tokenizer


_ORIGINAL_FLASH_ATTENTION_FORWARD: Optional[Callable[..., torch.Tensor]] = None


class VLLMModel(AbstractModel):
    def __init__(
        self,
        model_path: str,
        max_input_tokens: int = 8192,
        max_output_tokens: int = 256,
        dtype: torch.dtype = None,
        tensor_parallel_size: int = 1,
        seed: Optional[int] = 43,
        enable_thinking: bool = False,
    ):
        """
            vLLM uses forking to run the TP. Therefore we can't initialise CUDA,
            before the fork, as it will trigger an error. That's why we don't
            call get_device() etc.
        """
        self.model_path = model_path
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.dtype = dtype or torch.bfloat16
        self.tensor_parallel_size = tensor_parallel_size
        self.seed = seed
        self.enable_thinking = enable_thinking

        assert not torch.cuda.is_initialized(), "CUDA is not initialized"
        self.model = self._load_model(self.model_path)
        self.tokenizer = Tokenizer(self.model_path, thinking=self.enable_thinking)

    def _load_model(self, model_path: str):
        if 'Qwen' in model_path and self.max_input_tokens + self.max_output_tokens > 32768:
            factor = (self.max_input_tokens + self.max_output_tokens) / 32768
            hf_overrides = {
                "rope_scaling": {
                    "factor": factor,
                    "original_max_position_embeddings": 32768,
                    "rope_type": "yarn"
                }
            }
        else:
            hf_overrides = {}

        from vllm import LLM
        model = LLM(
            model=model_path,
            # skip_tokenizer_init=True,  # Disabled: Gemma requires tokenizer for model configuration
            enforce_eager=True,
            seed=self.seed,
            gpu_memory_utilization=0.85,
            max_num_batched_tokens=self.max_input_tokens + self.max_output_tokens,
            max_model_len=self.max_input_tokens + self.max_output_tokens,
            enable_chunked_prefill=False, # it's no-op in v1
            enable_prefix_caching=False,
            tensor_parallel_size=self.tensor_parallel_size,
            hf_overrides=hf_overrides,
            limit_mm_per_prompt={"image": 0},
            disable_hybrid_kv_cache_manager=True, # this fixes gemma3 which has sliding window cache and it impacts the way we handle kv cache
        )

        return model

    def _sampling_config(self) -> Dict:
        from vllm import SamplingParams

        common_kwargs = {
            'max_tokens': self.max_output_tokens,
            'seed': self.seed,
        }

        if self.enable_thinking:
            # Qwen 3 config for thinking mode
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                **common_kwargs,
            )
        else:
            sampling_params = SamplingParams(
                temperature=0.0,
                **common_kwargs,
            )

        return {
            'sampling_params': sampling_params,
            'use_tqdm': False,
        }

    @torch.no_grad()
    def generate(
        self,
        input_text: str,
    ) -> str:
        from vllm.inputs import TokensPrompt
        model_input = self.tokenizer.encode_for_generation(input_text, return_tensors=False)
        prompt = TokensPrompt(prompt_token_ids=model_input["input_ids"])

        sampling_config = self._sampling_config()
        output_ids = self.model.generate(
            prompts=[prompt],
            **sampling_config,
        )[0].outputs[0].token_ids

        decoded = self.tokenizer.decode(output_ids)

        output: dict[str, Any] = {
            'text': decoded[0] if isinstance(decoded, list) else decoded,
            'output_tokens_len': len(output_ids),
        }

        return output

def vllm_patched_forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
):
    global _ORIGINAL_FLASH_ATTENTION_FORWARD

    if attn_metadata is None:
        # This is vLLM's profiling run - initialize attention and pre-allocate memory
        ensure_attention_initialized_from_env(key)
        return output
    
    handler = get_attention_handler()
    num_tokens = query.shape[0]
    is_prefilling = num_tokens > 1

    # Initialize tracking state at layer 0 (needed for all layer types including sliding window)
    if handler.current_layer == 0:
        if is_prefilling:
            handler._begin_prefill(prompt_len=num_tokens)
        else:
            handler._begin_decode_step()

    sliding_window = getattr(self, "sliding_window", (-1, -1))

    if sliding_window != (-1, -1):
        handler.advance_layer()

        return _ORIGINAL_FLASH_ATTENTION_FORWARD(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
    else:
        handler(
            queries=query.contiguous(),
            keys=key.contiguous(),
            values=value.contiguous(),
            kv_cache=kv_cache.contiguous(),
            output=output,
        )
        return output


def swap_vllm_attention():
    is_attention_patch_enabled = os.getenv("SF_USE_ATTENTION_PATCH", "1") != "0"
    if not is_attention_patch_enabled:
        return

    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    global _ORIGINAL_FLASH_ATTENTION_FORWARD

    if _ORIGINAL_FLASH_ATTENTION_FORWARD is None:
        _ORIGINAL_FLASH_ATTENTION_FORWARD = FlashAttentionImpl.forward

    FlashAttentionImpl.forward = vllm_patched_forward
