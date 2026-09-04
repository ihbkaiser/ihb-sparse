import os
import json
import subprocess
import threading
import time
from typing import Callable, Dict, Optional, Any
import torch

from sparse_frontier.modelling.attention.registry import (
    get_attention_handler,
    ensure_attention_initialized_from_env,
)
from .abstract_model import AbstractModel
from sparse_frontier.modelling.tokenizer import Tokenizer


_ORIGINAL_FLASH_ATTENTION_FORWARD: Optional[Callable[..., torch.Tensor]] = None
_ORIGINAL_ROTARY_EMBEDDING_CALL: Optional[Callable[..., Any]] = None
_QUERY_CAPTURE_COLLECTOR = None


def _query_capture_enabled() -> bool:
    return bool(os.getenv("SF_QUERY_CAPTURE_DIR")) and os.getenv(
        "SF_ATTENTION_NAME"
    ) in {None, "", "dense"}


def _get_query_capture_collector():
    """Construct the rank-local calibration collector inside the vLLM worker."""
    global _QUERY_CAPTURE_COLLECTOR
    if _QUERY_CAPTURE_COLLECTOR is not None:
        return _QUERY_CAPTURE_COLLECTOR

    from sparse_frontier.modelling.attention.query_capture import QueryCaptureCollector

    required = {
        name: os.getenv(name)
        for name in (
            "SF_QUERY_CAPTURE_DIR",
            "SF_QUERY_CAPTURE_CONTEXT",
            "SF_MODEL_NUM_LAYERS",
            "SF_MODEL_NUM_Q_HEADS",
            "SF_MODEL_NUM_KV_HEADS",
            "SF_MODEL_HEAD_DIM",
            "SF_TP_SIZE",
        )
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "query capture is missing required environment variables: "
            + ", ".join(missing)
        )
    tp_size = int(required["SF_TP_SIZE"])
    global_q_heads = int(required["SF_MODEL_NUM_Q_HEADS"])
    global_kv_heads = int(required["SF_MODEL_NUM_KV_HEADS"])
    if global_q_heads % tp_size or global_kv_heads % tp_size:
        raise RuntimeError("query capture requires query and KV heads divisible by TP")
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        tp_rank = get_tensor_model_parallel_rank()
    except (AssertionError, ImportError, RuntimeError):
        tp_rank = int(os.getenv("RANK", "0")) % tp_size

    layer_spec = os.getenv("SF_QUERY_CAPTURE_LAYERS")
    capture_layers = None
    if layer_spec:
        try:
            parsed = json.loads(layer_spec)
            if not isinstance(parsed, list):
                raise TypeError("expected a JSON list")
            capture_layers = tuple(int(layer) for layer in parsed)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "SF_QUERY_CAPTURE_LAYERS must be a JSON list of layer indices"
            ) from exc

    _QUERY_CAPTURE_COLLECTOR = QueryCaptureCollector(
        capture_dir=required["SF_QUERY_CAPTURE_DIR"],
        context_path=required["SF_QUERY_CAPTURE_CONTEXT"],
        num_layers=int(required["SF_MODEL_NUM_LAYERS"]),
        num_q_heads=global_q_heads // tp_size,
        num_kv_heads=global_kv_heads // tp_size,
        head_dim=int(required["SF_MODEL_HEAD_DIM"]),
        tp_rank=tp_rank,
        capture_layers=capture_layers,
        capture_values=os.getenv("SF_QUERY_CAPTURE_VALUES", "0") == "1",
        capture_prompt_keys=os.getenv("SF_QUERY_CAPTURE_PROMPT_KEYS", "1") == "1",
        capture_prompt_queries=os.getenv("SF_QUERY_CAPTURE_PROMPT_QUERIES", "0") == "1",
        prompt_query_samples=int(os.getenv("SF_QUERY_CAPTURE_PROMPT_QUERY_SAMPLES", "16")),
        prompt_query_seed=int(os.getenv("SF_QUERY_CAPTURE_PROMPT_QUERY_SEED", "1043")),
        composition_tolerance=float(
            os.getenv("SF_QUERY_CAPTURE_COMPOSITION_TOLERANCE", "0.05")
        ),
    )
    return _QUERY_CAPTURE_COLLECTOR


def _sparse_chunked_prefill_requested() -> bool:
    """Whether the configured sparse run can be chunked by vLLM v1."""
    if os.getenv("SF_ATTENTION_NAME") not in {"quest", "shadowkv", "query_robust"}:
        return False
    try:
        max_batched = int(os.getenv("SF_MAX_NUM_BATCHED_TOKENS", "0"))
        max_model = int(os.getenv("SF_MAX_INPUT_TOKENS", "0")) + int(
            os.getenv("SF_MAX_OUTPUT_TOKENS", "0")
        )
    except ValueError:
        return False
    return max_batched > 0 and max_model > 0 and max_batched < max_model


class _GpuMemorySampler:
    """Sample device memory used by vLLM's worker process.

    vLLM v1 owns CUDA allocations in its engine worker, not in the frontend
    process that calls ``LLM.generate``.  ``torch.cuda.max_memory_allocated``
    in this process therefore reports zero; nvidia-smi is the least invasive
    dependency-free way to capture the requested peak for this runner.
    """

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read_bytes() -> int | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        values = []
        for line in result.stdout.splitlines():
            try:
                values.append(int(line.strip()) * 1024 * 1024)
            except ValueError:
                continue
        return max(values) if values else None

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            value = self._read_bytes()
            if value is not None:
                self.peak_bytes = max(self.peak_bytes or 0, value)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.peak_bytes = self._read_bytes()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 2))
        value = self._read_bytes()
        if value is not None:
            self.peak_bytes = max(self.peak_bytes or 0, value)
        return self.peak_bytes


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
        self.last_metrics: dict[str, Any] = {}

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
        gpu_memory_utilization = float(os.getenv("SF_GPU_MEMORY_UTILIZATION", "0.85"))
        if not 0.0 < gpu_memory_utilization <= 1.0:
            raise ValueError(
                "SF_GPU_MEMORY_UTILIZATION must be in (0, 1], got "
                f"{gpu_memory_utilization!r}"
            )
        max_num_batched_tokens = int(
            os.getenv(
                "SF_MAX_NUM_BATCHED_TOKENS",
                str(self.max_input_tokens + self.max_output_tokens),
            )
        )
        if max_num_batched_tokens < 1:
            raise ValueError(
                "SF_MAX_NUM_BATCHED_TOKENS must be positive, got "
                f"{max_num_batched_tokens!r}"
            )
        cpu_offload_gb = float(os.getenv("SF_CPU_OFFLOAD_GB", "0"))
        if cpu_offload_gb < 0.0:
            raise ValueError(
                "SF_CPU_OFFLOAD_GB must be non-negative, got "
                f"{cpu_offload_gb!r}"
            )
        kv_cache_memory_bytes_env = os.getenv("SF_KV_CACHE_MEMORY_BYTES")
        kv_cache_memory_bytes = None
        if kv_cache_memory_bytes_env is not None:
            try:
                kv_cache_memory_bytes = int(kv_cache_memory_bytes_env)
            except ValueError as exc:
                raise ValueError(
                    "SF_KV_CACHE_MEMORY_BYTES must be a positive integer, got "
                    f"{kv_cache_memory_bytes_env!r}"
                ) from exc
            if kv_cache_memory_bytes <= 0:
                raise ValueError(
                    "SF_KV_CACHE_MEMORY_BYTES must be a positive integer, got "
                    f"{kv_cache_memory_bytes!r}"
                )
        model = LLM(
            model=model_path,
            # skip_tokenizer_init=True,  # Disabled: Gemma requires tokenizer for model configuration
            enforce_eager=True,
            seed=self.seed,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            cpu_offload_gb=cpu_offload_gb,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            max_model_len=self.max_input_tokens + self.max_output_tokens,
            enable_chunked_prefill=False, # it's no-op in v1
            enable_prefix_caching=False,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            hf_overrides=hf_overrides,
            limit_mm_per_prompt={"image": 0},
            disable_hybrid_kv_cache_manager=True, # this fixes gemma3 which has sliding window cache and it impacts the way we handle kv cache
        )

        return model

    def _sampling_config(self, max_tokens: Optional[int] = None) -> Dict:
        from vllm import SamplingParams

        common_kwargs = {
            'max_tokens': self.max_output_tokens if max_tokens is None else int(max_tokens),
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
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        from vllm.inputs import TokensPrompt
        model_input = self.tokenizer.encode_for_generation(input_text, return_tensors=False)
        prompt = TokensPrompt(prompt_token_ids=model_input["input_ids"])

        if max_tokens is not None and max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        sampling_config = self._sampling_config(max_tokens=max_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        gpu_sampler = _GpuMemorySampler()
        gpu_sampler.start()
        start_time = time.perf_counter()
        try:
            request_outputs = self.model.generate(
                prompts=[prompt],
                **sampling_config,
            )
        finally:
            sampled_peak_memory = gpu_sampler.stop()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runtime_s = time.perf_counter() - start_time
        request_output = request_outputs[0]
        output_ids = request_output.outputs[0].token_ids

        request_metrics = getattr(request_output, "metrics", None)
        decode_latency_s = None
        decode_latency_source = None
        if request_metrics is not None:
            first = getattr(request_metrics, "first_token_time", None)
            last = getattr(request_metrics, "last_token_time", None)
            if last is None:
                last = getattr(request_metrics, "finished_time", None)
            if first is not None and last is not None:
                decode_latency_s = max(0.0, float(last - first))
                decode_latency_source = "request_first_to_last_token"
            elif getattr(request_metrics, "model_execute_time", None) is not None:
                decode_latency_s = float(request_metrics.model_execute_time)
                decode_latency_source = "model_execute_time_fallback"
        if decode_latency_s is None:
            # Older vLLM releases may omit RequestMetrics from returned
            # RequestOutput objects.  Keep the field populated and make the
            # unavoidable approximation explicit in the artifact.
            decode_latency_s = runtime_s
            decode_latency_source = "wall_runtime_fallback"
        peak_memory = None
        if torch.cuda.is_available():
            peak_memory = int(torch.cuda.max_memory_allocated())
        if sampled_peak_memory is not None:
            peak_memory = sampled_peak_memory

        decoded = self.tokenizer.decode(output_ids)

        output: dict[str, Any] = {
            'text': decoded[0] if isinstance(decoded, list) else decoded,
            'output_tokens_len': len(output_ids),
            'runtime_s': runtime_s,
            'decode_latency_s': decode_latency_s,
            'decode_latency_source': decode_latency_source,
            'peak_gpu_memory_bytes': peak_memory,
            'peak_gpu_memory_source': (
                'nvidia-smi' if sampled_peak_memory is not None else 'torch.cuda'
            ),
        }

        self.last_metrics = {
            'runtime_s': runtime_s,
            'decode_latency_s': decode_latency_s,
            'peak_gpu_memory_bytes': peak_memory,
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
        # Capture-only dense runs must preserve vLLM's profiling behavior. Sparse
        # methods retain the preallocation hook used by the custom handler.
        if os.getenv("SF_ATTENTION_NAME") in {None, "", "dense"}:
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
        ensure_attention_initialized_from_env(key)
        return output
    
    num_tokens = query.shape[0]
    is_prefilling = num_tokens > 1
    if _query_capture_enabled():
        _get_query_capture_collector().capture_attention(
            query, key, value, is_prefilling
        )

    if os.getenv("SF_ATTENTION_NAME") in {None, "", "dense"}:
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

    handler = get_attention_handler()
    if os.getenv("SF_ATTENTION_NAME") == "query_robust":
        expected_scale = query.shape[-1] ** -0.5
        backend_scale = float(getattr(self, "scale", expected_scale))
        if not abs(backend_scale - expected_scale) <= 1e-7 * expected_scale:
            raise RuntimeError(
                "Query-Robust v1 requires the backend attention scale to equal 1/sqrt(d)"
            )

    # Initialize tracking state at layer 0 (needed for all layer types including sliding window)
    if handler.current_layer == 0:
        if is_prefilling:
            handler._begin_prefill(prompt_len=num_tokens)
        else:
            if _sparse_chunked_prefill_requested():
                if attn_metadata.seq_lens.numel() != 1:
                    raise RuntimeError(
                        "Sparse chunked-prefill recovery currently supports one request; "
                        f"got seq_lens shape {tuple(attn_metadata.seq_lens.shape)}"
                    )
                handler._set_prompt_len_from_decode(
                    int(attn_metadata.seq_lens[0].item()) - 1
                )
            handler._begin_decode_step()

    sliding_window = getattr(self, "sliding_window", (-1, -1))

    if sliding_window != (-1, -1):
        if os.getenv("SF_ATTENTION_NAME") in {"shadowkv", "query_robust"}:
            raise RuntimeError(
                "The configured sparse method supports only full scalar-position attention; "
                f"sliding-window layer {sliding_window} is unsupported"
            )
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
        if is_prefilling and _sparse_chunked_prefill_requested():
            if os.getenv("SF_ATTENTION_NAME") == "shadowkv":
                raise RuntimeError(
                    "ShadowKV requires one full dense prefill to build its global "
                    "pre-RoPE SVD state; vLLM chunked prefill is unsupported without "
                    "the deferred/CPU staging path. Increase SF_MAX_NUM_BATCHED_TOKENS "
                    "or use a GPU configuration that fits the full prefill."
                )
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
        if is_prefilling and os.getenv("SF_ATTENTION_NAME") == "query_robust":
            # Query-Robust prefill is numerically dense. Build only its affine
            # router state here, then delegate attention and the canonical cache
            # write to vLLM's native paged FlashAttention forward. This avoids a
            # slower duplicate Python implementation of dense prefill.
            from sparse_frontier.modelling.attention.registry import get_attention

            attention = get_attention()
            attention.prefill_state(key, layer_idx=handler.current_layer)
            handler.tokens_per_layer_head[handler.current_layer] += num_tokens
            handler._maybe_report_prefill_sparsity(device=key.device)
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
        # Preserve vLLM's canonical paged cache semantics before the custom
        # handler reads it.  The handler may build temporary logical views,
        # but those must not replace the cache that vLLM passes on the next
        # decode step.
        if key is not None and value is not None:
            try:
                from vllm.attention.utils.fa_utils import reshape_and_cache_flash

                key_cache, value_cache = kv_cache.unbind(0)
                reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "The installed vLLM FlashAttention backend does not expose "
                    "the cache-write operation required by sparse attention"
                ) from exc
        handler(
            queries=query.contiguous(),
            keys=key.contiguous(),
            values=value.contiguous(),
            kv_cache=kv_cache.contiguous(),
            block_table=attn_metadata.block_table,
            output=output,
        )
        return output


def _shadowkv_rotary_embedding_call(self, *args, **kwargs):
    """Capture calibration Q or ShadowKV K before in-place RoPE."""
    positions = args[0] if len(args) > 0 else kwargs.get("positions")
    rope_forward = getattr(self, "_forward_method", None)
    if _query_capture_enabled():
        query = args[1] if len(args) > 1 else kwargs.get("query")
        if positions is None or query is None or rope_forward is None:
            raise RuntimeError(
                "query capture requires vLLM scalar RotaryEmbedding query inputs"
            )
        _get_query_capture_collector().capture_pre_rope(
            query, positions, rope_forward
        )

    if os.getenv("SF_ATTENTION_NAME") == "shadowkv":
        key = args[2] if len(args) > 2 else kwargs.get("key")
        if positions is None or key is None:
            raise RuntimeError(
                "ShadowKV requires a scalar-position RoPE call with a key tensor; "
                "the model's RoPE architecture is unsupported"
            )
        handler = get_attention_handler()
        if rope_forward is None:
            raise RuntimeError(
                "ShadowKV could not access vLLM's original RoPE dispatch method; "
                "the model's RoPE architecture is unsupported"
            )
        handler.capture_pre_rope(key, positions, rope_forward)

    if _ORIGINAL_ROTARY_EMBEDDING_CALL is None:
        raise RuntimeError("ShadowKV RoPE hook was invoked before initialization")
    result = _ORIGINAL_ROTARY_EMBEDDING_CALL(self, *args, **kwargs)
    if os.getenv("SF_ATTENTION_NAME") == "shadowkv":
        capture_rope_metadata = getattr(handler, "capture_rope_metadata", None)
        if capture_rope_metadata is not None:
            # vLLM may move/recast cos_sin_cache inside the original dispatch;
            # retain the current buffer rather than the pre-dispatch one.
            capture_rope_metadata(self)
    return result


def _patch_shadowkv_rope() -> None:
    global _ORIGINAL_ROTARY_EMBEDDING_CALL
    if _ORIGINAL_ROTARY_EMBEDDING_CALL is not None:
        return
    try:
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "ShadowKV requires vLLM's scalar RotaryEmbedding hook; "
            "the installed vLLM RoPE API is unsupported"
        ) from exc
    _ORIGINAL_ROTARY_EMBEDDING_CALL = RotaryEmbedding.__call__
    RotaryEmbedding.__call__ = _shadowkv_rotary_embedding_call


def swap_vllm_attention():
    is_attention_patch_enabled = os.getenv("SF_USE_ATTENTION_PATCH", "1") != "0"
    if not is_attention_patch_enabled:
        return

    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    global _ORIGINAL_FLASH_ATTENTION_FORWARD

    if _ORIGINAL_FLASH_ATTENTION_FORWARD is None:
        _ORIGINAL_FLASH_ATTENTION_FORWARD = FlashAttentionImpl.forward

    if os.getenv("SF_ATTENTION_NAME") == "shadowkv" or _query_capture_enabled():
        _patch_shadowkv_rope()

    FlashAttentionImpl.forward = vllm_patched_forward
