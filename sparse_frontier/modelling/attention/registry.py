from .efficient_prefilling import (
    DenseAttention,
    VerticalAndSlashAttentionMInference,
    BlockSparseAttentionMInference,
    FlexPrefill,
)
from .efficient_decoding import QuestAttention, TOVAAttention
from .kv_compression import SnapKVCompression, AdaSnapKVCompression
from .handler import AttentionHandler
import os
import json
import torch


ATTENTION_REGISTRY = {
    'dense': DenseAttention,
    'vertical_and_slash': VerticalAndSlashAttentionMInference,
    'block_sparse': BlockSparseAttentionMInference,
    'snapkv': SnapKVCompression,
    'ada_snapkv': AdaSnapKVCompression,
    'quest': QuestAttention,
    'tova': TOVAAttention,
    'flexprefill': FlexPrefill,
}

# Module-level singletons initialized from environment
_ATTENTION = None
_ATTN_HANDLER = None
_INIT_DONE = False


def ensure_attention_initialized_from_env(keys: torch.Tensor = None) -> None:
    global _ATTENTION, _ATTN_HANDLER, _INIT_DONE
    if _INIT_DONE and (_ATTENTION is not None) and (_ATTN_HANDLER is not None):
        return

    # Required env vars
    name = os.getenv('SF_ATTENTION_NAME')
    args_json = os.getenv('SF_ATTENTION_ARGS_JSON', '{}')
    tp_size = os.getenv('SF_TP_SIZE')
    num_q_heads = os.getenv('SF_MODEL_NUM_Q_HEADS')
    num_kv_heads = os.getenv('SF_MODEL_NUM_KV_HEADS')
    num_layers = os.getenv('SF_MODEL_NUM_LAYERS')
    max_input_tokens = os.getenv('SF_MAX_INPUT_TOKENS')
    max_output_tokens = os.getenv('SF_MAX_OUTPUT_TOKENS')
    kv_cache_block_size = os.getenv('SF_KV_CACHE_BLOCK_SIZE')

    missing = [
        k for k, v in [
            ('SF_ATTENTION_NAME', name),
            ('SF_TP_SIZE', tp_size),
            ('SF_MODEL_NUM_Q_HEADS', num_q_heads),
            ('SF_MODEL_NUM_KV_HEADS', num_kv_heads),
            ('SF_MODEL_NUM_LAYERS', num_layers),
            ('SF_MAX_INPUT_TOKENS', max_input_tokens),
            ('SF_MAX_OUTPUT_TOKENS', max_output_tokens),
            ('SF_KV_CACHE_BLOCK_SIZE', kv_cache_block_size),
        ] if v is None
    ]
    if missing:
        raise RuntimeError(f"Missing required env vars for sparse attention init: {', '.join(missing)}")

    try:
        attention_args = json.loads(args_json) if args_json else {}
    except Exception as e:
        raise RuntimeError(f"Failed to parse SF_ATTENTION_ARGS_JSON: {e}")

    # Derive block size (option a): quest uses page_size, else global kv block size
    if name == 'quest':
        if 'page_size' not in attention_args:
            raise RuntimeError("Quest attention requires 'page_size' in SF_ATTENTION_ARGS_JSON")
        block_size = int(attention_args['page_size'])
    else:
        block_size = int(kv_cache_block_size)

    # Build handler
    handler = AttentionHandler(
        tp_size=int(tp_size),
        model_q_heads=int(num_q_heads),
        model_kv_heads=int(num_kv_heads),
        model_layers=int(num_layers),
        max_input_tokens=int(max_input_tokens),
        max_output_tokens=int(max_output_tokens),
        block_size=block_size,
    )

    # Build attention
    if name not in ATTENTION_REGISTRY:
        raise RuntimeError(f"Unknown attention '{name}'. Available: {list(ATTENTION_REGISTRY.keys())}")

    extra_args = {
        'num_layers': int(num_layers),
        'max_input_tokens': int(max_input_tokens),
        'max_output_tokens': int(max_output_tokens),
    } if name == 'quest' else {}

    attention = ATTENTION_REGISTRY[name](**attention_args, **extra_args)

    _ATTENTION = attention
    _ATTN_HANDLER = handler
    _INIT_DONE = True

    # Pre-allocate memory if keys tensor provided (during profiling)
    if keys is not None:
        _ATTENTION.preallocate_memory(keys)

    print_attention_handler_config()


def get_attention():
    ensure_attention_initialized_from_env()
    return _ATTENTION


def get_attention_handler() -> AttentionHandler:
    ensure_attention_initialized_from_env()
    return _ATTN_HANDLER


def print_attention_handler_config() -> None:
    """Print a JSON summary of the initialized AttentionHandler configuration."""
    handler = _ATTN_HANDLER
    attention = _ATTENTION

    config = {
        "attention_impl": attention.__class__.__name__ if attention is not None else None,
        "tp_size": handler.tp_size,
        "model_q_heads": handler.model_q_heads,
        "model_kv_heads": handler.model_kv_heads,
        "q_heads_per_gpu": handler.q_heads_per_gpu,
        "kv_heads_per_gpu": handler.kv_heads_per_gpu,
        "q_heads_per_kv": handler.q_heads_per_kv,
        "model_layers": handler.model_layers,
        "max_seq_len": handler.max_seq_len,
        "max_blocks": handler.max_blocks,
        "block_size": handler.block_size,
    }

    print(json.dumps(config, indent=2, sort_keys=True))
