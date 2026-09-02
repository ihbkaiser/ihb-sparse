"""One-request real-vLLM smoke for the schema-2 Pile Query-Robust path."""

from __future__ import annotations

import json
import os


MODEL = "/workspace/.hf_home/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
POOL = "/dev/shm/pile_query_pool_20x2048_r3000_v8"


def main() -> None:
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_FLASH_ATTN_VERSION", "2")
    os.environ.setdefault("SF_GPU_MEMORY_UTILIZATION", "0.80")
    os.environ.setdefault("SF_MAX_NUM_BATCHED_TOKENS", "128")
    os.environ["SF_USE_ATTENTION_PATCH"] = "1"
    os.environ["SF_ATTENTION_NAME"] = "query_robust"
    os.environ["SF_ATTENTION_ARGS_JSON"] = json.dumps(
        {
            "token_budget": 1024,
            "chunk_size": 16,
            "generation_horizon": 1,
            "sink_chunks": 1,
            "recent_chunks": 1,
            "query_pool_path": POOL,
            "objective": "minimax",
            "initial_support": int(os.getenv("SF_QR_INITIAL_SUPPORT", "32")),
            "max_support": int(os.getenv("SF_QR_MAX_SUPPORT", "64")),
            "solver_gap_tolerance": float(os.getenv("SF_QR_TOLERANCE", "0.01")),
            "solver_max_iterations": int(os.getenv("SF_QR_MAX_ITERATIONS", "16")),
            "solver_chunk_batch_size": 16,
            "solver_violators_per_round": int(os.getenv("SF_QR_VIOLATORS_PER_ROUND", "4")),
            "summary_dtype": "bfloat16",
            "score_dtype": "float32",
        },
        sort_keys=True,
    )
    os.environ.update(
        {
            "SF_TP_SIZE": "1",
            "SF_MODEL_NUM_Q_HEADS": "32",
            "SF_MODEL_NUM_KV_HEADS": "8",
            "SF_MODEL_NUM_LAYERS": "32",
            "SF_MODEL_HEAD_DIM": "128",
            "SF_MAX_INPUT_TOKENS": "64",
            "SF_MAX_OUTPUT_TOKENS": "1",
            "SF_KV_CACHE_BLOCK_SIZE": "16",
            "SF_MODEL_ID": "NousResearch/Meta-Llama-3.1-8B-Instruct",
            "SF_MODEL_REVISION": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
            "SF_MODEL_ROPE_TYPE": "llama3",
            "SF_MODEL_ROPE_PARAMETERS_JSON": json.dumps(
                {
                    "factor": 8.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192,
                    "rope_theta": 500000.0,
                },
                sort_keys=True,
            ),
            "SF_ATTENTION_SCALE": str(128 ** -0.5),
        }
    )

    import torch
    from sparse_frontier.modelling.models.vllm_model import VLLMModel

    model = VLLMModel(
        model_path=MODEL,
        max_input_tokens=64,
        max_output_tokens=1,
        dtype=torch.bfloat16,
        tensor_parallel_size=1,
        seed=43,
    )
    prompt = (
        "Summarize this short calibration-style paragraph in one word: "
        "A frozen empirical query pool routes exact attention over selected chunks."
    )
    result = model.generate(prompt, max_tokens=1)
    from sparse_frontier.modelling.attention.registry import get_attention, get_attention_handler

    attention = get_attention()
    handler = get_attention_handler()
    result["query_robust_audit"] = {
        "objective": attention.objective,
        "summaries_built": attention.summaries_built,
        "summary_build_ms": attention.summary_build_ms,
        "solver_gap_max": attention.solver_gap_max,
        "solver_active_max": attention.solver_active_max,
        "solver_nonconverged": attention.solver_nonconverged,
        "quantized_error_max": attention.quantized_error_max,
        "fallback_count": attention.fallback_count,
        "dense_fallback": attention._dense_fallback,
        "decode_access_tokens": attention.last_accessed_tokens,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
