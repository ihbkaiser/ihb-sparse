#!/usr/bin/env python3
"""Evaluate sparse methods on the NVIDIA/kvpress RULER artifact.

The runner uses the same ``simonjegou/ruler`` dataset, context-length config
(``data_dir``), greedy generation, answer-prefix handling, and RULER scorer as
NVIDIA/kvpress.  It emits auditable raw, parsed, per-sample, and aggregate
artifacts.  Sparse-vLLM receives the exact tokenizer-produced prompt that the
kvpress pipeline constructs for each context/question pair.

The native engine currently compresses at the end of its prefill step, so the
question is included in that prefill.  This is an explicit runtime semantic
difference from kvpress' context-only cache press and is recorded in
``run_info.json``; it must not be confused with a byte-for-byte press match.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class EvalConfig:
    model_path: str
    output_dir: str
    dataset_name: str = "simonjegou/ruler"
    data_dir: str = "4096"
    dataset_path: str | None = None
    sparse_method: str = "quest"
    fraction: float = 1.0
    max_samples: int | None = None
    max_context_length: int | None = None
    max_new_tokens: int | None = None
    seed: int = 42
    device: str = "cuda:0"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.90
    batch_size: int = 4
    decode_graph: bool = False
    quest_chunk_size: int = 16
    decode_keep_tokens: int = 2048
    # Quest's native protocol exposes one query-aware token budget. Keep the
    # shared sink/recent regions empty so the effective budget is exactly 2048.
    sink_keep_tokens: int = 0
    recent_keep_tokens: int = 0
    shadowkv_sparse_budget: int = 2048
    shadowkv_rank: int = 160
    shadowkv_chunk_size: int = 8
    shadowkv_recent_tokens: int = 512
    shadowkv_svd_batch_size: int = 4
    shadowkv_svd_method: str = "exact"
    shadowkv_svd_oversample: int = 16
    shadowkv_svd_niter: int = 2
    shadowkv_kernel_backend: str = "auto"
    shadowkv_cutlass_root: str | None = None
    shadowkv_decode_backend: str = "auto"
    shadowkv_flashinfer_backend: str = "auto"
    # ShadowKV's paper protocol stores the value cache in the CPU shadow. Keep
    # the GPU-cache throughput overlay explicit at the command line.
    shadowkv_storage: str = "cpu"
    shadowkv_gpu_cache_tokens: int = 0
    shadowkv_multistream_gather: bool = True
    shadowkv_gather_copy_with_offsets: bool = True
    query_robust_vertices_path: str | None = None
    query_robust_num_vertices: int = 8
    query_robust_chunk_size: int = 16
    query_robust_solver_iters: int = 24
    query_robust_solver_lr: float = 0.25
    query_robust_score_alpha: float = 0.5
    query_robust_skip_layers: int = 0
    query_robust_uniform_p: bool = False


def _parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="simonjegou/ruler")
    parser.add_argument("--data-dir", default="4096")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument(
        "--sparse-method",
        choices=("quest", "shadowkv", "query_robust"),
        required=True,
    )
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of visible GPU memory reserved for the inference runtime (0, 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum requests decoded concurrently in one engine batch.",
    )
    parser.add_argument(
        "--decode-graph",
        action="store_true",
        help="Capture/replay fixed-shape decode CUDA Graphs.",
    )
    parser.add_argument("--quest-chunk-size", type=int, default=16)
    parser.add_argument("--decode-keep-tokens", type=int, default=2048)
    parser.add_argument("--sink-keep-tokens", type=int, default=0)
    parser.add_argument("--recent-keep-tokens", type=int, default=0)
    parser.add_argument("--shadowkv-sparse-budget", type=int, default=2048)
    parser.add_argument("--shadowkv-rank", type=int, default=160)
    parser.add_argument("--shadowkv-chunk-size", type=int, default=8)
    parser.add_argument("--shadowkv-recent-tokens", type=int, default=512)
    parser.add_argument("--shadowkv-svd-batch-size", type=int, default=4)
    parser.add_argument(
        "--shadowkv-svd-method", choices=("exact", "lowrank"), default="exact"
    )
    parser.add_argument("--shadowkv-svd-oversample", type=int, default=16)
    parser.add_argument("--shadowkv-svd-niter", type=int, default=2)
    parser.add_argument(
        "--shadowkv-kernel-backend",
        choices=("auto", "torch", "cutlass"),
        default="auto",
    )
    parser.add_argument("--shadowkv-cutlass-root", default=None)
    parser.add_argument(
        "--shadowkv-decode-backend",
        choices=("auto", "flashinfer", "triton"),
        default="auto",
        help="ShadowKV per-head decode provider; use triton for the graph ablation.",
    )
    parser.add_argument(
        "--shadowkv-flashinfer-backend",
        choices=("auto", "fa2", "fa3", "cute-dsl"),
        default="auto",
        help=(
            "FlashInfer kernel backend for ShadowKV; auto selects CuTe DSL on "
            "Blackwell for dynamic per-head lengths."
        ),
    )
    parser.add_argument(
        "--shadowkv-storage", choices=("cpu", "gpu_cache"), default="cpu"
    )
    parser.add_argument(
        "--shadowkv-gpu-cache-tokens",
        type=int,
        default=0,
        help="GPU-cache capacity in tokens; 0 derives it from max_model_len.",
    )
    parser.add_argument(
        "--shadowkv-multistream-gather",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlap eager host-value gather with key reconstruction when possible.",
    )
    parser.add_argument(
        "--shadowkv-gather-copy-with-offsets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse selected value chunks with ShadowKV's offset-copy kernel.",
    )
    parser.add_argument("--query-robust-vertices-path", default=None)
    parser.add_argument("--query-robust-num-vertices", type=int, default=8)
    parser.add_argument("--query-robust-chunk-size", type=int, default=16)
    parser.add_argument("--query-robust-solver-iters", type=int, default=24)
    parser.add_argument("--query-robust-solver-lr", type=float, default=0.25)
    parser.add_argument("--query-robust-score-alpha", type=float, default=0.5)
    parser.add_argument("--query-robust-skip-layers", type=int, default=0)
    parser.add_argument(
        "--query-robust-uniform-p",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1].")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when set.")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive when set.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.shadowkv_svd_batch_size <= 0:
        raise ValueError("--shadowkv-svd-batch-size must be positive.")
    if args.shadowkv_svd_oversample <= 0:
        raise ValueError("--shadowkv-svd-oversample must be positive.")
    if args.shadowkv_svd_niter <= 0:
        raise ValueError("--shadowkv-svd-niter must be positive.")
    if args.shadowkv_gpu_cache_tokens < 0:
        raise ValueError("--shadowkv-gpu-cache-tokens must be non-negative.")
    if args.sparse_method == "query_robust":
        if not args.query_robust_vertices_path:
            raise ValueError(
                "--query-robust-vertices-path is required for --sparse-method query_robust."
            )
        if args.query_robust_num_vertices < 2:
            raise ValueError("--query-robust-num-vertices must be at least 2.")
        if args.query_robust_chunk_size <= 0:
            raise ValueError("--query-robust-chunk-size must be positive.")
        if args.query_robust_solver_iters <= 0:
            raise ValueError("--query-robust-solver-iters must be positive.")
        if args.query_robust_solver_lr <= 0:
            raise ValueError("--query-robust-solver-lr must be positive.")
        if args.query_robust_score_alpha not in (0.0, 0.5, 1.0):
            raise ValueError(
                "--query-robust-score-alpha must be one of 0, 0.5, or 1."
            )
        if args.query_robust_skip_layers < 0:
            raise ValueError("--query-robust-skip-layers must be non-negative.")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1).")
    return EvalConfig(**vars(args))


def _load_rows(config: EvalConfig) -> list[dict[str, Any]]:
    if config.dataset_path:
        path = Path(config.dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"RULER dataset path does not exist: {path}")
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        elif path.suffix == ".json":
            rows = json.loads(path.read_text())
        else:
            raise ValueError("--dataset-path must point to a .json or .jsonl file.")
    else:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "The kvpress RULER runner requires the 'datasets' package, or use --dataset-path."
            ) from error
        rows = load_dataset(
            config.dataset_name,
            data_dir=config.data_dir or None,
            split="test",
        )
        rows = [dict(row) for row in rows]
    if not isinstance(rows, list) or not rows:
        raise ValueError("RULER dataset is empty or not a list of rows.")
    required = {"context", "question", "answer_prefix", "answer", "task", "max_new_tokens"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"RULER dataset is missing required columns: {missing}")
    rng = random.Random(config.seed)
    rows = list(rows)
    if config.fraction < 1.0:
        count = max(1, int(len(rows) * config.fraction))
        rows = [rows[index] for index in sorted(rng.sample(range(len(rows)), count))]
    if config.max_samples is not None:
        rows = rows[: config.max_samples]
    return rows


def _references(value: Any, *, task: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"RULER row task={task!r} has no answer references.")
    return [str(reference) for reference in value]


def _resolve_max_new_tokens(
    rows: list[dict[str, Any]], *, override: int | None
) -> list[int]:
    """Resolve the RULER generation budget independently for every row."""
    resolved: list[int] = []
    for index, row in enumerate(rows):
        task = str(row.get("task", "<unknown>"))
        raw_value = override if override is not None else row["max_new_tokens"]
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as error:
            source = "--max-new-tokens" if override is not None else "row['max_new_tokens']"
            raise ValueError(
                f"RULER {source} must be an integer at row {index} task={task!r}; "
                f"got {raw_value!r}."
            ) from error
        if value <= 0:
            source = "--max-new-tokens" if override is not None else "row['max_new_tokens']"
            raise ValueError(
                f"RULER {source} must be positive at row {index} task={task!r}; "
                f"got {value}."
            )
        resolved.append(value)
    return resolved


def _kvpress_prompt(
    tokenizer,
    context: str,
    question: str,
    answer_prefix: str,
    *,
    max_context_length: int,
) -> list[int]:
    """Reproduce KVPress pipeline.py's tokenizer-visible prompt construction."""
    if tokenizer.chat_template is None:
        context_text = getattr(tokenizer, "bos_token", "") + context
        question_suffix = "\n"
    else:
        separator = "#" * (len(context) + 10)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": context + separator}],
            add_generation_prompt=True,
            tokenize=False,
        )
        context_text, question_suffix = rendered.split(separator)
    # kvpress tokenizes these two pieces independently and concatenates the
    # resulting tensors.  Encoding the combined string can merge BPE tokens at
    # the boundary and is not protocol-equivalent.
    context_ids = tokenizer.encode(context_text, add_special_tokens=False)
    context_ids = context_ids[:max_context_length]
    question_ids = tokenizer.encode(
        question + question_suffix + answer_prefix,
        add_special_tokens=False,
    )
    return context_ids + question_ids


def _score(task: str, prediction: str, references: list[str]) -> float:
    normalized = re.sub(r"[\x00-\x1f]", "", prediction.strip()).lower()
    if not references:
        raise ValueError(f"RULER row task={task!r} has no references.")
    if task.split("_")[0] == "qa":
        return max(float(reference.lower() in normalized) for reference in references)
    return sum(float(reference.lower() in normalized) for reference in references) / len(references)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_infer_config(
    config: EvalConfig, *, resolved_max_model_len: int
) -> dict[str, Any]:
    infer_config: dict[str, Any] = {
        "max_model_len": resolved_max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "tensor_parallel_size": 1,
        "max_num_seqs_in_batch": config.batch_size,
        "max_decoding_seqs": config.batch_size,
        "max_num_seqs_in_gpu": config.batch_size,
        "decode_graph": config.decode_graph,
        "decode_graph_capture_sizes": [config.batch_size],
        "enable_prefix_caching": False,
    }
    if config.sparse_method == "quest":
        infer_config.update(
            {
                "quest_chunk_size": config.quest_chunk_size,
                "decode_keep_tokens": config.decode_keep_tokens,
                "sink_keep_tokens": config.sink_keep_tokens,
                "recent_keep_tokens": config.recent_keep_tokens,
            }
        )
    elif config.sparse_method == "shadowkv":
        infer_config.update(
            {
                "shadowkv_sparse_budget": config.shadowkv_sparse_budget,
                "shadowkv_rank": config.shadowkv_rank,
                "shadowkv_chunk_size": config.shadowkv_chunk_size,
                "shadowkv_recent_tokens": config.shadowkv_recent_tokens,
                "shadowkv_svd_batch_size": config.shadowkv_svd_batch_size,
                "shadowkv_svd_method": config.shadowkv_svd_method,
                "shadowkv_svd_oversample": config.shadowkv_svd_oversample,
                "shadowkv_svd_niter": config.shadowkv_svd_niter,
                "shadowkv_kernel_backend": config.shadowkv_kernel_backend,
                "shadowkv_cutlass_root": config.shadowkv_cutlass_root,
                "shadowkv_decode_backend": config.shadowkv_decode_backend,
                "shadowkv_flashinfer_backend": config.shadowkv_flashinfer_backend,
                "shadowkv_storage": config.shadowkv_storage,
                "shadowkv_gpu_cache_tokens": config.shadowkv_gpu_cache_tokens,
                "shadowkv_multistream_gather": config.shadowkv_multistream_gather,
                "shadowkv_gather_copy_with_offsets": config.shadowkv_gather_copy_with_offsets,
            }
        )
    elif config.sparse_method == "query_robust":
        infer_config.update(
            {
                "query_robust_vertices_path": config.query_robust_vertices_path,
                "query_robust_num_vertices": config.query_robust_num_vertices,
                "query_robust_chunk_size": config.query_robust_chunk_size,
                "query_robust_solver_iters": config.query_robust_solver_iters,
                "query_robust_solver_lr": config.query_robust_solver_lr,
                "query_robust_score_alpha": config.query_robust_score_alpha,
                "query_robust_skip_layers": config.query_robust_skip_layers,
                "query_robust_uniform_p": config.query_robust_uniform_p,
                "sink_keep_tokens": config.sink_keep_tokens,
                "decode_keep_tokens": config.decode_keep_tokens,
                "recent_keep_tokens": config.recent_keep_tokens,
            }
        )
    else:
        raise ValueError(f"Unsupported sparse method: {config.sparse_method!r}")
    return infer_config


def main() -> None:
    config = _parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    from transformers import AutoTokenizer
    from benchmark.model_adapters.sparsevllm import get_sparsevllm_generate_api

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=False)
    prompts: list[list[int]] = []
    max_new_tokens = _resolve_max_new_tokens(rows, override=config.max_new_tokens)
    for row in rows:
        context = str(row["context"])
        max_context_length = config.max_context_length
        if max_context_length is None:
            max_context_length = min(int(tokenizer.model_max_length), int(1e10))
        prompts.append(
            _kvpress_prompt(
                tokenizer,
                context,
                str(row["question"]),
                str(row["answer_prefix"]),
                max_context_length=max_context_length,
            )
        )
    inferred_max_model_len = max(len(prompt) + tokens for prompt, tokens in zip(prompts, max_new_tokens)) + 16
    if config.max_model_len is not None and config.max_model_len < inferred_max_model_len:
        raise ValueError(
            f"--max-model-len={config.max_model_len} is smaller than the evaluated prompt budget "
            f"{inferred_max_model_len}."
        )
    resolved_max_model_len = config.max_model_len or inferred_max_model_len
    if (
        config.sparse_method == "shadowkv"
        and config.shadowkv_storage == "gpu_cache"
        and config.shadowkv_gpu_cache_tokens == 0
    ):
        # GPU-cache mode needs a fixed allocation before the first prefill.
        # Deriving it here keeps the fast profile usable without a duplicated
        # max-model-len flag while still recording the effective capacity.
        config.shadowkv_gpu_cache_tokens = int(resolved_max_model_len)
    infer_config = _build_infer_config(
        config,
        resolved_max_model_len=resolved_max_model_len,
    )
    derived_config = {
        "target_context_length": int(config.data_dir)
        if config.data_dir is not None and str(config.data_dir).isdigit()
        else None,
    }
    if config.sparse_method == "quest":
        # The shared config names are retained for other sparse methods, but
        # Quest's native protocol is a single query-aware token budget.
        derived_config["quest_effective_token_budget"] = (
            config.sink_keep_tokens
            + config.decode_keep_tokens
            + config.recent_keep_tokens
        )
    elif config.sparse_method == "shadowkv":
        from sparsevllm.configs.sparse import resolve_shadowkv_outlier_chunks

        derived_config["shadowkv_outlier_chunks"] = resolve_shadowkv_outlier_chunks(
            config.shadowkv_sparse_budget
        )
    else:
        derived_config["query_robust_effective_token_budget"] = (
            config.sink_keep_tokens
            + config.decode_keep_tokens
            + config.recent_keep_tokens
        )
    run_info = {
        "protocol": "NVIDIA/kvpress evaluation/evaluate.py RULER dataset and scorer",
        "protocol_source": "https://github.com/NVIDIA/kvpress/tree/main/evaluation",
        "config": asdict(config),
        "infer_config": infer_config,
        "derived_config": derived_config,
        "max_new_tokens_source": (
            "cli_override" if config.max_new_tokens is not None else "dataset_row"
        ),
        "dataset_rows": len(rows),
        "prompt_semantics": "kvpress tokenizer-visible context + question + answer_prefix",
        "native_semantic_difference": "Sparse-vLLM compresses after its combined prefill; kvpress presses context before question decoding.",
        "device": config.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": int(config.batch_size),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    started = time.time()
    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    try:
        if config.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"Requested {config.device}, but CUDA is unavailable.")
            torch.cuda.set_device(torch.device(config.device))
        generate = get_sparsevllm_generate_api(
            config.model_path,
            infer_config,
            sparse_method=config.sparse_method,
        )
        llm = getattr(generate, "_sparsevllm_llm", None)
        from sparsevllm.operators.registry import operator_binding_reports

        runtime_start = {
            "operator_bindings": operator_binding_reports(),
            "cuda_graph_requested": bool(config.decode_graph),
        }
        (output_dir / "runtime_bindings.json").write_text(
            json.dumps(runtime_start, indent=2, default=str), encoding="utf-8"
        )
        outputs = generate(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        if not isinstance(outputs, list) or len(outputs) != len(rows):
            raise RuntimeError(f"Native generation returned {len(outputs) if isinstance(outputs, list) else type(outputs)} outputs for {len(rows)} rows.")
        for index, (row, prediction) in enumerate(zip(rows, outputs)):
            prediction = str(prediction)
            raw_rows.append({"index": index, "prediction": prediction})
            parsed_rows.append({"index": index, "predicted_answer": prediction.strip()})
            result_rows.append(
                {
                    "index": index,
                    "task": str(row["task"]),
                    "status": "success",
                    "predicted_answer": prediction,
                    "answer": _references(row["answer"], task=str(row["task"])),
                    "score": _score(
                        str(row["task"]),
                        prediction,
                        _references(row["answer"], task=str(row["task"])),
                    ),
                    "input_token_count": len(prompts[index]),
                    "max_new_tokens": max_new_tokens[index],
                    "output_token_count": len(
                        tokenizer.encode(prediction, add_special_tokens=False)
                    ),
                }
            )
        runtime_diagnostics: dict[str, Any] = {
            "operator_bindings": operator_binding_reports(),
            "cuda": {
                "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "device_capability": list(torch.cuda.get_device_capability()),
                "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
            },
        }
        if llm is not None:
            runtime_diagnostics["operator_runtime_stats"] = llm.operator_runtime_stats()
            runtime_diagnostics["sparse_state_summaries"] = llm.debug_sparse_state_summaries()
        (output_dir / "runtime_diagnostics.json").write_text(
            json.dumps(runtime_diagnostics, indent=2, default=str), encoding="utf-8"
        )
    except Exception as error:
        # Preserve an explicit terminal status for every evaluated sample.
        for index, row in enumerate(rows):
            result_rows.append(
                {
                    "index": index,
                    "task": str(row["task"]),
                    "status": "model_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "answer": _references(row["answer"], task=str(row["task"])),
                    "input_token_count": len(prompts[index]),
                    "max_new_tokens": max_new_tokens[index],
                }
            )
        (output_dir / "error.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2),
            encoding="utf-8",
        )
    _write_jsonl(output_dir / "raw_outputs.jsonl", raw_rows)
    _write_jsonl(output_dir / "parsed_outputs.jsonl", parsed_rows)
    _write_jsonl(output_dir / "per_sample_results.jsonl", result_rows)
    successful = [row for row in result_rows if row["status"] == "success"]
    by_task: dict[str, float] = {}
    for task in sorted({str(row["task"]) for row in result_rows}):
        values = [float(row["score"]) for row in successful if row["task"] == task]
        by_task[task] = round(100.0 * float(np.mean(values)), 2) if values else None
    aggregate = {
        "protocol": run_info["protocol"],
        "sparse_method": config.sparse_method,
        "status": "success" if len(successful) == len(rows) else "failed",
        "num_samples": len(rows),
        "num_success": len(successful),
        "overall_score": round(100.0 * float(np.mean([row["score"] for row in successful])), 2) if successful else None,
        "score_by_task": by_task,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    aggregate["input_tokens"] = sum(
        int(row["input_token_count"]) for row in successful
    )
    aggregate["requested_output_tokens"] = sum(
        int(row["max_new_tokens"]) for row in successful
    )
    aggregate["generated_output_tokens"] = sum(
        int(row["output_token_count"]) for row in successful
    )
    if aggregate["elapsed_seconds"] > 0:
        aggregate["samples_per_second"] = round(
            len(successful) / aggregate["elapsed_seconds"], 4
        )
        aggregate["generated_output_tokens_per_second"] = round(
            aggregate["generated_output_tokens"] / aggregate["elapsed_seconds"], 4
        )
        aggregate["requested_output_tokens_per_second"] = round(
            aggregate["requested_output_tokens"] / aggregate["elapsed_seconds"], 4
        )
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    if aggregate["status"] != "success":
        raise RuntimeError("RULER evaluation failed; inspect per_sample_results.jsonl and error.json.")


if __name__ == "__main__":
    main()
