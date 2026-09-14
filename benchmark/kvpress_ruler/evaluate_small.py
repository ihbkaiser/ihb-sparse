#!/usr/bin/env python3
"""Evaluate sparse methods on ShadowKV's local per-task RULER artifact.

This variant accepts the offline layout produced by ShadowKV's RULER builder:
``<root>/<task>/validation.jsonl`` with rows containing ``input``, ``outputs``,
``index``, and ``length``.  It normalizes those rows into the same internal
prompt/answer representation used by the main evaluator, then reuses its
greedy generation, scoring, and auditable output artifacts.

Example for the local 128K Qwen artifact:

    python benchmark/kvpress_ruler/evaluate_small.py \
      --model-path /path/to/Qwen3-4B-Instruct-2507 \
      --sparse-method query_robust \
      --dataset-path /workspace/b200_sparse_attention_128k/source/ShadowKV/data/ruler/data/qwen/131072 \
      --query-robust-vertices-path /path/to/qwen3_qr_vertices.pt \
      --sink-keep-tokens 32 \
      --decode-keep-tokens 4096 \
      --recent-keep-tokens 256 \
      --output-dir results/qr-qwen3-131072

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
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


LOCAL_TASK_MAX_NEW_TOKENS = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multivalue": 128,
    "niah_multiquery": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}


@dataclass(frozen=True)
class EvalConfig:
    model_path: str
    output_dir: str
    dataset_name: str = "simonjegou/ruler"
    data_dir: str = "4096"
    dataset_path: str | None = None
    dataset_repo_id: str | None = None
    dataset_revision: str = "main"
    dataset_file_template: str = "ruler-{data_dir}.jsonl"
    sparse_method: str = "quest"
    fraction: float = 1.0
    max_samples: int | None = None
    max_context_length: int | None = None
    max_new_tokens: int | None = None
    seed: int = 42
    device: str = "cuda:0"
    gpu_ids: tuple[int, ...] | None = None
    # Internal worker controls used by --gpu-ids orchestration.  A worker
    # receives the globally selected rows whose ordinal satisfies
    # ordinal % shard_count == shard_index.
    shard_index: int = 0
    shard_count: int = 1
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.90
    batch_size: int = 4
    max_batched_tokens: int = 65536
    decode_graph: bool = False
    quest_chunk_size: int = 16
    # Matched Quest/Query-Robust RULER protocol: 32 sink tokens, up to 4096
    # query-selected middle tokens, and 256 recent tokens.
    decode_keep_tokens: int = 4096
    sink_keep_tokens: int = 32
    recent_keep_tokens: int = 256
    shadowkv_sparse_budget: int = 2048
    shadowkv_rank: int = 160
    shadowkv_chunk_size: int = 8
    # ShadowKV paper convention: 4 local chunks at chunk_size=8 means
    # 32 exact local tokens.  Keep both controls explicit for reproducible
    # budget-matched evaluations.
    shadowkv_local_chunks: int = 4
    shadowkv_outlier_chunks: int | None = None
    shadowkv_recent_tokens: int = 512
    shadowkv_svd_batch_size: int = 4
    shadowkv_svd_method: str = "exact"
    shadowkv_svd_oversample: int = 16
    shadowkv_svd_niter: int = 2
    shadowkv_kernel_backend: str = "auto"
    shadowkv_cutlass_root: str | None = None
    shadowkv_decode_backend: str = "auto"
    shadowkv_flashinfer_backend: str = "auto"
    # Prefer the fast GPU-cache profile for normal evaluation. The CPU-shadow
    # implementation remains available as an explicit ablation via the CLI.
    shadowkv_storage: str = "gpu_cache"
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
    query_robust_model_fingerprint: str | None = None
    query_robust_uniform_p: bool = False


def _phase_log(message: str) -> None:
    """Print a flushed phase marker before long preprocessing/runtime steps."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[RULER {timestamp}] {message}", flush=True)


def _evaluation_timing(
    *,
    started_at: float,
    serving_started_at: float | None,
    finished_at: float,
) -> dict[str, float | None]:
    """Return auditable cold-start and warm-serving timing intervals.

    ``serving_started_at`` is recorded after engine construction returns,
    which is after model loading, startup profiling, graph capture, and
    warmup. Successful throughput metrics must use that interval instead of
    the evaluation lifetime.
    """
    started_at = float(started_at)
    finished_at = float(finished_at)
    if finished_at < started_at:
        raise ValueError("finished_at must be greater than or equal to started_at.")
    total_elapsed = finished_at - started_at
    if serving_started_at is None:
        return {
            "startup_seconds": None,
            "serving_elapsed_seconds": None,
            "elapsed_seconds": round(total_elapsed, 3),
            "total_elapsed_seconds": round(total_elapsed, 3),
        }

    serving_started_at = float(serving_started_at)
    if serving_started_at < started_at:
        raise ValueError("serving_started_at must be greater than or equal to started_at.")
    if finished_at < serving_started_at:
        raise ValueError("finished_at must be greater than or equal to serving_started_at.")
    startup_elapsed = serving_started_at - started_at
    serving_elapsed = finished_at - serving_started_at
    return {
        "startup_seconds": round(startup_elapsed, 3),
        "serving_elapsed_seconds": round(serving_elapsed, 3),
        # Preserve the historical field name as the primary warm-serving interval.
        "elapsed_seconds": round(serving_elapsed, 3),
        "total_elapsed_seconds": round(total_elapsed, 3),
    }


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    """Parse physical CUDA IDs supplied to the multi-worker launcher."""
    values = [part.strip() for part in str(value).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(
            "--gpu-ids must be a comma-separated list such as 0 or 0,1."
        )
    try:
        gpu_ids = tuple(int(part) for part in values)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--gpu-ids must contain only non-negative integer IDs."
        ) from error
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError(
            "--gpu-ids must contain only non-negative integer IDs."
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("--gpu-ids must not contain duplicates.")
    return gpu_ids


def _parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="simonjegou/ruler")
    parser.add_argument("--data-dir", default="131072")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Local .json/.jsonl file or a directory containing one "
            "<task>/validation.jsonl file per RULER task."
        ),
    )
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--dataset-file-template",
        default="ruler-{data_dir}.jsonl",
        help="Filename template used with --dataset-repo-id.",
    )
    parser.add_argument(
        "--sparse-method",
        choices=("vanilla", "quest", "shadowkv", "query_robust"),
        required=True,
    )
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the official per-task generation cap for every local row.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gpu-ids",
        type=_parse_gpu_ids,
        default=None,
        help=(
            "Run independent TP1 workers on these physical GPUs. For example, "
            "--gpu-ids 0,1 shards the selected RULER rows across two workers."
        ),
    )
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, default=1, help=argparse.SUPPRESS)
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
        "--max-batched-tokens",
        type=int,
        default=65536,
        help="Maximum tokens scheduled in one prefill step.",
    )
    parser.add_argument(
        "--decode-graph",
        action="store_true",
        help="Capture/replay fixed-shape decode CUDA Graphs.",
    )
    parser.add_argument("--quest-chunk-size", type=int, default=16)
    parser.add_argument("--decode-keep-tokens", type=int, default=4096)
    parser.add_argument("--sink-keep-tokens", type=int, default=32)
    parser.add_argument("--recent-keep-tokens", type=int, default=256)
    parser.add_argument("--shadowkv-sparse-budget", type=int, default=2048)
    parser.add_argument("--shadowkv-rank", type=int, default=160)
    parser.add_argument("--shadowkv-chunk-size", type=int, default=8)
    parser.add_argument(
        "--shadowkv-local-chunks",
        type=int,
        default=4,
        help="Exact local tail in chunks; 4 chunks equals 32 tokens with chunk_size=8.",
    )
    parser.add_argument(
        "--shadowkv-outlier-chunks",
        type=int,
        default=None,
        help="Explicit per-KV-head outlier chunk count; otherwise derive from budget.",
    )
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
        "--shadowkv-storage",
        choices=("cpu", "gpu_cache"),
        default="gpu_cache",
        help="ShadowKV storage mode; gpu_cache is the default fast path, cpu is opt-in.",
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
    parser.add_argument("--query-robust-model-fingerprint", default=None)
    parser.add_argument(
        "--query-robust-uniform-p",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.dataset_path and args.dataset_repo_id:
        raise ValueError("Use only one of --dataset-path and --dataset-repo-id.")
    try:
        filename = args.dataset_file_template.format(data_dir=args.data_dir)
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(
            "--dataset-file-template must be format-compatible with {data_dir}."
        ) from error
    if args.dataset_repo_id and (
        not filename or filename.startswith("/") or ".." in Path(filename).parts
    ):
        raise ValueError(
            "--dataset-file-template must resolve to a relative file path without '..'."
        )
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1].")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when set.")
    if args.max_context_length is not None and args.max_context_length <= 0:
        raise ValueError("--max-context-length must be positive when set.")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive when set.")
    if args.max_model_len is not None and args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive when set.")
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError(
            "--shard-index must be in [0, --shard-count): "
            f"got index={args.shard_index} count={args.shard_count}."
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_batched_tokens <= 0:
        raise ValueError("--max-batched-tokens must be positive.")
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


def _iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid RULER JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"RULER row at {path}:{line_number} must be an object; "
                    f"got {type(row).__name__}."
                )
            yield row


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"RULER dataset path does not exist: {path}")
    if path.suffix == ".jsonl":
        return list(_iter_jsonl_rows(path))
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"RULER JSON dataset must contain a list: {path}")
        if not all(isinstance(row, dict) for row in value):
            raise ValueError(f"Every RULER JSON row must be an object: {path}")
        return [dict(row) for row in value]
    raise ValueError("RULER dataset must be a .json or .jsonl file.")


def _local_validation_files(path: Path) -> list[Path]:
    """Return deterministic local ShadowKV validation files for a path."""
    if path.is_file():
        if path.suffix != ".jsonl":
            raise ValueError(
                "ShadowKV local RULER data must be a validation .jsonl file "
                f"or a directory: {path}"
            )
        files = [path]
    elif path.is_dir():
        files = sorted(
            path.rglob("validation.jsonl"),
            key=lambda candidate: candidate.relative_to(path).as_posix(),
        )
    else:
        raise FileNotFoundError(f"Local RULER dataset path does not exist: {path}")
    if not files:
        raise FileNotFoundError(
            f"No validation.jsonl files found under local RULER path: {path}"
        )
    return files


def _local_directory_row_factory(
    path: Path,
) -> tuple[Callable[[], Iterator[dict[str, Any]]], str]:
    """Normalize ShadowKV's ``input``/``outputs`` task files for this runner."""
    path = Path(path)
    files = _local_validation_files(path)
    relative_root = path if path.is_dir() else path.parent
    source = str(path)

    def factory() -> Iterator[dict[str, Any]]:
        output_index = 0
        for validation_path in files:
            task = validation_path.parent.name
            if task not in LOCAL_TASK_MAX_NEW_TOKENS:
                raise ValueError(
                    f"Unknown local RULER task directory {task!r} in {validation_path}. "
                    f"Expected one of {sorted(LOCAL_TASK_MAX_NEW_TOKENS)}."
                )
            source_file = validation_path.relative_to(relative_root).as_posix()
            for line_number, row in enumerate(
                _iter_jsonl_rows(validation_path), start=1
            ):
                source_prefix = f"{validation_path}:{line_number}"
                input_text = row.get("input")
                if not isinstance(input_text, str) or not input_text:
                    raise ValueError(
                        f"Local RULER row {source_prefix} must have non-empty string input."
                    )
                outputs = row.get("outputs")
                if (
                    not isinstance(outputs, (list, tuple))
                    or not outputs
                    or not all(isinstance(value, str) and value for value in outputs)
                ):
                    raise ValueError(
                        f"Local RULER row {source_prefix} must have a non-empty "
                        "outputs list of strings."
                    )
                raw_index = row.get("index")
                raw_length = row.get("length")
                if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                    raise ValueError(
                        f"Local RULER row {source_prefix} has invalid integer index: "
                        f"{raw_index!r}."
                    )
                if isinstance(raw_length, bool) or not isinstance(raw_length, int):
                    raise ValueError(
                        f"Local RULER row {source_prefix} has invalid integer length: "
                        f"{raw_length!r}."
                    )
                if raw_index < 0 or raw_length <= 0:
                    raise ValueError(
                        f"Local RULER row {source_prefix} has out-of-range index/length: "
                        f"index={raw_index} length={raw_length}."
                    )
                normalized = dict(row)
                normalized.update(
                    {
                        "prompt": input_text,
                        "answer_prefix": "",
                        "answer": list(outputs),
                        "task": task,
                        "max_new_tokens": LOCAL_TASK_MAX_NEW_TOKENS[task],
                        "source_index": raw_index,
                        "source_length": raw_length,
                        "source_file": source_file,
                        "_sparsevllm_global_index": output_index,
                    }
                )
                output_index += 1
                yield normalized

    return factory, source


def _is_local_shadowkv_file(path: Path) -> bool:
    """Detect the ShadowKV row schema without changing normal kvpress files."""
    try:
        first_row = next(_iter_jsonl_rows(path))
    except StopIteration:
        return False
    return "input" in first_row and "outputs" in first_row


def _download_hf_dataset_file(
    *, repo_id: str, filename: str, revision: str
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "--dataset-repo-id requires the 'huggingface_hub' package."
        ) from error
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )


def _validate_dataset_row(
    row: dict[str, Any], *, index: int, source: str, schema: str
) -> None:
    legacy_required = {
        "context",
        "question",
        "answer_prefix",
        "answer",
        "task",
        "max_new_tokens",
    }
    prompt_required = {"prompt", "answer_prefix", "answer", "task", "max_new_tokens"}
    required = prompt_required if schema == "prompt" else legacy_required
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(
            f"RULER dataset row {index} from {source} is missing columns: {missing}"
        )
    if schema == "prompt" and "prompt" not in row:
        raise ValueError(
            f"RULER dataset mixes prompt and context/question schemas at row {index}."
        )
    if schema == "context_question" and "prompt" in row:
        raise ValueError(
            f"RULER dataset mixes context/question and prompt schemas at row {index}."
        )
    text_fields = ("prompt", "answer_prefix") if schema == "prompt" else (
        "context",
        "question",
        "answer_prefix",
    )
    for field in text_fields:
        if not isinstance(row[field], str):
            raise ValueError(
                f"RULER dataset row {index} field {field!r} from {source} "
                f"must be a string; got {type(row[field]).__name__}."
            )
    if not row["prompt" if schema == "prompt" else "context"]:
        raise ValueError(
            f"RULER dataset row {index} from {source} has an empty prompt/context."
        )
    answer = row["answer"]
    if isinstance(answer, str):
        if not answer:
            raise ValueError(
                f"RULER dataset row {index} from {source} has an empty answer."
            )
    elif isinstance(answer, (list, tuple)) and answer:
        if not all(isinstance(reference, str) and reference for reference in answer):
            raise ValueError(
                f"RULER dataset row {index} from {source} has invalid answer references."
            )
    else:
        raise ValueError(
            f"RULER dataset row {index} from {source} must have a string or "
            "non-empty list of answer references."
        )
    raw_max_new_tokens = row["max_new_tokens"]
    if isinstance(raw_max_new_tokens, bool):
        raise ValueError(
            f"RULER dataset row {index} from {source} has invalid max_new_tokens."
        )
    try:
        parsed_max_new_tokens = int(raw_max_new_tokens)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"RULER dataset row {index} from {source} has non-integer "
            f"max_new_tokens={raw_max_new_tokens!r}."
        ) from error
    if parsed_max_new_tokens <= 0:
        raise ValueError(
            f"RULER dataset row {index} from {source} has non-positive "
            f"max_new_tokens={parsed_max_new_tokens}."
        )


def _validated_row_factory(
    raw_factory: Callable[[], Iterator[dict[str, Any]]], *, source: str
) -> Callable[[], Iterator[dict[str, Any]]]:
    def factory() -> Iterator[dict[str, Any]]:
        raw_rows = iter(raw_factory())
        try:
            first_row = next(raw_rows)
        except StopIteration as error:
            raise ValueError(f"RULER dataset is empty: {source}") from error
        schema = "prompt" if "prompt" in first_row else "context_question"
        _validate_dataset_row(first_row, index=0, source=source, schema=schema)
        yield first_row
        for index, row in enumerate(raw_rows, start=1):
            _validate_dataset_row(row, index=index, source=source, schema=schema)
            yield row

    return factory


def _dataset_row_factory(
    config: EvalConfig,
) -> tuple[Callable[[], Iterator[dict[str, Any]]], str]:
    if config.dataset_path:
        path = Path(config.dataset_path)
        if path.is_dir() or (path.is_file() and _is_local_shadowkv_file(path)):
            return _local_directory_row_factory(path)
        if path.suffix == ".jsonl":
            raw_factory = lambda: _iter_jsonl_rows(path)
        else:
            json_rows = _read_json_rows(path)
            raw_factory = lambda: iter(json_rows)
        source = config.dataset_path
    elif config.dataset_repo_id:
        try:
            filename = config.dataset_file_template.format(data_dir=config.data_dir)
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(
                "--dataset-file-template must be format-compatible with {data_dir}."
            ) from error
        downloaded_path = Path(
            _download_hf_dataset_file(
                repo_id=config.dataset_repo_id,
                filename=filename,
                revision=config.dataset_revision,
            )
        )
        if downloaded_path.suffix != ".jsonl":
            json_rows = _read_json_rows(downloaded_path)
            raw_factory = lambda: iter(json_rows)
        else:
            raw_factory = lambda: _iter_jsonl_rows(downloaded_path)
        source = f"hf://datasets/{config.dataset_repo_id}/{filename}@{config.dataset_revision}"
    else:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "The kvpress RULER runner requires the 'datasets' package, or use "
                "--dataset-path/--dataset-repo-id."
            ) from error
        dataset_rows = [
            dict(row)
            for row in load_dataset(
                config.dataset_name,
                data_dir=config.data_dir or None,
                split="test",
            )
        ]
        raw_factory = lambda: iter(dataset_rows)
        source = f"datasets://{config.dataset_name}/{config.data_dir}"
    return _validated_row_factory(raw_factory, source=source), source


def _selected_row_factory(
    row_factory: Callable[[], Iterator[dict[str, Any]]], config: EvalConfig
) -> Callable[[], Iterator[dict[str, Any]]]:
    selected_indices: list[int] | None = None
    if config.fraction < 1.0:
        total = sum(1 for _ in row_factory())
        count = max(1, int(total * config.fraction))
        rng = random.Random(config.seed)
        selected_indices = sorted(rng.sample(range(total), count))
        if config.max_samples is not None:
            selected_indices = selected_indices[: config.max_samples]

    def factory() -> Iterator[dict[str, Any]]:
        if selected_indices is None:
            limit = config.max_samples
            for index, row in enumerate(row_factory()):
                if limit is not None and index >= limit:
                    break
                yield row
            return
        selected = iter(selected_indices)
        next_selected = next(selected, None)
        for index, row in enumerate(row_factory()):
            if next_selected is None:
                break
            if index == next_selected:
                yield row
                next_selected = next(selected, None)

    return factory


def _sharded_row_factory(
    selected_factory: Callable[[], Iterator[dict[str, Any]]],
    *,
    shard_index: int,
    shard_count: int,
) -> Callable[[], Iterator[dict[str, Any]]]:
    """Select one deterministic shard after global fraction/max-sample selection.

    The hidden ordinal lets the parent process merge worker artifacts back into
    the same sample order as a single-GPU evaluation without copying the large
    128K prompt strings into temporary shard files.
    """
    shard_index = int(shard_index)
    shard_count = int(shard_count)
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError(
            "Invalid shard selection: "
            f"index={shard_index} count={shard_count}."
        )
    if shard_count == 1:
        return selected_factory

    def factory() -> Iterator[dict[str, Any]]:
        for global_index, row in enumerate(selected_factory()):
            if global_index % shard_count != shard_index:
                continue
            sharded_row = dict(row)
            sharded_row["_sparsevllm_global_index"] = int(global_index)
            yield sharded_row

    return factory


def _load_rows(config: EvalConfig) -> list[dict[str, Any]]:
    row_factory, _ = _dataset_row_factory(config)
    selected_factory = _selected_row_factory(row_factory, config)
    sharded_factory = _sharded_row_factory(
        selected_factory,
        shard_index=config.shard_index,
        shard_count=config.shard_count,
    )
    return list(sharded_factory())


def _count_selected_rows(config: EvalConfig) -> int:
    """Count globally selected rows without retaining their large text fields."""
    row_factory, _ = _dataset_row_factory(config)
    selected_factory = _selected_row_factory(row_factory, config)
    return sum(1 for _ in selected_factory())


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


def _build_evaluation_groups(
    tasks: list[str], max_new_tokens: list[int]
) -> list[dict[str, Any]]:
    """Group selected rows by task and their resolved generation budget.

    Groups retain first-seen order, while each group's indices retain the
    original selected-dataset order. Including the budget in the key keeps a
    malformed or custom RULER file from mixing requests with different decode
    lengths in one batch.
    """
    if len(tasks) != len(max_new_tokens):
        raise ValueError(
            "tasks and max_new_tokens must have the same number of rows: "
            f"{len(tasks)} != {len(max_new_tokens)}."
        )
    groups_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    groups: list[dict[str, Any]] = []
    for index, (task, budget) in enumerate(zip(tasks, max_new_tokens)):
        key = (str(task), int(budget))
        group = groups_by_key.get(key)
        if group is None:
            group = {
                "task": key[0],
                "max_new_tokens": key[1],
                "indices": [],
            }
            groups_by_key[key] = group
            groups.append(group)
        group["indices"].append(index)
    return groups


def _iter_rows_at_indices(
    row_factory: Callable[[], Iterator[dict[str, Any]]],
    indices: list[int],
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Replay selected rows at sorted positions without materializing them."""
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("row indices must be sorted and unique")
    wanted = iter(indices)
    next_index = next(wanted, None)
    for index, row in enumerate(row_factory()):
        if next_index is None:
            return
        if index == next_index:
            yield index, row
            next_index = next(wanted, None)
        elif index > next_index:
            raise RuntimeError(
                "RULER dataset changed between tokenizer and generation passes: "
                f"missing selected row index {next_index}."
            )
    if next_index is not None:
        raise RuntimeError(
            "RULER dataset changed between tokenizer and generation passes: "
            f"missing selected row index {next_index}."
        )


def _row_output_index(row: dict[str, Any], local_index: int) -> int:
    """Return the stable selected-dataset ordinal for a worker row."""
    value = row.get("_sparsevllm_global_index")
    if value is None:
        return int(local_index)
    if isinstance(value, bool):
        raise ValueError("_sparsevllm_global_index must be an integer.")
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "_sparsevllm_global_index must be an integer, "
            f"got {value!r}."
        ) from error
    if value < 0:
        raise ValueError("_sparsevllm_global_index must be non-negative.")
    return value


def _source_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Return provenance fields when evaluating ShadowKV local rows."""
    return {
        key: row[key]
        for key in ("source_file", "source_index", "source_length")
        if key in row
    }


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


def _row_prompt(tokenizer, row: dict[str, Any], *, max_context_length: int) -> list[int]:
    """Build one prompt from either the legacy or preformatted RULER schema."""
    if "prompt" in row:
        prompt_text = str(row["prompt"])
        answer_prefix = str(row["answer_prefix"])
        prompt = tokenizer.encode(
            prompt_text + answer_prefix,
            add_special_tokens=False,
        )
        if len(prompt) <= max_context_length:
            return prompt
        # Keep the generation prefix even when a caller explicitly imposes a
        # shorter limit. The full prompt path above remains the normal,
        # boundary-faithful path; this branch is only for truncation.
        prefix_ids = tokenizer.encode(answer_prefix, add_special_tokens=False)
        if len(prefix_ids) >= max_context_length:
            return prefix_ids[-max_context_length:]
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        return prompt_ids[: max_context_length - len(prefix_ids)] + prefix_ids
    return _kvpress_prompt(
        tokenizer,
        str(row["context"]),
        str(row["question"]),
        str(row["answer_prefix"]),
        max_context_length=max_context_length,
    )


def _infer_max_model_len(
    prompt_lengths: Sequence[int], max_new_tokens: Sequence[int]
) -> int:
    """Return the exact maximum prompt-plus-generation budget.

    ``max_model_len`` is a hard model-context limit, so adding a safety margin
    here can make a valid sample set impossible to run at the model boundary.
    """
    if not prompt_lengths or len(prompt_lengths) != len(max_new_tokens):
        raise ValueError(
            "prompt_lengths and max_new_tokens must be non-empty and have equal lengths."
        )
    return max(
        int(prompt_length) + int(tokens)
        for prompt_length, tokens in zip(prompt_lengths, max_new_tokens)
    )


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


def _make_evaluation_progress(total: int):
    return tqdm(
        total=int(total),
        desc="Evaluating",
        unit="sample",
        dynamic_ncols=True,
    )


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
        "max_num_batched_tokens": config.max_batched_tokens,
        "decode_graph": config.decode_graph,
        "decode_graph_capture_sizes": [config.batch_size],
        "enable_prefix_caching": False,
    }
    if config.sparse_method == "vanilla":
        # Full Attention uses only the shared runtime/cache defaults.
        pass
    elif config.sparse_method == "quest":
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
                "shadowkv_local_chunks": config.shadowkv_local_chunks,
                "shadowkv_outlier_chunks": config.shadowkv_outlier_chunks,
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
                "query_robust_model_fingerprint": config.query_robust_model_fingerprint,
                "query_robust_uniform_p": config.query_robust_uniform_p,
                "sink_keep_tokens": config.sink_keep_tokens,
                "decode_keep_tokens": config.decode_keep_tokens,
                "recent_keep_tokens": config.recent_keep_tokens,
            }
        )
    else:
        raise ValueError(f"Unsupported sparse method: {config.sparse_method!r}")
    return infer_config


def _strip_cli_options(argv: Sequence[str], options: set[str]) -> list[str]:
    """Remove value-taking options before appending worker-specific values."""
    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        token = str(argv[index])
        option = token.split("=", 1)[0]
        if option in options:
            index += 1
            if "=" not in token:
                if index >= len(argv):
                    raise ValueError(f"Missing value for {option}.")
                index += 1
            continue
        cleaned.append(token)
        index += 1
    return cleaned


def _worker_master_port(worker_index: int) -> int:
    """Return a distinct localhost rendezvous port for one TP1 worker."""
    configured = os.environ.get("SPARSEVLLM_MASTER_PORT")
    if configured is not None:
        try:
            base_port = int(configured)
        except ValueError as error:
            raise ValueError(
                "SPARSEVLLM_MASTER_PORT must be an integer when --gpu-ids is used."
            ) from error
        port = base_port + int(worker_index)
        if not 1 <= port <= 65535:
            raise ValueError(
                "SPARSEVLLM_MASTER_PORT plus worker index must stay in [1, 65535], "
                f"got {port}."
            )
        return port

    # Ask the OS for a free port when the caller did not configure a base.
    # The child immediately binds the returned port through torch.distributed;
    # using different ports avoids the TP1 workers contending on the default.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return _read_json_rows(path)


def _terminate_worker_processes(
    child_processes: list[tuple[int, int, subprocess.Popen, Any, Path]],
) -> None:
    """Terminate and reap workers when orchestration itself aborts."""
    for _, _, process, _, _ in child_processes:
        if process.poll() is None:
            process.terminate()
    for _, _, process, log_handle, _ in child_processes:
        if process.poll() is None:
            process.wait()
        log_handle.close()


def _merge_worker_rows(
    *,
    worker_rows: list[dict[str, Any]],
    expected_indices: set[int],
    worker_index: int,
    gpu_id: int,
    output_name: str,
) -> list[dict[str, Any]]:
    """Validate and annotate rows emitted by one worker."""
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in worker_rows:
        if "index" not in row:
            raise RuntimeError(
                f"Worker {worker_index} {output_name} row has no index: {row!r}."
            )
        try:
            index = int(row["index"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Worker {worker_index} {output_name} row has invalid index "
                f"{row.get('index')!r}."
            ) from error
        if index not in expected_indices:
            raise RuntimeError(
                f"Worker {worker_index} emitted {output_name} index={index}, "
                f"outside its assigned shard."
            )
        if index in seen:
            raise RuntimeError(
                f"Worker {worker_index} emitted duplicate {output_name} index={index}."
            )
        seen.add(index)
        merged.append(
            {
                **row,
                "index": index,
                "shard_index": int(worker_index),
                "gpu_id": int(gpu_id),
            }
        )
    return merged


def _missing_worker_result_rows(
    *,
    selected_rows: list[dict[str, Any]],
    missing_indices: set[int],
    worker_index: int,
    gpu_id: int,
    max_new_tokens_override: int | None,
    error: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in sorted(missing_indices):
        source_row = selected_rows[index]
        task = str(source_row["task"])
        try:
            answer = _references(source_row["answer"], task=task)
        except Exception:
            answer = source_row.get("answer")
        max_new_tokens = _resolve_max_new_tokens(
            [source_row], override=max_new_tokens_override
        )[0]
        rows.append(
            {
                "index": int(index),
                "task": task,
                "status": "model_failed",
                "error_type": "WorkerOutputMissing",
                "error": error,
                "answer": answer,
                "input_token_count": None,
                "max_new_tokens": max_new_tokens,
                "shard_index": int(worker_index),
                "gpu_id": int(gpu_id),
            }
        )
    return rows


def _run_multi_gpu(config: EvalConfig) -> None:
    """Run disjoint TP1 workers and merge their auditable artifacts."""
    gpu_ids = tuple(config.gpu_ids or ())
    if len(gpu_ids) <= 1:
        raise ValueError("_run_multi_gpu requires at least two GPU IDs.")
    if config.shard_count != 1 or config.shard_index != 0:
        raise ValueError("--gpu-ids cannot be combined with internal shard options.")

    parent_started = time.perf_counter()
    total_rows = _count_selected_rows(config)
    if total_rows <= 0:
        raise ValueError("RULER dataset is empty after applying selection.")
    if len(gpu_ids) > total_rows:
        raise ValueError(
            "Cannot assign more workers than selected RULER rows: "
            f"workers={len(gpu_ids)} rows={total_rows}."
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "run_info.json",
        "aggregate_metrics.json",
        "raw_outputs.jsonl",
        "parsed_outputs.jsonl",
        "per_sample_results.jsonl",
        "multi_gpu_manifest.json",
    ):
        target = output_dir / filename
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing multi-GPU artifact: {target}"
            )
    shard_root = output_dir / "shards"
    if shard_root.exists() and any(shard_root.iterdir()):
        raise FileExistsError(
            f"Multi-GPU shard directory is not empty; choose a fresh output directory: "
            f"{shard_root}"
        )
    shard_root.mkdir(parents=True, exist_ok=True)
    parent_cwd = os.getcwd()

    child_argv = _strip_cli_options(
        sys.argv[1:],
        {
            "--gpu-ids",
            "--output-dir",
            "--device",
            "--shard-index",
            "--shard-count",
        },
    )
    child_processes: list[tuple[int, int, subprocess.Popen, Any, Path]] = []
    worker_records: list[dict[str, Any]] = []
    worker_ports: set[int] = set()
    try:
        for worker_index, gpu_id in enumerate(gpu_ids):
            worker_dir = shard_root / f"worker-{worker_index:02d}-gpu-{gpu_id}"
            worker_dir.mkdir(parents=True, exist_ok=False)
            log_path = worker_dir / "worker.log"
            worker_args = [
                *child_argv,
                "--output-dir",
                str(worker_dir),
                "--device",
                "cuda:0",
                "--shard-index",
                str(worker_index),
                "--shard-count",
                str(len(gpu_ids)),
            ]
            command = [sys.executable, str(Path(__file__).resolve()), *worker_args]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            master_port = _worker_master_port(worker_index)
            while master_port in worker_ports:
                master_port = _worker_master_port(worker_index)
            worker_ports.add(master_port)
            env["SPARSEVLLM_MASTER_PORT"] = str(master_port)
            pythonpath = [str(REPO_ROOT), str(SRC_ROOT)]
            if env.get("PYTHONPATH"):
                pythonpath.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(pythonpath)
            record = {
                "worker_index": int(worker_index),
                "gpu_id": int(gpu_id),
                "shard_rule": f"global_selected_index % {len(gpu_ids)} == {worker_index}",
                "output_dir": str(worker_dir),
                "log_path": str(log_path),
                "command": shlex.join(command),
                "master_port": int(env["SPARSEVLLM_MASTER_PORT"]),
            }
            log_handle = log_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command,
                    # Preserve relative model/dataset/output paths exactly as the
                    # parent invocation sees them.  REPO_ROOT is still injected in
                    # PYTHONPATH below for imports.
                    cwd=parent_cwd,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception as error:
                log_handle.close()
                record.update(
                    {
                        "returncode": None,
                        "launch_error_type": type(error).__name__,
                        "launch_error": str(error),
                    }
                )
                worker_records.append(record)
                continue
            child_processes.append((worker_index, gpu_id, process, log_handle, worker_dir))
            worker_records.append(record)
            _phase_log(
                f"Launched TP1 worker {worker_index + 1}/{len(gpu_ids)} "
                f"on physical GPU {gpu_id}: pid={process.pid}"
            )
    except BaseException:
        _phase_log("Multi-GPU worker launch aborted; terminating launched workers.")
        _terminate_worker_processes(child_processes)
        raise

    try:
        for worker_index, gpu_id, process, log_handle, worker_dir in child_processes:
            returncode = process.wait()
            log_handle.close()
            for record in worker_records:
                if record["worker_index"] == worker_index:
                    record["returncode"] = int(returncode)
                    break
            _phase_log(
                f"Worker {worker_index} on GPU {gpu_id} finished with returncode={returncode}."
            )
    except KeyboardInterrupt:
        _phase_log("Interrupted; terminating outstanding multi-GPU workers.")
        _terminate_worker_processes(child_processes)
        raise

    worker_records.sort(key=lambda record: int(record["worker_index"]))
    worker_result_rows: list[dict[str, Any]] = []
    worker_raw_rows: list[dict[str, Any]] = []
    worker_parsed_rows: list[dict[str, Any]] = []
    child_aggregates: list[dict[str, Any]] = []
    missing_by_worker: dict[int, set[int]] = {}
    worker_output_contract_ok = True
    for record in worker_records:
        worker_index = int(record["worker_index"])
        gpu_id = int(record["gpu_id"])
        expected_indices = set(range(worker_index, total_rows, len(gpu_ids)))
        worker_dir = Path(record["output_dir"])
        result_path = worker_dir / "per_sample_results.jsonl"
        try:
            result_rows = _read_jsonl_if_present(result_path)
        except Exception as error:
            result_rows = []
            record["result_read_error_type"] = type(error).__name__
            record["result_read_error"] = str(error)
        merged_results = _merge_worker_rows(
            worker_rows=result_rows,
            expected_indices=expected_indices,
            worker_index=worker_index,
            gpu_id=gpu_id,
            output_name="per_sample_results.jsonl",
        )
        seen_indices = {int(row["index"]) for row in merged_results}
        missing_indices = expected_indices - seen_indices
        if missing_indices:
            missing_by_worker[worker_index] = missing_indices
        worker_result_rows.extend(merged_results)
        for filename, target in (
            ("raw_outputs.jsonl", worker_raw_rows),
            ("parsed_outputs.jsonl", worker_parsed_rows),
        ):
            auxiliary_path = worker_dir / filename
            if (
                record.get("returncode") == 0
                and result_rows
                and not auxiliary_path.is_file()
            ):
                record.setdefault("missing_output_files", []).append(filename)
                worker_output_contract_ok = False
            rows = _read_jsonl_if_present(auxiliary_path)
            target.extend(
                _merge_worker_rows(
                    worker_rows=rows,
                    expected_indices=expected_indices,
                    worker_index=worker_index,
                    gpu_id=gpu_id,
                    output_name=filename,
                )
            )
        aggregate_path = worker_dir / "aggregate_metrics.json"
        if aggregate_path.is_file():
            child_aggregates.append(
                json.loads(aggregate_path.read_text(encoding="utf-8"))
            )
        elif record.get("returncode") == 0:
            record.setdefault("missing_output_files", []).append(
                "aggregate_metrics.json"
            )
            worker_output_contract_ok = False

    if missing_by_worker:
        selected_rows = _load_rows(
            replace(config, gpu_ids=None, shard_index=0, shard_count=1)
        )
        if len(selected_rows) != total_rows:
            raise RuntimeError(
                "RULER dataset changed while merging multi-GPU workers: "
                f"count_before={total_rows} count_after={len(selected_rows)}."
            )
        for worker_index, missing_indices in missing_by_worker.items():
            record = worker_records[worker_index]
            worker_result_rows.extend(
                _missing_worker_result_rows(
                    selected_rows=selected_rows,
                    missing_indices=missing_indices,
                    worker_index=worker_index,
                    gpu_id=int(record["gpu_id"]),
                    max_new_tokens_override=config.max_new_tokens,
                    error=(
                        "Worker did not emit a per-sample result; inspect "
                        f"{record['log_path']} and its returncode."
                    ),
                )
            )

    worker_result_rows.sort(key=lambda row: int(row["index"]))
    worker_raw_rows.sort(key=lambda row: int(row["index"]))
    worker_parsed_rows.sort(key=lambda row: int(row["index"]))
    _write_jsonl(output_dir / "raw_outputs.jsonl", worker_raw_rows)
    _write_jsonl(output_dir / "parsed_outputs.jsonl", worker_parsed_rows)
    _write_jsonl(output_dir / "per_sample_results.jsonl", worker_result_rows)

    observed_indices = {int(row["index"]) for row in worker_result_rows}
    expected_all = set(range(total_rows))
    if observed_indices != expected_all:
        raise RuntimeError(
            "Merged multi-GPU results do not cover the selected dataset exactly: "
            f"missing={sorted(expected_all - observed_indices)[:10]} "
            f"extra={sorted(observed_indices - expected_all)[:10]}."
        )

    successful = [row for row in worker_result_rows if row.get("status") == "success"]
    by_task: dict[str, float | None] = {}
    for task in sorted({str(row["task"]) for row in worker_result_rows}):
        values = [
            float(row["score"])
            for row in successful
            if str(row["task"]) == task
        ]
        by_task[task] = round(100.0 * float(np.mean(values)), 2) if values else None

    worker_startups = [
        float(value["startup_seconds"])
        for value in child_aggregates
        if value.get("startup_seconds") is not None
    ]
    worker_serving = [
        float(value["serving_elapsed_seconds"])
        for value in child_aggregates
        if value.get("serving_elapsed_seconds") is not None
    ]
    parallel_serving = max(worker_serving) if worker_serving else None
    parent_elapsed = time.perf_counter() - parent_started
    all_workers_succeeded = worker_output_contract_ok and all(
        record.get("returncode") == 0 for record in worker_records
    ) and len(worker_records) == len(gpu_ids)
    aggregate: dict[str, Any] = {
        "protocol": "NVIDIA/kvpress evaluation/evaluate.py RULER dataset and scorer",
        "sparse_method": config.sparse_method,
        "status": (
            "success"
            if all_workers_succeeded and len(successful) == total_rows
            else "failed"
        ),
        "num_samples": total_rows,
        "num_success": len(successful),
        "overall_score": (
            round(100.0 * float(np.mean([row["score"] for row in successful])), 2)
            if successful
            else None
        ),
        "score_by_task": by_task,
        "startup_seconds": round(max(worker_startups), 3) if worker_startups else None,
        "serving_elapsed_seconds": (
            round(parallel_serving, 3) if parallel_serving is not None else None
        ),
        "elapsed_seconds": (
            round(parallel_serving, 3) if parallel_serving is not None else None
        ),
        "total_elapsed_seconds": round(parent_elapsed, 3),
        "multi_gpu_wall_seconds": round(parent_elapsed, 3),
        "gpu_ids": list(gpu_ids),
        "shard_count": len(gpu_ids),
    }
    aggregate["input_tokens"] = sum(
        int(row["input_token_count"])
        for row in successful
        if row.get("input_token_count") is not None
    )
    aggregate["requested_output_tokens"] = sum(
        int(row["max_new_tokens"]) for row in successful
    )
    aggregate["generated_output_tokens"] = sum(
        int(row["output_token_count"])
        for row in successful
        if row.get("output_token_count") is not None
    )
    if parallel_serving is not None and parallel_serving > 0:
        aggregate["samples_per_second"] = round(
            len(successful) / parallel_serving, 4
        )
        aggregate["generated_output_tokens_per_second"] = round(
            aggregate["generated_output_tokens"] / parallel_serving, 4
        )
        aggregate["requested_output_tokens_per_second"] = round(
            aggregate["requested_output_tokens"] / parallel_serving, 4
        )
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8"
    )
    run_info = {
        "protocol": aggregate["protocol"],
        "multi_gpu": True,
        "gpu_ids": list(gpu_ids),
        "shard_count": len(gpu_ids),
        "shard_rule": "global selected row ordinal modulo shard_count",
        "dataset_source": (
            config.dataset_path
            or config.dataset_repo_id
            or f"datasets://{config.dataset_name}/{config.data_dir}"
        ),
        "dataset_rows": total_rows,
        "config": asdict(config),
        "workers": worker_records,
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    (output_dir / "multi_gpu_manifest.json").write_text(
        json.dumps(
            {
                "status": aggregate["status"],
                "gpu_ids": list(gpu_ids),
                "selected_rows": total_rows,
                "shard_rule": run_info["shard_rule"],
                "workers": worker_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _phase_log(
        "Multi-GPU evaluation finished: "
        f"status={aggregate['status']} samples={aggregate['num_success']}/{total_rows} "
        f"wall={aggregate['multi_gpu_wall_seconds']}s"
    )
    print(json.dumps(aggregate, indent=2))
    if aggregate["status"] != "success":
        raise RuntimeError(
            "Multi-GPU RULER evaluation failed; inspect multi_gpu_manifest.json "
            "and each shards/*/worker.log."
        )


def main() -> None:
    config = _parse_args()
    if config.gpu_ids is not None:
        if len(config.gpu_ids) == 1:
            # --gpu-ids names physical devices.  The evaluator itself then
            # sees the selected physical device as logical cuda:0.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids[0])
            config = replace(config, device="cuda:0")
        else:
            _run_multi_gpu(config)
            return
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _phase_log(
        "Loading dataset metadata: "
        f"source={config.dataset_path or config.dataset_repo_id or config.dataset_name} "
        f"data_dir={config.data_dir} fraction={config.fraction} "
        f"max_samples={config.max_samples}"
    )
    row_factory, dataset_source = _dataset_row_factory(config)
    selected_factory = _sharded_row_factory(
        _selected_row_factory(row_factory, config),
        shard_index=config.shard_index,
        shard_count=config.shard_count,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    from transformers import AutoTokenizer
    from benchmark.model_adapters.sparsevllm import get_sparsevllm_generate_api

    tokenizer_started = time.perf_counter()
    _phase_log(f"Loading tokenizer from {config.model_path!r}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=False)
    _phase_log(
        "Tokenizer ready: "
        f"model_max_length={tokenizer.model_max_length} "
        f"elapsed={time.perf_counter() - tokenizer_started:.2f}s"
    )
    max_new_tokens: list[int] = []
    prompt_schema: str | None = None
    max_context_length = config.max_context_length
    if max_context_length is None:
        max_context_length = min(int(tokenizer.model_max_length), int(1e10))
    # Keep only lengths in memory. A complete 128K artifact can contain
    # thousands of prompts, so retaining every token-id list would consume
    # multiple gigabytes before inference starts.
    prompt_lengths: list[int] = []
    task_names: list[str] = []
    global_row_indices: list[int] = []
    first_tokenize_started = time.perf_counter()
    _phase_log(
        "Tokenizing prompts (pass 1/2) to infer prompt lengths and max_model_len..."
    )
    with tqdm(
        selected_factory(),
        total=config.max_samples,
        desc="Tokenizing prompts (pass 1/2)",
        unit="sample",
        dynamic_ncols=True,
    ) as token_progress:
        for row in token_progress:
            if prompt_schema is None:
                prompt_schema = "prompt" if "prompt" in row else "context_question"
            task_names.append(str(row["task"]))
            global_row_indices.append(_row_output_index(row, len(prompt_lengths)))
            max_new_tokens.append(
                _resolve_max_new_tokens([row], override=config.max_new_tokens)[0]
            )
            prompt_lengths.append(
                len(
                    _row_prompt(
                        tokenizer,
                        row,
                        max_context_length=max_context_length,
                    )
                )
            )
    dataset_rows = len(prompt_lengths)
    if not dataset_rows or prompt_schema is None:
        raise ValueError("RULER dataset is empty after applying selection.")
    evaluation_groups = _build_evaluation_groups(task_names, max_new_tokens)
    inferred_max_model_len = _infer_max_model_len(prompt_lengths, max_new_tokens)
    _phase_log(
        "Tokenization pass 1 complete: "
        f"rows={dataset_rows} max_prompt_tokens={max(prompt_lengths)} "
        f"elapsed={time.perf_counter() - first_tokenize_started:.2f}s"
    )
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
        config = replace(
            config, shadowkv_gpu_cache_tokens=int(resolved_max_model_len)
        )
    infer_config = _build_infer_config(
        config,
        resolved_max_model_len=resolved_max_model_len,
    )
    derived_config = {
        "target_context_length": int(config.data_dir)
        if config.data_dir is not None and str(config.data_dir).isdigit()
        else None,
    }
    if config.sparse_method == "vanilla":
        derived_config["attention_mode"] = "full"
    elif config.sparse_method == "quest":
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
            config.shadowkv_sparse_budget,
            config.shadowkv_outlier_chunks,
        )
    elif config.sparse_method == "query_robust":
        derived_config["query_robust_effective_token_budget"] = (
            config.sink_keep_tokens
            + config.decode_keep_tokens
            + config.recent_keep_tokens
        )
    else:
        raise ValueError(f"Unsupported sparse method: {config.sparse_method!r}")
    run_info = {
        "protocol": "NVIDIA/kvpress evaluation/evaluate.py RULER dataset and scorer",
        "protocol_source": "https://github.com/NVIDIA/kvpress/tree/main/evaluation",
        "config": asdict(config),
        "infer_config": infer_config,
        "derived_config": derived_config,
        "max_new_tokens_source": (
            "cli_override" if config.max_new_tokens is not None else "dataset_row"
        ),
        "dataset_source": dataset_source,
        "dataset_rows": dataset_rows,
        "evaluation_grouping": {
            "policy": "task_and_max_new_tokens",
            "num_groups": len(evaluation_groups),
            "groups": [
                {
                    "group_index": group_index,
                    "task": group["task"],
                    "max_new_tokens": group["max_new_tokens"],
                    "num_rows": len(group["indices"]),
                    "row_indices": [
                        global_row_indices[index] for index in group["indices"]
                    ],
                }
                for group_index, group in enumerate(evaluation_groups)
            ],
        },
        "prompt_semantics": (
            "preformatted RULER prompt + answer_prefix"
            if prompt_schema == "prompt"
            else "kvpress tokenizer-visible context + question + answer_prefix"
        ),
        "native_semantic_difference": "Sparse-vLLM compresses after its combined prefill; kvpress presses context before question decoding.",
        "device": config.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": int(config.batch_size),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    started = time.perf_counter()
    serving_started: float | None = None
    evaluation_finished: float | None = None
    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    processed_indices: set[int] = set()
    progress = None
    llm = None
    try:
        if config.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"Requested {config.device}, but CUDA is unavailable.")
            torch.cuda.set_device(torch.device(config.device))
        engine_started = time.perf_counter()
        _phase_log(
            "Initializing Sparse-vLLM engine: model loading, KV allocation, "
            "startup profiling, and warmup..."
        )
        generate = get_sparsevllm_generate_api(
            config.model_path,
            infer_config,
            sparse_method=config.sparse_method,
        )
        # LLM construction returns after model loading, startup profiling,
        # graph capture, and warmup, immediately after the engine logs
        # "Startup completed". Exclude that cold-start interval from rates.
        serving_started = time.perf_counter()
        _phase_log(
            "Engine ready; entering generation/evaluation: "
            f"startup_elapsed={serving_started - engine_started:.2f}s "
            f"rows={dataset_rows} batch_size={config.batch_size}"
        )
        progress = _make_evaluation_progress(dataset_rows)
        llm = getattr(generate, "_sparsevllm_llm", None)
        from sparsevllm.operators.registry import operator_binding_reports

        runtime_start = {
            "operator_bindings": operator_binding_reports(),
            "cuda_graph_requested": bool(config.decode_graph),
        }
        (output_dir / "runtime_bindings.json").write_text(
            json.dumps(runtime_start, indent=2, default=str), encoding="utf-8"
        )
        for group_index, group in enumerate(evaluation_groups, start=1):
            group_indices = group["indices"]
            group_budget = int(group["max_new_tokens"])
            grouped_rows = iter(_iter_rows_at_indices(selected_factory, group_indices))
            total_group_batches = (
                len(group_indices) + config.batch_size - 1
            ) // config.batch_size
            for batch_start in range(0, len(group_indices), config.batch_size):
                batch_number = batch_start // config.batch_size + 1
                batch_indices = group_indices[
                    batch_start : batch_start + config.batch_size
                ]
                batch_pairs = list(islice(grouped_rows, len(batch_indices)))
                if [index for index, _ in batch_pairs] != batch_indices:
                    raise RuntimeError(
                        "RULER dataset changed between tokenizer and generation passes: "
                        f"expected row indices {batch_indices}, got "
                        f"{[index for index, _ in batch_pairs]}."
                    )
                batch_rows = [row for _, row in batch_pairs]
                for index, row in batch_pairs:
                    replayed_task = str(row["task"])
                    replayed_budget = _resolve_max_new_tokens(
                        [row], override=config.max_new_tokens
                    )[0]
                    if replayed_task != group["task"] or replayed_budget != group_budget:
                        raise RuntimeError(
                            "RULER dataset changed between tokenizer and generation passes: "
                            f"row {index} is task={replayed_task!r}, "
                            f"max_new_tokens={replayed_budget}, expected "
                            f"task={group['task']!r}, max_new_tokens={group_budget}."
                        )
                batch_tokenize_started = time.perf_counter()
                _phase_log(
                    f"Tokenizing batch (pass 2/2): group={group_index}/"
                    f"{len(evaluation_groups)} batch={batch_number}/"
                    f"{total_group_batches} rows={batch_indices[0]}-"
                    f"{batch_indices[-1]}"
                )
                batch_prompts = [
                    _row_prompt(
                        tokenizer,
                        row,
                        max_context_length=max_context_length,
                    )
                    for row in batch_rows
                ]
                _phase_log(
                    "Batch tokenization complete: "
                    f"tokens={sum(len(prompt) for prompt in batch_prompts)} "
                    f"elapsed={time.perf_counter() - batch_tokenize_started:.2f}s; "
                    "generating..."
                )
                generation_started = time.perf_counter()
                outputs = generate(
                    batch_prompts,
                    max_new_tokens=[group_budget] * len(batch_rows),
                    do_sample=False,
                    temperature=0.0,
                )
                if not isinstance(outputs, list) or len(outputs) != len(batch_rows):
                    raise RuntimeError(
                        "Native generation returned "
                        f"{len(outputs) if isinstance(outputs, list) else type(outputs)} "
                        f"outputs for batch size {len(batch_rows)}."
                    )
                for (index, row), prediction in zip(batch_pairs, outputs):
                    processed_indices.add(index)
                    prediction = str(prediction)
                    task = str(row["task"])
                    output_index = _row_output_index(row, index)
                    raw_rows.append(
                        {
                            "index": output_index,
                            **_source_fields(row),
                            "prediction": prediction,
                        }
                    )
                    parsed_rows.append(
                        {
                            "index": output_index,
                            **_source_fields(row),
                            "predicted_answer": prediction.strip(),
                        }
                    )
                    base_result = {
                        "index": output_index,
                        **_source_fields(row),
                        "task": task,
                        "predicted_answer": prediction,
                        "input_token_count": prompt_lengths[index],
                        "max_new_tokens": max_new_tokens[index],
                    }
                    try:
                        references = _references(row["answer"], task=task)
                    except Exception as row_error:
                        result_rows.append(
                            {
                                **base_result,
                                "status": "parse_failed",
                                "error_type": type(row_error).__name__,
                                "error": str(row_error),
                                "answer": row.get("answer"),
                            }
                        )
                        continue
                    try:
                        score = _score(task, prediction, references)
                    except Exception as row_error:
                        result_rows.append(
                            {
                                **base_result,
                                "status": "metric_failed",
                                "error_type": type(row_error).__name__,
                                "error": str(row_error),
                                "answer": references,
                            }
                        )
                        continue
                    try:
                        output_token_count = len(
                            tokenizer.encode(prediction, add_special_tokens=False)
                        )
                    except Exception as row_error:
                        result_rows.append(
                            {
                                **base_result,
                                "status": "parse_failed",
                                "error_type": type(row_error).__name__,
                                "error": str(row_error),
                                "answer": references,
                            }
                        )
                        continue
                    result_rows.append(
                        {
                            **base_result,
                            "status": "success",
                            "answer": references,
                            "score": score,
                            "output_token_count": output_token_count,
                        }
                    )
                _phase_log(
                    "Batch generation complete: "
                    f"rows={batch_indices[0]}-{batch_indices[-1]} "
                    f"elapsed={time.perf_counter() - generation_started:.2f}s"
                )
                progress.update(len(batch_pairs))
        evaluation_finished = time.perf_counter()
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
        _phase_log(
            f"Evaluation failed after {len(processed_indices)}/{dataset_rows} "
            f"processed samples: {type(error).__name__}: {error}"
        )
        # Preserve an explicit terminal status for every evaluated sample.
        for index, row in enumerate(selected_factory()):
            if index in processed_indices:
                continue
            output_index = _row_output_index(row, index)
            result_rows.append(
                {
                    "index": output_index,
                    **_source_fields(row),
                    "task": str(row["task"]),
                    "status": "model_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "answer": _references(row["answer"], task=str(row["task"])),
                    "input_token_count": prompt_lengths[index],
                    "max_new_tokens": max_new_tokens[index],
                }
            )
        (output_dir / "error.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2),
            encoding="utf-8",
        )
    finally:
        if progress is not None:
            progress.close()
        if llm is not None:
            # Explicitly stop the engine before interpreter shutdown.  This
            # avoids the atexit path trying to create a shutdown thread after
            # Python has already begun tearing down its threading runtime.
            llm.exit()
    raw_rows.sort(key=lambda row: row["index"])
    parsed_rows.sort(key=lambda row: row["index"])
    result_rows.sort(key=lambda row: row["index"])
    _write_jsonl(output_dir / "raw_outputs.jsonl", raw_rows)
    _write_jsonl(output_dir / "parsed_outputs.jsonl", parsed_rows)
    _write_jsonl(output_dir / "per_sample_results.jsonl", result_rows)
    successful = [row for row in result_rows if row["status"] == "success"]
    by_task: dict[str, float] = {}
    for task in sorted({str(row["task"]) for row in result_rows}):
        values = [float(row["score"]) for row in successful if row["task"] == task]
        by_task[task] = round(100.0 * float(np.mean(values)), 2) if values else None
    timing = _evaluation_timing(
        started_at=started,
        serving_started_at=serving_started,
        finished_at=(
            evaluation_finished
            if evaluation_finished is not None
            else time.perf_counter()
        ),
    )
    aggregate = {
        "protocol": run_info["protocol"],
        "sparse_method": config.sparse_method,
        "status": "success" if len(successful) == dataset_rows else "failed",
        "num_samples": dataset_rows,
        "num_success": len(successful),
        "overall_score": round(100.0 * float(np.mean([row["score"] for row in successful])), 2) if successful else None,
        "score_by_task": by_task,
        **timing,
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
    serving_elapsed = aggregate["serving_elapsed_seconds"]
    if serving_elapsed is not None and serving_elapsed > 0:
        aggregate["samples_per_second"] = round(
            len(successful) / serving_elapsed, 4
        )
        aggregate["generated_output_tokens_per_second"] = round(
            aggregate["generated_output_tokens"] / serving_elapsed, 4
        )
        aggregate["requested_output_tokens_per_second"] = round(
            aggregate["requested_output_tokens"] / serving_elapsed, 4
        )
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8")
    _phase_log(
        "Evaluation finished: "
        f"status={aggregate['status']} samples={aggregate['num_success']}/{aggregate['num_samples']} "
        f"total_elapsed={timing['total_elapsed_seconds']}s"
    )
    print(json.dumps(aggregate, indent=2))
    if aggregate["status"] != "success":
        raise RuntimeError("RULER evaluation failed; inspect per_sample_results.jsonl and error.json.")


if __name__ == "__main__":
    main()
