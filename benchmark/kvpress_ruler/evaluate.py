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
from dataclasses import asdict, dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from tqdm.auto import tqdm

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
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.90
    batch_size: int = 4
    max_batched_tokens: int = 65536
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


def _parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="simonjegou/ruler")
    parser.add_argument("--data-dir", default="4096")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--dataset-file-template",
        default="ruler-{data_dir}.jsonl",
        help="Filename template used with --dataset-repo-id.",
    )
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
    parser.add_argument("--decode-keep-tokens", type=int, default=2048)
    parser.add_argument("--sink-keep-tokens", type=int, default=0)
    parser.add_argument("--recent-keep-tokens", type=int, default=0)
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


def _load_rows(config: EvalConfig) -> list[dict[str, Any]]:
    row_factory, _ = _dataset_row_factory(config)
    selected_factory = _selected_row_factory(row_factory, config)
    return list(selected_factory())


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


def main() -> None:
    config = _parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _phase_log(
        "Loading dataset metadata: "
        f"source={config.dataset_path or config.dataset_repo_id or config.dataset_name} "
        f"data_dir={config.data_dir} fraction={config.fraction} "
        f"max_samples={config.max_samples}"
    )
    row_factory, dataset_source = _dataset_row_factory(config)
    selected_factory = _selected_row_factory(row_factory, config)
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
    inferred_max_model_len = max(
        prompt_length + tokens
        for prompt_length, tokens in zip(prompt_lengths, max_new_tokens)
    ) + 16
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
            config.shadowkv_sparse_budget,
            config.shadowkv_outlier_chunks,
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
                    "row_indices": group["indices"],
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
                    raw_rows.append({"index": index, "prediction": prediction})
                    parsed_rows.append(
                        {"index": index, "predicted_answer": prediction.strip()}
                    )
                    base_result = {
                        "index": index,
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
            result_rows.append(
                {
                    "index": index,
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
