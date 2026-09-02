"""Reproducible RULER pilot runner for Dense, Quest, ShadowKV, and Query-Robust."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sparse_frontier.evaluation import evaluate_jsonl_dataset
from sparse_frontier.ruler_pilot import load_pilot_rows


# 96, 128, and 256 remain safely above Query-Robust's mandatory
# sink/recent/current reservation with 16-token chunks.  In particular,
# ShadowKV's fixed local/outlier set consumes 416 tokens, so its 96-token
# selectable budget matches a 512-token total-access comparison.
SPARSE_BUDGETS = (96, 128, 256, 512, 1024, 2048)
_CAPTURE_TASKS = ("niah_single", "niah_multikey", "niah_multiquery", "vt", "fwe")


@dataclass(frozen=True)
class RunSpec:
    method: str
    budget: int | None

    @property
    def name(self) -> str:
        return self.method if self.budget is None else f"{self.method}_b{self.budget}"


def validate_budget(method: str, budget: int | None) -> None:
    if method == "dense":
        if budget is not None:
            raise ValueError("Dense does not accept a sparse budget")
    elif method in {"quest", "shadowkv", "query_robust"}:
        if budget not in SPARSE_BUDGETS:
            raise ValueError(f"{method} budget must be one of {SPARSE_BUDGETS}, got {budget!r}")
    else:
        raise ValueError(
            f"Unsupported RULER method {method!r}; expected dense, quest, shadowkv, or query_robust"
        )


def build_run_matrix(smoke: bool = False) -> list[RunSpec]:
    if smoke:
        return [RunSpec("dense", None), RunSpec("quest", 512), RunSpec("shadowkv", 512)]
    return [
        RunSpec("dense", None),
        *(RunSpec("quest", budget) for budget in SPARSE_BUDGETS),
        *(RunSpec("shadowkv", budget) for budget in SPARSE_BUDGETS),
    ]


def _select_task_indices(
    rows: Iterable[dict[str, Any]], task_indices: tuple[int, ...]
) -> list[dict[str, Any]]:
    requested = set(task_indices)
    if not requested or any(index < 0 for index in requested):
        raise ValueError("task indices must be a nonempty set of nonnegative values")
    selected = [row for row in rows if int(row["task_index"]) in requested]
    observed = {int(row["task_index"]) for row in selected}
    if observed != requested:
        raise ValueError(
            f"task-index selection is missing {sorted(requested - observed)}"
        )
    return selected


def _select_tasks(
    rows: Iterable[dict[str, Any]], task_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Keep an explicit task-family slice without changing its examples.

    This is deliberately separate from ``--task_indices``: a targeted pilot
    must still use the same task-index example as the matched multi-task run,
    rather than silently selecting the first available row in a family.
    """

    requested = set(task_names)
    allowed = set(_CAPTURE_TASKS)
    if not requested or not requested <= allowed:
        raise ValueError(
            f"tasks must be a nonempty subset of {sorted(allowed)}"
        )
    selected = [row for row in rows if str(row["task"]) in requested]
    observed = {str(row["task"]) for row in selected}
    if observed != requested:
        raise ValueError(f"task selection is missing {sorted(requested - observed)}")
    return selected


def _load_model_config(
    model_path: str | Path,
    tp: int,
    max_input_tokens: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Load attention geometry from the pinned checkpoint rather than a model name."""
    config_path = Path(model_path) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        num_q_heads = int(config["num_attention_heads"])
        hidden_size = int(config["hidden_size"])
        model_config = {
            "tp": int(tp),
            "num_q_heads": num_q_heads,
            "num_kv_heads": int(config["num_key_value_heads"]),
            "num_layers": int(config["num_hidden_layers"]),
            "head_dim": hidden_size // num_q_heads,
            "max_input_tokens": int(max_input_tokens),
            "max_output_tokens": int(max_output_tokens),
            "kv_cache_block_size": 256,
            "model_id": _model_id_from_snapshot(model_path),
            "model_revision": Path(model_path).resolve().name,
            "rope_type": str(
                (config.get("rope_scaling") or {}).get(
                    "rope_type", (config.get("rope_scaling") or {}).get("type", "default")
                )
            ),
            "rope_parameters": {
                **{
                    key: value
                    for key, value in (config.get("rope_scaling") or {}).items()
                    if key not in {"rope_type", "type"}
                },
                "rope_theta": config.get("rope_theta"),
            },
            "attention_scale": (hidden_size // num_q_heads) ** -0.5,
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot load model attention geometry from {config_path}: {exc}") from exc
    if hidden_size % num_q_heads:
        raise ValueError("model hidden size must be divisible by query heads")
    if model_config["num_kv_heads"] % model_config["tp"]:
        raise ValueError("model KV heads must be divisible by tensor parallel size")
    return model_config


def _capture_context_payload(
    sample: dict[str, Any], prompt_length: int
) -> dict[str, Any]:
    """Build the strict, text-free context passed to each worker capture hook."""
    task_index = int(sample["task_index"])
    if task_index < 5:
        split = "calibration"
    elif task_index < 10:
        split = "validation"
    elif task_index < 15:
        split = "pilot"
    else:
        split = "locked"
    task = str(sample["task"])
    if task not in _CAPTURE_TASKS:
        raise ValueError(f"unsupported capture task {task!r}")
    context_length = int(sample.get("context_length", prompt_length))
    length_bucket = 0 if context_length <= 8192 else 1 if context_length <= 16384 else 2
    identity = json.dumps(
        {
            "index": int(sample["index"]),
            "task": task,
            "task_index": task_index,
            "context_length": context_length,
            "input_text": str(sample["input_text"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    sequence_id_hash = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & (
        2**63 - 1
    )
    return {
        "sequence_id_hash": sequence_id_hash,
        "task": task,
        "task_index": task_index,
        "split": split,
        "prompt_length": int(prompt_length),
        "stratum_id": _CAPTURE_TASKS.index(task) * 3 + length_bucket,
    }


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _model_id_from_snapshot(model_path: str | Path) -> str:
    resolved = Path(model_path).resolve()
    try:
        cache_name = resolved.parents[1].name
    except IndexError:
        cache_name = ""
    if cache_name.startswith("models--"):
        return cache_name.removeprefix("models--").replace("--", "/")
    return str(resolved)


def _prepare_raw_capture_manifest(
    capture_dir: str | Path,
    model_path: str | Path,
    model_cfg: dict[str, int],
    data_path: str | Path,
    max_decode_offset: int,
) -> None:
    from sparse_frontier.modelling.attention.query_pool import (
        REPRESENTATION,
        QueryPoolManifest,
        SCHEMA_VERSION,
    )

    root = Path(capture_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() or any(root.glob("request_*")):
        raise FileExistsError(
            f"raw query capture already exists in {root}; choose a new directory"
        )
    model_root = Path(model_path).resolve()
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    rope_scaling = dict(config.get("rope_scaling") or {})
    rope_type = str(rope_scaling.pop("rope_type", rope_scaling.pop("type", "default")))
    rope_parameters = {**rope_scaling, "rope_theta": config.get("rope_theta")}
    source_digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_commit = "unknown"
    manifest = QueryPoolManifest(
        schema_version=SCHEMA_VERSION,
        model_id=_model_id_from_snapshot(model_root),
        model_revision=model_root.name,
        num_layers=model_cfg["num_layers"],
        num_q_heads=model_cfg["num_q_heads"],
        num_kv_heads=model_cfg["num_kv_heads"],
        head_dim=model_cfg["head_dim"],
        tp_size=model_cfg["tp"],
        rope_type=rope_type,
        rope_parameters=rope_parameters,
        representation=REPRESENTATION,
        attention_scale=model_cfg["head_dim"] ** -0.5,
        max_decode_offset=int(max_decode_offset),
        pool_size_per_kv_group=1,
        coreset_sizes=[],
        capture_split="task-index split recorded per request",
        source_manifest_sha256=source_digest,
        created_with_git_commit=git_commit,
    )
    _atomic_write_json(manifest_path, manifest.to_dict())


def _configure_attention(
    spec: RunSpec,
    model_cfg: dict[str, int],
    shadowkv_svd_backend: str = "randomized",
    shadowkv_svd_oversample: int = 32,
    shadowkv_svd_niter: int = 2,
    shadowkv_fused_retrieval: bool = True,
    capture_query_dir: str | Path | None = None,
    capture_context_path: str | Path | None = None,
    capture_layers: tuple[int, ...] | None = None,
    capture_prompt_keys: bool = True,
    query_pool_path: str | Path | None = None,
    query_robust_generation_horizon: int | None = None,
    query_robust_recent_chunks: int = 1,
    query_robust_objective: str = "minimax",
    query_robust_cvar_alpha: float = 0.95,
    query_robust_initial_support: int = 64,
    query_robust_max_support: int = 1024,
    query_robust_solver_gap_tolerance: float = 1e-3,
    query_robust_solver_max_iterations: int = 64,
    query_robust_solver_violators_per_round: int = 4,
    query_robust_solver_chunk_batch_size: int = 16,
    query_robust_empirical_query_budget: int | None = None,
    query_robust_bias_mode: str = "raw_entropy",
    query_robust_share_chunks_across_kv_heads: bool = True,
    query_robust_shared_chunk_aggregation: str = "sum",
    query_robust_solver_fail_closed: bool = True,
    query_robust_async_prefill_build: bool = False,
    query_robust_solver_armijo: bool = True,
) -> None:
    os.environ["VLLM_USE_V1"] = "1"
    # Query-Robust owns request-local tensors and its attention registry in the
    # Python process that configures vLLM.  On TP=1, a spawned worker receives
    # neither registry state nor the initialized tensor-parallel group.  Keep
    # this execution in process (unless the caller supplied an explicit mode),
    # matching the validated smoke path.  Other methods retain vLLM's normal
    # multiprocessing default.
    if spec.method == "query_robust" and int(model_cfg["tp"]) == 1:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    else:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if spec.method == "dense" and capture_query_dir is None:
        os.environ["SF_USE_ATTENTION_PATCH"] = "0"
        os.environ.pop("SF_ATTENTION_NAME", None)
        os.environ.pop("SF_ATTENTION_ARGS_JSON", None)
        for name in (
            "SF_QUERY_CAPTURE_DIR",
            "SF_QUERY_CAPTURE_CONTEXT",
            "SF_QUERY_CAPTURE_LAYERS",
        ):
            os.environ.pop(name, None)
        return

    if capture_query_dir is not None:
        if spec.method != "dense":
            raise ValueError("the first query-capture implementation requires dense attention")
        if capture_context_path is None:
            raise ValueError("query capture requires a context path")
        if "head_dim" not in model_cfg:
            raise ValueError("query capture requires model head_dim")
        os.environ["SF_USE_ATTENTION_PATCH"] = "1"
        os.environ.pop("SF_ATTENTION_NAME", None)
        os.environ.pop("SF_ATTENTION_ARGS_JSON", None)
        os.environ["SF_QUERY_CAPTURE_DIR"] = str(Path(capture_query_dir).resolve())
        os.environ["SF_QUERY_CAPTURE_CONTEXT"] = str(
            Path(capture_context_path).resolve()
        )
        os.environ["SF_QUERY_CAPTURE_PROMPT_KEYS"] = "1" if capture_prompt_keys else "0"
        if capture_layers is None:
            os.environ.pop("SF_QUERY_CAPTURE_LAYERS", None)
        else:
            os.environ["SF_QUERY_CAPTURE_LAYERS"] = json.dumps(list(capture_layers))
        os.environ["SF_TP_SIZE"] = str(model_cfg["tp"])
        os.environ["SF_MODEL_NUM_Q_HEADS"] = str(model_cfg["num_q_heads"])
        os.environ["SF_MODEL_NUM_KV_HEADS"] = str(model_cfg["num_kv_heads"])
        os.environ["SF_MODEL_NUM_LAYERS"] = str(model_cfg["num_layers"])
        os.environ["SF_MODEL_HEAD_DIM"] = str(model_cfg["head_dim"])
        os.environ["SF_MAX_INPUT_TOKENS"] = str(model_cfg["max_input_tokens"])
        os.environ["SF_MAX_OUTPUT_TOKENS"] = str(model_cfg["max_output_tokens"])
        os.environ["SF_KV_CACHE_BLOCK_SIZE"] = str(model_cfg["kv_cache_block_size"])
        return

    if spec.method == "quest":
        attention_args = {"token_budget": spec.budget, "page_size": 16, "share_pages": True}
    elif spec.method == "shadowkv":
        attention_args = {
            "sparse_budget": spec.budget,
            "chunk_size": 8,
            "rank": 160,
            "local_chunk": 4,
            "outlier_chunk": 48,
            "svd_backend": shadowkv_svd_backend,
            "svd_oversample": shadowkv_svd_oversample,
            "svd_niter": shadowkv_svd_niter,
            "fused_retrieval": shadowkv_fused_retrieval,
        }
    else:
        if query_pool_path is None:
            raise ValueError("query_robust requires --query_pool_path")
        if query_robust_generation_horizon is None:
            query_robust_generation_horizon = int(model_cfg["max_output_tokens"])
        attention_args = {
            "objective": str(query_robust_objective),
            "cvar_alpha": float(query_robust_cvar_alpha),
            "initial_support": int(query_robust_initial_support),
            "max_support": int(query_robust_max_support),
            "solver_gap_tolerance": float(query_robust_solver_gap_tolerance),
            "solver_max_iterations": int(query_robust_solver_max_iterations),
            "solver_violators_per_round": int(query_robust_solver_violators_per_round),
            "solver_chunk_batch_size": int(query_robust_solver_chunk_batch_size),
            "empirical_query_budget": query_robust_empirical_query_budget,
            "bias_mode": str(query_robust_bias_mode),
            "share_chunks_across_kv_heads": bool(query_robust_share_chunks_across_kv_heads),
            "shared_chunk_aggregation": str(query_robust_shared_chunk_aggregation),
            "solver_fail_closed": bool(query_robust_solver_fail_closed),
            "async_prefill_build": bool(query_robust_async_prefill_build),
            "solver_armijo": bool(query_robust_solver_armijo),
            "token_budget": spec.budget,
            "chunk_size": 16,
            "query_pool_path": str(Path(query_pool_path).resolve()),
            "generation_horizon": int(query_robust_generation_horizon),
            "sink_chunks": 1,
            "recent_chunks": int(query_robust_recent_chunks),
            "summary_dtype": "bfloat16",
            "score_dtype": "float32",
        }
    os.environ["SF_USE_ATTENTION_PATCH"] = "1"
    os.environ["SF_ATTENTION_NAME"] = spec.method
    os.environ["SF_ATTENTION_ARGS_JSON"] = json.dumps(attention_args, sort_keys=True)
    os.environ["SF_TP_SIZE"] = str(model_cfg["tp"])
    os.environ["SF_MODEL_NUM_Q_HEADS"] = str(model_cfg["num_q_heads"])
    os.environ["SF_MODEL_NUM_KV_HEADS"] = str(model_cfg["num_kv_heads"])
    os.environ["SF_MODEL_NUM_LAYERS"] = str(model_cfg["num_layers"])
    if "head_dim" in model_cfg:
        os.environ["SF_MODEL_HEAD_DIM"] = str(model_cfg["head_dim"])
    os.environ["SF_MAX_INPUT_TOKENS"] = str(model_cfg["max_input_tokens"])
    os.environ["SF_MAX_OUTPUT_TOKENS"] = str(model_cfg["max_output_tokens"])
    os.environ["SF_KV_CACHE_BLOCK_SIZE"] = str(model_cfg["kv_cache_block_size"])
    if spec.method == "query_robust":
        os.environ["SF_MODEL_ID"] = str(model_cfg["model_id"])
        os.environ["SF_MODEL_REVISION"] = str(model_cfg["model_revision"])
        os.environ["SF_MODEL_ROPE_TYPE"] = str(model_cfg["rope_type"])
        os.environ["SF_MODEL_ROPE_PARAMETERS_JSON"] = json.dumps(
            model_cfg["rope_parameters"], sort_keys=True
        )
        os.environ["SF_ATTENTION_SCALE"] = str(model_cfg["attention_scale"])


def _start_metrics_server(spec: RunSpec):
    if spec.method == "dense":
        return None
    from vllm.utils import get_open_port
    from sparse_frontier.utils.sparsity_server import configure_sparsity_env, start_sparsity_server

    handle = start_sparsity_server(
        host="127.0.0.1",
        port=get_open_port(),
        authkey=secrets.token_hex(16),
    )
    configure_sparsity_env(handle.config)
    return handle


def _stop_metrics_server(handle) -> None:
    if handle is not None and handle.process.is_alive():
        handle.process.terminate()
        handle.process.join(timeout=5)


def _generate_one(
    model: Any,
    sample: dict[str, Any],
    spec: RunSpec,
    capture_context_path: str | Path | None = None,
    capture_max_tokens: int | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        if capture_context_path is not None:
            encoded = model.tokenizer.encode_for_generation(
                sample["input_text"], return_tensors=False
            )
            _atomic_write_json(
                capture_context_path,
                _capture_context_payload(sample, int(encoded["input_length"])),
            )
        generation_tokens = min(
            int(sample["tokens_to_generate"]), int(model.max_output_tokens)
        )
        if capture_max_tokens is not None:
            generation_tokens = min(generation_tokens, int(capture_max_tokens))
        output = model.generate(
            sample["input_text"], max_tokens=generation_tokens
        )
        result = {
            "index": sample["index"],
            "task": sample["task"],
            "method": spec.method,
            "budget": spec.budget,
            "pred": output["text"],
            "output_tokens_len": output["output_tokens_len"],
            "runtime_s": output.get("runtime_s", time.perf_counter() - start),
            "decode_latency_s": output.get("decode_latency_s"),
            "decode_latency_source": output.get("decode_latency_source"),
            "peak_gpu_memory_bytes": output.get("peak_gpu_memory_bytes"),
            "peak_gpu_memory_source": output.get("peak_gpu_memory_source"),
            "failure": False,
        }
        return result
    except Exception as exc:  # every failed sample is persisted and counted
        return {
            "index": sample["index"],
            "task": sample["task"],
            "method": spec.method,
            "budget": spec.budget,
            "pred": "",
            "output_tokens_len": 0,
            "runtime_s": time.perf_counter() - start,
            "decode_latency_s": None,
            "peak_gpu_memory_bytes": None,
            "failure": True,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_spec(
    spec: RunSpec,
    rows: Iterable[dict[str, Any]],
    data_path: str | Path,
    output_dir: str | Path,
    model_path: str,
    max_input_tokens: int = 8192,
    max_output_tokens: int = 1024,
    seed: int = 43,
    tp: int = 1,
    smoke: bool = False,
    shadowkv_svd_backend: str = "randomized",
    shadowkv_svd_oversample: int = 32,
    shadowkv_svd_niter: int = 2,
    shadowkv_fused_retrieval: bool = True,
    capture_query_dir: str | Path | None = None,
    capture_context_path: str | Path | None = None,
    capture_layers: tuple[int, ...] | None = None,
    capture_max_tokens: int | None = None,
    capture_prompt_keys: bool = True,
    query_pool_path: str | Path | None = None,
    query_robust_generation_horizon: int | None = None,
    query_robust_recent_chunks: int = 1,
    query_robust_objective: str = "minimax",
    query_robust_cvar_alpha: float = 0.95,
    query_robust_initial_support: int = 64,
    query_robust_max_support: int = 1024,
    query_robust_solver_gap_tolerance: float = 1e-3,
    query_robust_solver_max_iterations: int = 64,
    query_robust_solver_violators_per_round: int = 4,
    query_robust_solver_chunk_batch_size: int = 16,
    query_robust_empirical_query_budget: int | None = None,
    query_robust_bias_mode: str = "raw_entropy",
    query_robust_share_chunks_across_kv_heads: bool = True,
    query_robust_shared_chunk_aggregation: str = "sum",
    query_robust_solver_fail_closed: bool = True,
    query_robust_async_prefill_build: bool = False,
    query_robust_solver_armijo: bool = True,
) -> dict[str, Any]:
    """Run one method/budget and write predictions plus aggregate artifacts."""
    validate_budget(spec.method, spec.budget)
    selected_rows = list(rows)
    if smoke:
        selected_rows = selected_rows[:1]
    if not selected_rows:
        raise ValueError("Cannot run RULER with an empty dataset")

    run_dir = Path(output_dir) / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"
    aggregate_json = run_dir / "aggregate.json"
    aggregate_csv = run_dir / "aggregate.csv"
    manifest_path = run_dir / "run.json"
    run_data_path = run_dir / "dataset.jsonl"
    if pred_path.exists() or aggregate_json.exists() or aggregate_csv.exists():
        raise FileExistsError(
            f"Run artifacts already exist in {run_dir}; choose a new output_dir to avoid mixing runs"
        )
    with run_data_path.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    model_cfg = _load_model_config(
        model_path,
        tp=tp,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
    )
    resolved_capture_context: Path | None = None
    if capture_query_dir is not None:
        if capture_max_tokens is None:
            capture_max_tokens = max_output_tokens
        if capture_max_tokens < 1:
            raise ValueError("capture_max_tokens must be positive")
        resolved_capture_context = (
            Path(capture_context_path)
            if capture_context_path is not None
            else Path(capture_query_dir) / ".capture-context.json"
        )
        _prepare_raw_capture_manifest(
            capture_query_dir,
            model_path,
            model_cfg,
            data_path,
            max_decode_offset=capture_max_tokens - 1,
        )
    _configure_attention(
        spec,
        model_cfg,
        shadowkv_svd_backend=shadowkv_svd_backend,
        shadowkv_svd_oversample=shadowkv_svd_oversample,
        shadowkv_svd_niter=shadowkv_svd_niter,
        shadowkv_fused_retrieval=shadowkv_fused_retrieval,
        capture_query_dir=capture_query_dir,
        capture_context_path=resolved_capture_context,
        capture_layers=capture_layers,
        capture_prompt_keys=capture_prompt_keys,
        query_pool_path=query_pool_path,
        query_robust_generation_horizon=query_robust_generation_horizon,
        query_robust_recent_chunks=query_robust_recent_chunks,
        query_robust_objective=query_robust_objective,
        query_robust_cvar_alpha=query_robust_cvar_alpha,
        query_robust_initial_support=query_robust_initial_support,
        query_robust_max_support=query_robust_max_support,
        query_robust_solver_gap_tolerance=query_robust_solver_gap_tolerance,
        query_robust_solver_max_iterations=query_robust_solver_max_iterations,
        query_robust_solver_violators_per_round=query_robust_solver_violators_per_round,
        query_robust_solver_chunk_batch_size=query_robust_solver_chunk_batch_size,
        query_robust_empirical_query_budget=query_robust_empirical_query_budget,
        query_robust_bias_mode=query_robust_bias_mode,
        query_robust_share_chunks_across_kv_heads=query_robust_share_chunks_across_kv_heads,
        query_robust_shared_chunk_aggregation=query_robust_shared_chunk_aggregation,
        query_robust_solver_fail_closed=query_robust_solver_fail_closed,
        query_robust_async_prefill_build=query_robust_async_prefill_build,
        query_robust_solver_armijo=query_robust_solver_armijo,
    )
    metrics_handle = _start_metrics_server(spec)
    model = None
    results: list[dict[str, Any]] = []
    init_failure: str | None = None
    try:
        import torch
        from sparse_frontier.modelling.models.vllm_model import VLLMModel

        model = VLLMModel(
            model_path=model_path,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            dtype=torch.bfloat16,
            tensor_parallel_size=tp,
            seed=seed,
            enable_thinking=False,
        )
        for sample in selected_rows:
            result = _generate_one(
                model,
                sample,
                spec,
                capture_context_path=resolved_capture_context,
                capture_max_tokens=capture_max_tokens,
            )
            if metrics_handle is not None:
                if spec.method == "query_robust" and not result["failure"]:
                    # Flush trailing periodic counters after vLLM completes the
                    # request, avoiding telemetry IPC on every decode step.
                    from sparse_frontier.modelling.attention.registry import (
                        get_attention_handler,
                    )

                    get_attention_handler().flush_decode_access(torch.device("cuda"))
                from sparse_frontier.utils.sparsity_server import fetch_sparsity_and_reset

                stats = fetch_sparsity_and_reset()
                if stats:
                    result.update({
                        "prefill_sparsity": stats.get("prefill_sparsity"),
                        "decode_access_sum": stats.get("decode_access_sum"),
                        "decode_dense_sum": stats.get("decode_dense_sum"),
                        "shadowkv_state_ready": stats.get("shadowkv_state_ready"),
                        "shadowkv_layers_built": stats.get("shadowkv_layers_built"),
                        "shadowkv_decode_seen": stats.get("shadowkv_decode_seen"),
                        "query_robust_state_ready": stats.get(
                            "query_robust_state_ready"
                        ),
                        "query_robust_summaries_built": stats.get(
                            "query_robust_summaries_built"
                        ),
                        "query_robust_summary_build_ms": stats.get(
                            "query_robust_summary_build_ms"
                        ),
                        "query_robust_fallback_count": stats.get(
                            "query_robust_fallback_count"
                        ),
                        "query_robust_decode_seen": stats.get(
                            "query_robust_decode_seen"
                        ),
                        "query_robust_objective": stats.get("query_robust_objective"),
                        "query_robust_solver_gap_max": stats.get("query_robust_solver_gap_max"),
                        "query_robust_solver_active_max": stats.get("query_robust_solver_active_max"),
                        "query_robust_solver_nonconverged": stats.get("query_robust_solver_nonconverged"),
                        "query_robust_quantized_error_max": stats.get("query_robust_quantized_error_max"),
                    })
                    if spec.method == "shadowkv" and not result["failure"]:
                        if stats.get("shadowkv_state_ready") is not True:
                            result["failure"] = True
                            result["error"] = "ShadowKV per-layer state was not reported as built"
                        elif stats.get("shadowkv_decode_seen") is not True:
                            result["failure"] = True
                            result["error"] = "ShadowKV sparse decode path was not reported as executed"
                    if spec.method == "query_robust" and not result["failure"]:
                        if stats.get("query_robust_state_ready") is not True:
                            result["failure"] = True
                            result["error"] = "Query-Robust per-layer state was not reported as built"
                        elif stats.get("query_robust_decode_seen") is not True:
                            result["failure"] = True
                            result["error"] = "Query-Robust sparse decode path was not reported as executed"
                        elif int(stats.get("query_robust_fallback_count") or 0) != 0:
                            result["failure"] = True
                            result["error"] = "Query-Robust reported a state fallback"
                elif not result["failure"]:
                    result["failure"] = True
                    result["error"] = "attention telemetry server returned no per-example state"
            results.append(result)
    except Exception as exc:
        init_failure = f"{type(exc).__name__}: {exc}"
        # A model initialization failure has no valid per-example output; write
        # explicit failure rows so the run is auditable.
        results = [
            {
                "index": sample["index"],
                "task": sample["task"],
                "method": spec.method,
                "budget": spec.budget,
                "pred": "",
                "output_tokens_len": 0,
                "runtime_s": None,
                "decode_latency_s": None,
                "peak_gpu_memory_bytes": None,
                "failure": True,
                "error": f"model initialization failed: {init_failure}",
            }
            for sample in selected_rows
        ]
    finally:
        _stop_metrics_server(metrics_handle)
        if model is not None:
            del model

    with pred_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    aggregate = evaluate_jsonl_dataset(
        data_path=run_data_path,
        predictions_path=pred_path,
        output_json=aggregate_json,
        output_csv=aggregate_csv,
        method=spec.method,
        budget=spec.budget,
    )
    manifest = {
        "method": spec.method,
        "budget": spec.budget,
        "model_path": model_path,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "max_num_batched_tokens": int(
            os.getenv("SF_MAX_NUM_BATCHED_TOKENS", str(max_input_tokens + max_output_tokens))
        ),
        "gpu_memory_utilization": float(os.getenv("SF_GPU_MEMORY_UTILIZATION", "0.85")),
        "cpu_offload_gb": float(os.getenv("SF_CPU_OFFLOAD_GB", "0")),
        "kv_cache_memory_bytes": (
            int(os.environ["SF_KV_CACHE_MEMORY_BYTES"])
            if os.getenv("SF_KV_CACHE_MEMORY_BYTES") is not None
            else None
        ),
        "dtype": "bfloat16",
        "batch_size": 1,
        "seed": seed,
        "tp": tp,
        "smoke": smoke,
        "shadowkv_svd_backend": shadowkv_svd_backend,
        "shadowkv_svd_oversample": shadowkv_svd_oversample,
        "shadowkv_svd_niter": shadowkv_svd_niter,
        "shadowkv_fused_retrieval": shadowkv_fused_retrieval,
        "num_examples": len(selected_rows),
        "dataset_path": str(data_path),
        "run_dataset_path": str(run_data_path),
        "initialization_failure": init_failure,
        "generation_allowances": sorted(
            {int(sample["tokens_to_generate"]) for sample in selected_rows}
        ),
        "capture_query_dir": str(capture_query_dir) if capture_query_dir else None,
        "capture_context_path": (
            str(resolved_capture_context) if resolved_capture_context else None
        ),
        "capture_layers": list(capture_layers) if capture_layers is not None else None,
        "capture_max_tokens": capture_max_tokens,
        "capture_prompt_keys": capture_prompt_keys if capture_query_dir else None,
        "query_pool_path": str(query_pool_path) if query_pool_path else None,
        "query_robust_generation_horizon": query_robust_generation_horizon,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "predictions_path": str(pred_path),
        "aggregate_json": str(aggregate_json),
        "aggregate_csv": str(aggregate_csv),
        "manifest_path": str(manifest_path),
        "failures": int(aggregate["failures"]),
    }


def _write_matrix_aggregate(
    output_dir: str | Path,
    data_path: str | Path,
    run_specs: list[RunSpec],
) -> None:
    """Combine per-run task rows into one auditable matrix artifact."""
    output = Path(output_dir)
    task_rows: list[dict[str, Any]] = []
    failures = 0
    for spec in run_specs:
        aggregate_path = output / spec.name / "aggregate.json"
        if not aggregate_path.exists():
            continue
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        task_rows.extend(aggregate.get("tasks", []))
        failures += int(aggregate.get("failures", 0))
    matrix = {
        "data_path": str(data_path),
        "runs": [spec.name for spec in run_specs],
        "failures": failures,
        "tasks": task_rows,
    }
    (output / "aggregate.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = list(task_rows[0]) if task_rows else ["method", "task", "budget", "failures"]
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(task_rows)


def run_cli(args: argparse.Namespace) -> int:
    rows = load_pilot_rows(args.data_path)
    if args.tasks is not None:
        rows = _select_tasks(rows, tuple(args.tasks))
    if args.task_indices is not None:
        rows = _select_task_indices(rows, tuple(args.task_indices))
    if args.capture_query_dir is not None and args.method != "dense":
        raise ValueError("query capture requires an explicit --method dense run")
    specs = build_run_matrix(smoke=args.smoke)
    if args.method is not None:
        validate_budget(args.method, args.budget)
        specs = [RunSpec(args.method, args.budget)]

    # vLLM initializes CUDA in the parent process and cannot safely be
    # re-created for the next method in the same interpreter.  Keep each
    # matrix entry isolated while retaining the direct run_spec API for tests
    # and callers that execute one configuration at a time.
    if args.method is None and len(specs) > 1:
        all_failures = 0
        for spec in specs:
            command = [
                sys.executable,
                "-m",
                "sparse_frontier.ruler_runner",
                "--data_path",
                str(args.data_path),
                "--model_path",
                str(args.model_path),
                "--output_dir",
                str(args.output_dir),
                "--method",
                spec.method,
                "--max_input_tokens",
                str(args.max_input_tokens),
                "--max_output_tokens",
                str(args.max_output_tokens),
                "--seed",
                str(args.seed),
                "--tp",
                str(args.tp),
                "--shadowkv_svd_backend",
                args.shadowkv_svd_backend,
                "--shadowkv_svd_oversample",
                str(args.shadowkv_svd_oversample),
                "--shadowkv_svd_niter",
                str(args.shadowkv_svd_niter),
            ]
            command.append(
                "--shadowkv_fused_retrieval"
                if args.shadowkv_fused_retrieval
                else "--no-shadowkv_fused_retrieval"
            )
            if spec.budget is not None:
                command.extend(["--budget", str(spec.budget)])
            if args.smoke:
                command.append("--smoke")
            completed = subprocess.run(command, check=False)
            all_failures += int(completed.returncode != 0)
        _write_matrix_aggregate(args.output_dir, args.data_path, specs)
        return 2 if all_failures else 0

    all_failures = 0
    for spec in specs:
        run = run_spec(
            spec=spec,
            rows=rows,
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_path=args.model_path,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            seed=args.seed,
            tp=args.tp,
            smoke=args.smoke,
            shadowkv_svd_backend=args.shadowkv_svd_backend,
            shadowkv_svd_oversample=args.shadowkv_svd_oversample,
            shadowkv_svd_niter=args.shadowkv_svd_niter,
            shadowkv_fused_retrieval=args.shadowkv_fused_retrieval,
            capture_query_dir=args.capture_query_dir,
            capture_context_path=args.capture_context_path,
            capture_layers=(
                tuple(args.capture_layers) if args.capture_layers is not None else None
            ),
            capture_max_tokens=args.capture_max_tokens,
            capture_prompt_keys=not args.capture_queries_only,
            query_pool_path=args.query_pool_path,
            query_robust_generation_horizon=args.query_robust_generation_horizon,
            query_robust_recent_chunks=args.query_robust_recent_chunks,
            query_robust_objective=args.query_robust_objective,
            query_robust_cvar_alpha=args.query_robust_cvar_alpha,
            query_robust_initial_support=args.query_robust_initial_support,
            query_robust_max_support=args.query_robust_max_support,
            query_robust_solver_gap_tolerance=args.query_robust_solver_gap_tolerance,
            query_robust_solver_max_iterations=args.query_robust_solver_max_iterations,
            query_robust_solver_violators_per_round=args.query_robust_solver_violators_per_round,
            query_robust_solver_chunk_batch_size=args.query_robust_solver_chunk_batch_size,
            query_robust_empirical_query_budget=args.query_robust_empirical_query_budget,
            query_robust_bias_mode=args.query_robust_bias_mode,
            query_robust_share_chunks_across_kv_heads=args.query_robust_share_chunks_across_kv_heads,
            query_robust_shared_chunk_aggregation=args.query_robust_shared_chunk_aggregation,
            query_robust_solver_fail_closed=args.query_robust_solver_fail_closed,
            query_robust_async_prefill_build=args.query_robust_async_prefill_build,
            query_robust_solver_armijo=args.query_robust_solver_armijo,
        )
        all_failures += int(run["failures"])
        print(json.dumps({"run": spec.name, **run}, sort_keys=True))
    if args.method is None:
        _write_matrix_aggregate(args.output_dir, args.data_path, specs)
    return 2 if all_failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Dense/Quest/ShadowKV/Query-Robust RULER pilot")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--smoke", action="store_true", help="one example for each method")
    parser.add_argument(
        "--method", choices=("dense", "quest", "shadowkv", "query_robust")
    )
    parser.add_argument("--budget", type=int)
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--max_output_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--task_indices",
        type=int,
        nargs="+",
        help="run these task_index values for every task family",
    )
    parser.add_argument(
        "--tasks",
        choices=_CAPTURE_TASKS,
        nargs="+",
        help="restrict a pilot to named RULER task families without resampling examples",
    )
    parser.add_argument(
        "--capture_query_dir",
        help="raw dense query/key capture directory (requires --method dense)",
    )
    parser.add_argument(
        "--capture_context_path",
        help="atomic request-context handoff path; defaults inside capture_query_dir",
    )
    parser.add_argument(
        "--capture_layers",
        type=int,
        nargs="+",
        help="zero-based model layers to capture; defaults to every layer",
    )
    parser.add_argument(
        "--capture_max_tokens",
        type=int,
        help="cap generated tokens during capture without changing source data",
    )
    parser.add_argument(
        "--capture_queries_only",
        action="store_true",
        help="omit large prompt-key shards when building only an empirical query pool",
    )
    parser.add_argument(
        "--query_pool_path",
        help="validated schema-2 Pile empirical query artifact for query_robust",
    )
    parser.add_argument(
        "--query_robust_generation_horizon",
        type=int,
        help="declared decode horizon used to position the empirical queries",
    )
    parser.add_argument(
        "--query_robust_recent_chunks",
        type=int,
        default=1,
        help="mandatory recent logical chunks included inside the token budget",
    )
    parser.add_argument(
        "--query_robust_objective", choices=("minimax", "cvar"), default="minimax",
        help="robust empirical objective used per key chunk",
    )
    parser.add_argument("--query_robust_cvar_alpha", type=float, default=0.95)
    parser.add_argument("--query_robust_initial_support", type=int, default=64)
    parser.add_argument("--query_robust_max_support", type=int, default=1024)
    parser.add_argument("--query_robust_solver_gap_tolerance", type=float, default=1e-3)
    parser.add_argument("--query_robust_solver_max_iterations", type=int, default=64)
    parser.add_argument(
        "--query_robust_solver_violators_per_round",
        type=int,
        default=4,
        help="inactive full-pool violators admitted per active-set round",
    )
    parser.add_argument("--query_robust_solver_chunk_batch_size", type=int, default=16)
    parser.add_argument(
        "--query_robust_empirical_query_budget",
        type=int,
        help="total balanced captured queries per KV group; omit for the full pool",
    )
    parser.add_argument(
        "--query_robust_bias_mode",
        choices=("raw_entropy", "mean_residual", "minimax_midpoint"),
        default="raw_entropy",
        help="affine scalar; calibrated modes are exploratory and require fail-closed routing disabled",
    )
    parser.add_argument(
        "--query_robust_share_chunks_across_kv_heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="share one routed chunk set across KV heads; disable for the per-KV-head quality ablation",
    )
    parser.add_argument(
        "--query_robust_shared_chunk_aggregation",
        choices=("sum", "max", "raw_score_sum"),
        default="sum",
        help="aggregate GQA-group chunk mass across KV heads when using one shared cache table",
    )
    parser.add_argument(
        "--query_robust_solver_fail_closed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="route only certified summaries; disable solely for exploratory router pilots",
    )
    parser.add_argument(
        "--query_robust_async_prefill_build",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="overlap immutable CUDA summary construction with later prefill work",
    )
    parser.add_argument(
        "--query_robust_solver_armijo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply Armijo backtracking; disable only for fixed-support exploratory one-step pilots",
    )
    parser.add_argument(
        "--shadowkv_svd_backend",
        choices=("exact", "gesvdj", "randomized"),
        default="randomized",
        help="ShadowKV SVD backend; gesvdj is full-SVD CUDA, randomized is approximate",
    )
    parser.add_argument("--shadowkv_svd_oversample", type=int, default=32)
    parser.add_argument("--shadowkv_svd_niter", type=int, default=2)
    parser.add_argument(
        "--shadowkv_fused_retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the optional fused landmark GEMM+softmax kernel",
    )
    args = parser.parse_args()
    if not args.smoke and args.method is None:
        print("Running the full 7-configuration matrix; use --smoke first for the required smoke pass.")
    raise SystemExit(run_cli(args))


if __name__ == "__main__":
    main()
