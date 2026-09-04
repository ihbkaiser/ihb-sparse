"""Validated request-local prompt-query reservoir artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class PromptQuerySupport:
    """Prompt pre-RoPE rows grouped by local KV head."""

    queries_by_kv_head: Tensor
    positions_by_kv_head: Tensor
    weights_by_kv_head: Tensor
    source_path: Path


def _normalise_model_id(value: str) -> str:
    marker = "models--"
    if marker in value:
        value = value.split(marker, 1)[1].split("/snapshots", 1)[0]
        return value.replace("--", "/")
    return value


def validate_prompt_query_manifest(
    root: str | Path, expected_manifest: Mapping[str, Any]
) -> None:
    """Require prompt shards to come from the same model geometry and RoPE."""

    path = Path(root) / "manifest.json"
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"cannot inspect prompt-query capture manifest: {exc}") from exc
    if not isinstance(actual, dict):
        raise ValueError("prompt-query capture manifest is not a JSON object")
    required = {
        "model_id",
        "model_revision",
        "num_layers",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "tp_size",
        "rope_type",
        "rope_parameters",
        "attention_scale",
        "representation",
    }
    missing = sorted(required - set(actual))
    if missing:
        raise ValueError(f"prompt-query capture manifest is missing {missing}")
    exact_fields = (
        "model_revision",
        "num_layers",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "tp_size",
        "rope_type",
        "rope_parameters",
    )
    mismatches = [
        name
        for name in exact_fields
        if actual[name] != expected_manifest.get(name)
    ]
    if _normalise_model_id(str(actual["model_id"])) != _normalise_model_id(
        str(expected_manifest.get("model_id"))
    ):
        mismatches.append("model_id")
    expected_scale = float(expected_manifest.get("attention_scale"))
    if not math.isclose(
        float(actual["attention_scale"]), expected_scale, rel_tol=1e-7, abs_tol=1e-9
    ):
        mismatches.append("attention_scale")
    if actual["representation"] != "prompt_origin_aligned_pre_scale":
        mismatches.append("representation")
    if mismatches:
        raise ValueError(
            "prompt-query capture manifest mismatch: " + ", ".join(sorted(set(mismatches)))
        )


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"request prompt-query shard is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with torch < 2.6
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"request prompt-query shard is corrupt: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"request prompt-query shard is not a dictionary: {path}")
    return payload


def load_prompt_query_support(
    root: str | Path,
    request_id: int,
    layer_idx: int,
    *,
    tp_rank: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    prompt_length: int,
    samples_per_q_head: int | None = None,
    expected_manifest: Mapping[str, Any] | None = None,
) -> PromptQuerySupport:
    """Load and validate one request's prompt-only support shard.

    The artifact contains pre-RoPE Q rows, so callers must apply the exact
    source-position-to-target-position transport before scoring. This loader
    intentionally accepts no generated-query fields and never searches another
    request when the requested shard is absent.
    """

    if request_id < 0 or layer_idx < 0 or tp_rank < 0:
        raise ValueError("request, layer, and TP identifiers must be nonnegative")
    positive = {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "prompt_length": prompt_length,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if num_q_heads % num_kv_heads:
        raise ValueError("request prompt-query geometry has invalid GQA divisibility")
    if expected_manifest is not None:
        validate_prompt_query_manifest(root, expected_manifest)
    path = (
        Path(root)
        / f"request_{int(request_id):016x}"
        / f"prompt_queries_layer_{int(layer_idx):03d}_rank_{int(tp_rank):03d}.pt"
    )
    payload = _load(path)
    required = {
        "prompt_pre_rope_queries",
        "prompt_positions",
        "layer_idx",
        "representation",
        "sampling",
        "metadata",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        unexpected = sorted(set(payload) - required)
        raise ValueError(
            f"request prompt-query shard fields invalid: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if int(payload["layer_idx"]) != layer_idx:
        raise ValueError("request prompt-query shard layer does not match requested layer")
    if payload["representation"] != "q_norm_pre_rope_pre_scale":
        raise ValueError("request prompt-query shard has unsupported representation")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("request prompt-query shard metadata is invalid")
    if int(metadata.get("sequence_id_hash", -1)) != request_id:
        raise ValueError("request prompt-query shard identity does not match requested request")
    if int(metadata.get("prompt_length", -1)) != prompt_length:
        raise ValueError("request prompt-query shard prompt length does not match the request")
    queries = payload["prompt_pre_rope_queries"]
    positions = payload["prompt_positions"]
    if not isinstance(queries, Tensor) or not isinstance(positions, Tensor):
        raise ValueError("request prompt-query shard tensors are invalid")
    if queries.ndim != 3 or positions.ndim != 2:
        raise ValueError("request prompt-query shard tensors have invalid rank")
    if queries.shape[:2] != positions.shape:
        raise ValueError("request prompt-query queries and positions are unaligned")
    if queries.shape != (num_q_heads, queries.shape[1], head_dim):
        raise ValueError("request prompt-query shard has invalid query geometry")
    if queries.dtype != torch.bfloat16 or positions.dtype not in {torch.int32, torch.int64}:
        raise ValueError("request prompt-query shard has invalid dtypes")
    if not torch.isfinite(queries).all():
        raise ValueError("request prompt-query shard contains non-finite queries")
    if torch.any(positions < 0) or torch.any(positions >= prompt_length):
        raise ValueError("request prompt-query positions are outside the prompt")
    samples = int(queries.shape[1])
    if samples_per_q_head is not None and samples != samples_per_q_head:
        raise ValueError("request prompt-query sample count does not match configuration")
    declared_samples = int(payload["sampling"].get("samples_per_q_head", -1))
    if declared_samples != samples:
        raise ValueError("request prompt-query sampling metadata does not match tensors")

    group_size = num_q_heads // num_kv_heads
    grouped_queries = queries.view(num_kv_heads, group_size, samples, head_dim).reshape(
        num_kv_heads, group_size * samples, head_dim
    )
    grouped_positions = positions.view(num_kv_heads, group_size, samples).reshape(
        num_kv_heads, group_size * samples
    )
    weights = torch.full(
        (num_kv_heads, group_size * samples),
        1.0 / float(group_size * samples),
        dtype=torch.float32,
    )
    return PromptQuerySupport(
        queries_by_kv_head=grouped_queries,
        positions_by_kv_head=grouped_positions,
        weights_by_kv_head=weights,
        source_path=path,
    )
