"""Versioned empirical-query artifacts for Query-Robust attention."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor


SCHEMA_VERSION = 1
REPRESENTATION = "prompt_origin_aligned_pre_scale"


@dataclass(frozen=True)
class QueryPoolManifest:
    schema_version: int
    model_id: str
    model_revision: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    tp_size: int
    rope_type: str
    rope_parameters: dict[str, Any]
    representation: str
    attention_scale: float
    max_decode_offset: int
    pool_size_per_kv_group: int
    coreset_sizes: list[int]
    capture_split: str
    source_manifest_sha256: str
    created_with_git_commit: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported query-pool schema version {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        positive = {
            "num_layers": self.num_layers,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "tp_size": self.tp_size,
            "pool_size_per_kv_group": self.pool_size_per_kv_group,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"manifest {name} must be a positive integer")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("manifest query heads must be divisible by KV heads")
        if self.num_kv_heads % self.tp_size != 0:
            raise ValueError("manifest KV heads must be divisible by TP size")
        if self.max_decode_offset < 0:
            raise ValueError("manifest max decode offset must be nonnegative")
        if not math.isfinite(self.attention_scale) or self.attention_scale <= 0:
            raise ValueError("manifest attention scale must be positive and finite")
        if not self.model_id or not self.model_revision or not self.rope_type:
            raise ValueError("manifest model and RoPE identifiers cannot be empty")
        if self.representation != REPRESENTATION:
            raise ValueError(
                f"unsupported query representation {self.representation!r}; expected {REPRESENTATION!r}"
            )
        normalized_sizes = sorted(set(self.coreset_sizes))
        if normalized_sizes != self.coreset_sizes:
            raise ValueError("manifest coreset sizes must be sorted and unique")
        if any(size < 1 or size > self.pool_size_per_kv_group for size in self.coreset_sizes):
            raise ValueError("manifest coreset sizes must be within the full pool size")
        if len(self.source_manifest_sha256) != 64:
            raise ValueError("source manifest SHA256 must contain 64 hexadecimal characters")
        try:
            int(self.source_manifest_sha256, 16)
        except ValueError as exc:
            raise ValueError("source manifest SHA256 is not hexadecimal") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QueryPoolManifest":
        if not isinstance(payload, dict):
            raise ValueError("query-pool manifest must be a JSON object")
        expected = {field.name for field in fields(cls)}
        actual = set(payload)
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        if unexpected:
            raise ValueError(f"query-pool manifest has unexpected fields: {unexpected}")
        if missing:
            raise ValueError(f"query-pool manifest is missing fields: {missing}")
        return cls(**payload)


@dataclass(frozen=True)
class QueryPoolExpectations:
    model_id: str
    model_revision: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    tp_size: int
    rope_type: str
    rope_parameters: dict[str, Any]
    representation: str
    attention_scale: float


@dataclass
class QueryPoolLayer:
    centroid_by_horizon: Tensor
    coreset_queries: dict[int, Tensor]
    coreset_weights: dict[int, Tensor]
    full_queries: Tensor | None = None
    full_weights: Tensor | None = None


@dataclass(frozen=True)
class QueryPool:
    manifest: QueryPoolManifest
    layers: tuple[QueryPoolLayer, ...]
    tp_rank: int


@dataclass(frozen=True)
class CoresetSelection:
    indices: Tensor
    weights: Tensor


@dataclass(frozen=True)
class CaptureRecord:
    sequence_id_hash: int
    task_index: int
    split: str


def _validate_manifest_matches(
    manifest: QueryPoolManifest, expected: QueryPoolExpectations
) -> None:
    exact_fields = (
        "model_id",
        "model_revision",
        "num_layers",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "tp_size",
        "rope_type",
        "rope_parameters",
        "representation",
    )
    for name in exact_fields:
        actual_value = getattr(manifest, name)
        expected_value = getattr(expected, name)
        if actual_value != expected_value:
            label = name.replace("_", " ")
            raise ValueError(
                f"query-pool {label} mismatch: artifact={actual_value!r}, expected={expected_value!r}"
            )
    if not math.isclose(
        manifest.attention_scale,
        expected.attention_scale,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "query-pool attention scale mismatch: "
            f"artifact={manifest.attention_scale}, expected={expected.attention_scale}"
        )


def _validate_layer(
    layer: QueryPoolLayer,
    manifest: QueryPoolManifest,
    include_full_queries: bool,
) -> None:
    local_kv_heads = manifest.num_kv_heads // manifest.tp_size
    expected_centroid = (
        local_kv_heads,
        manifest.max_decode_offset + 1,
        manifest.head_dim,
    )
    if layer.centroid_by_horizon.shape != expected_centroid:
        raise ValueError(
            f"query-pool centroid shape {tuple(layer.centroid_by_horizon.shape)} "
            f"does not match {expected_centroid}"
        )
    if layer.centroid_by_horizon.dtype != torch.float32:
        raise ValueError("query-pool centroids must use FP32")
    if not torch.isfinite(layer.centroid_by_horizon).all():
        raise ValueError("query-pool centroids must be finite")
    if set(layer.coreset_queries) != set(layer.coreset_weights):
        raise ValueError("query-pool coreset query and weight sizes differ")
    if set(layer.coreset_queries) != set(manifest.coreset_sizes):
        raise ValueError("query-pool coreset sizes do not match the manifest")
    for size in manifest.coreset_sizes:
        queries = layer.coreset_queries[size]
        weights = layer.coreset_weights[size]
        if queries.shape != (local_kv_heads, size, manifest.head_dim):
            raise ValueError(f"query-pool coreset R={size} has an invalid query shape")
        if weights.shape != (local_kv_heads, size):
            raise ValueError(f"query-pool coreset R={size} has an invalid weight shape")
        if queries.dtype != torch.bfloat16 or weights.dtype != torch.float32:
            raise ValueError("query-pool coresets must use BF16 queries and FP32 weights")
        if not torch.isfinite(queries).all() or not torch.isfinite(weights).all():
            raise ValueError("query-pool coresets must be finite")
        if torch.any(weights < 0):
            raise ValueError("query-pool coreset weights must be nonnegative")
        torch.testing.assert_close(weights.sum(-1), torch.ones(local_kv_heads))
    if include_full_queries:
        if layer.full_queries is None or layer.full_weights is None:
            raise ValueError("query-pool full offline shard is missing")
        expected_queries = (
            local_kv_heads,
            manifest.pool_size_per_kv_group,
            manifest.head_dim,
        )
        expected_weights = (
            local_kv_heads,
            manifest.pool_size_per_kv_group,
        )
        if layer.full_queries.shape != expected_queries or layer.full_weights.shape != expected_weights:
            raise ValueError("query-pool full offline tensors have invalid shapes")
        if layer.full_queries.dtype != torch.bfloat16 or layer.full_weights.dtype != torch.float32:
            raise ValueError("query-pool full offline tensors have invalid dtypes")
        if torch.any(layer.full_weights < 0):
            raise ValueError("query-pool full weights must be nonnegative")
        torch.testing.assert_close(layer.full_weights.sum(-1), torch.ones(local_kv_heads))


def _layer_to_payload(layer: QueryPoolLayer) -> dict[str, Any]:
    return {
        "centroid_by_horizon": layer.centroid_by_horizon.cpu(),
        "coreset_queries": {
            str(size): tensor.cpu() for size, tensor in layer.coreset_queries.items()
        },
        "coreset_weights": {
            str(size): tensor.cpu() for size, tensor in layer.coreset_weights.items()
        },
    }


def _full_to_payload(layer: QueryPoolLayer) -> dict[str, Tensor]:
    if layer.full_queries is None or layer.full_weights is None:
        raise ValueError("cannot serialize an absent full query pool")
    return {
        "queries": layer.full_queries.cpu(),
        "weights": layer.full_weights.cpu(),
    }


def write_query_pool(
    manifest: QueryPoolManifest,
    layers: Sequence[QueryPoolLayer],
    output_dir: str | Path,
) -> Path:
    """Atomically publish an already finalized query pool."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"query-pool output already exists: {output}")
    if len(layers) != manifest.num_layers:
        raise ValueError("query-pool layer count does not match the manifest")
    for layer in layers:
        _validate_layer(layer, manifest, include_full_queries=layer.full_queries is not None)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        (temporary / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if any(layer.full_queries is not None for layer in layers):
            (temporary / "full").mkdir()
        for layer_idx, layer in enumerate(layers):
            online_path = temporary / f"layer_{layer_idx:03d}_rank_000.pt"
            torch.save(_layer_to_payload(layer), online_path)
            if layer.full_queries is not None:
                torch.save(
                    _full_to_payload(layer),
                    temporary / "full" / f"layer_{layer_idx:03d}_rank_000.pt",
                )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _safe_load_shard(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"query-pool layer shard is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"failed to load corrupt query-pool shard {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"query-pool shard {path} is not a tensor dictionary")
    return payload


def load_query_pool(
    path: str | Path,
    expected: QueryPoolExpectations,
    tp_rank: int,
    include_full_queries: bool = False,
) -> QueryPool:
    """Load and validate only the online state unless the offline pool is requested."""

    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"query-pool manifest is missing: {manifest_path}")
    try:
        manifest = QueryPoolManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid query-pool manifest {manifest_path}: {exc}") from exc
    _validate_manifest_matches(manifest, expected)
    if not isinstance(tp_rank, int) or isinstance(tp_rank, bool) or not 0 <= tp_rank < manifest.tp_size:
        raise ValueError(f"TP rank must be in [0, {manifest.tp_size}), got {tp_rank!r}")

    layers: list[QueryPoolLayer] = []
    for layer_idx in range(manifest.num_layers):
        online = _safe_load_shard(
            root / f"layer_{layer_idx:03d}_rank_{tp_rank:03d}.pt"
        )
        required = {"centroid_by_horizon", "coreset_queries", "coreset_weights"}
        if set(online) != required:
            raise ValueError(f"query-pool online shard {layer_idx} has invalid fields")
        try:
            coreset_queries = {
                int(size): tensor for size, tensor in online["coreset_queries"].items()
            }
            coreset_weights = {
                int(size): tensor for size, tensor in online["coreset_weights"].items()
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"query-pool online shard {layer_idx} has invalid coresets") from exc
        full_queries = None
        full_weights = None
        if include_full_queries:
            full = _safe_load_shard(
                root / "full" / f"layer_{layer_idx:03d}_rank_{tp_rank:03d}.pt"
            )
            if set(full) != {"queries", "weights"}:
                raise ValueError(f"query-pool full shard {layer_idx} has invalid fields")
            full_queries = full["queries"]
            full_weights = full["weights"]
        layer = QueryPoolLayer(
            centroid_by_horizon=online["centroid_by_horizon"],
            coreset_queries=coreset_queries,
            coreset_weights=coreset_weights,
            full_queries=full_queries,
            full_weights=full_weights,
        )
        try:
            _validate_layer(layer, manifest, include_full_queries)
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise ValueError(f"invalid query-pool layer {layer_idx}: {exc}") from exc
        layers.append(layer)
    return QueryPool(manifest=manifest, layers=tuple(layers), tp_rank=tp_rank)


def balanced_empirical_weights(query_head_id: Tensor, stratum_id: Tensor) -> Tensor:
    """Balance head mass, then stratum mass, then samples within each stratum."""

    if query_head_id.ndim != 1 or stratum_id.ndim != 1 or query_head_id.shape != stratum_id.shape:
        raise ValueError("query head and stratum IDs must be aligned vectors")
    if query_head_id.numel() == 0:
        raise ValueError("cannot weight an empty empirical query set")
    heads = torch.unique(query_head_id, sorted=True)
    weights = torch.zeros(query_head_id.shape, dtype=torch.float64)
    head_mass = 1.0 / heads.numel()
    for head in heads.tolist():
        head_mask = query_head_id == head
        strata = torch.unique(stratum_id[head_mask], sorted=True)
        stratum_mass = head_mass / strata.numel()
        for stratum in strata.tolist():
            mask = head_mask & (stratum_id == stratum)
            weights[mask] = stratum_mass / int(mask.sum().item())
    return weights / weights.sum()


def select_response_coreset(
    queries: Tensor,
    features: Tensor,
    weights: Tensor,
    size: int,
    seed: int,
) -> CoresetSelection:
    """Select real queries by deterministic response-space kernel herding."""

    if queries.ndim != 2 or features.ndim != 2 or weights.ndim != 1:
        raise ValueError("coreset inputs must be queries/features matrices and a weight vector")
    if queries.shape[0] != features.shape[0] or queries.shape[0] != weights.shape[0]:
        raise ValueError("coreset inputs have different sample counts")
    if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= queries.shape[0]:
        raise ValueError("coreset size must be within the candidate count")
    if not torch.isfinite(queries).all() or not torch.isfinite(features).all() or not torch.isfinite(weights).all():
        raise ValueError("coreset inputs must be finite")
    if torch.any(weights < 0) or not bool(weights.sum() > 0):
        raise ValueError("coreset weights must be nonnegative with positive sum")

    feature64 = features.detach().cpu().to(torch.float64)
    weight64 = weights.detach().cpu().to(torch.float64)
    weight64 = weight64 / weight64.sum()
    target = weight64 @ feature64
    generator = torch.Generator().manual_seed(seed)
    priority = torch.randperm(queries.shape[0], generator=generator).tolist()
    selected: list[int] = []
    running = torch.zeros_like(target)
    for count in range(size):
        best_index = None
        best_error = None
        for index in priority:
            if index in selected:
                continue
            candidate_mean = (running + feature64[index]) / (count + 1)
            error = float(torch.linalg.vector_norm(candidate_mean - target).item())
            if best_error is None or error < best_error:
                best_error = error
                best_index = index
        if best_index is None:
            raise RuntimeError("coreset selection exhausted candidates unexpectedly")
        selected.append(best_index)
        running += feature64[best_index]

    from scipy.optimize import nnls
    import numpy as np

    selected_features = feature64[selected]
    constraint_scale = max(10.0, math.sqrt(features.shape[1]))
    matrix = torch.cat(
        [selected_features.T, torch.full((1, size), constraint_scale, dtype=torch.float64)],
        dim=0,
    )
    target_augmented = torch.cat(
        [target, torch.tensor([constraint_scale], dtype=torch.float64)]
    )
    fitted, _ = nnls(matrix.numpy(), target_augmented.numpy())
    fitted_tensor = torch.from_numpy(np.asarray(fitted)).to(torch.float64)
    if not bool(fitted_tensor.sum() > 0):
        fitted_tensor.fill_(1.0 / size)
    else:
        fitted_tensor /= fitted_tensor.sum()
    return CoresetSelection(
        indices=torch.tensor(selected, dtype=torch.long),
        weights=fitted_tensor.to(torch.float32),
    )


def validate_no_split_overlap(records: Sequence[CaptureRecord]) -> None:
    """Reject sequence hashes or task indices appearing in multiple splits."""

    by_sequence: dict[int, set[str]] = {}
    by_task_index: dict[int, set[str]] = {}
    for record in records:
        by_sequence.setdefault(record.sequence_id_hash, set()).add(record.split)
        by_task_index.setdefault(record.task_index, set()).add(record.split)
    sequence_overlap = sorted(key for key, splits in by_sequence.items() if len(splits) > 1)
    if sequence_overlap:
        raise ValueError(f"sequence split overlap detected for hashes {sequence_overlap}")
    index_overlap = sorted(key for key, splits in by_task_index.items() if len(splits) > 1)
    if index_overlap:
        raise ValueError(f"task_index split overlap detected for indices {index_overlap}")


def _balanced_selection_indices(
    head_ids: Tensor, strata: Tensor, size: int, seed: int
) -> Tensor:
    if size > head_ids.numel():
        raise ValueError(
            f"requested pool size {size} exceeds {head_ids.numel()} captured candidates"
        )
    generator = torch.Generator().manual_seed(seed)
    priority = torch.randperm(head_ids.numel(), generator=generator).tolist()
    buckets: dict[tuple[int, int], list[int]] = {}
    for index in priority:
        key = (int(head_ids[index].item()), int(strata[index].item()))
        buckets.setdefault(key, []).append(index)
    selected: list[int] = []
    ordered_keys = sorted(buckets)
    while len(selected) < size:
        made_progress = False
        for key in ordered_keys:
            if buckets[key] and len(selected) < size:
                selected.append(buckets[key].pop(0))
                made_progress = True
        if not made_progress:
            break
    if len(selected) != size:
        raise RuntimeError("balanced query selection could not fill the requested pool")
    return torch.tensor(selected, dtype=torch.long)


def finalize_capture(
    input_dir: str | Path,
    output_dir: str | Path,
    pool_size: int,
    coreset_sizes: Sequence[int],
    seed: int,
) -> Path:
    """Finalize raw prompt-origin query steps into online and offline shards."""

    root = Path(input_dir)
    try:
        manifest = QueryPoolManifest.from_dict(
            json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid raw capture manifest: {exc}") from exc
    if pool_size < 1:
        raise ValueError("pool size must be positive")
    normalized_coresets = sorted(set(int(size) for size in coreset_sizes))
    if any(size < 1 or size > pool_size for size in normalized_coresets):
        raise ValueError("coreset sizes must be positive and no larger than the pool")

    if manifest.tp_size != 1:
        raise ValueError(
            "raw capture finalization v1 supports TP=1 only; rank-local TP merging is not implemented"
        )
    steps = sorted(root.rglob("step*.pt"))
    if not steps:
        raise ValueError("raw capture contains no tensor step shards")
    raw_queries: dict[int, list[Tensor]] = {
        layer: [] for layer in range(manifest.num_layers)
    }
    offsets: dict[int, list[int]] = {
        layer: [] for layer in range(manifest.num_layers)
    }
    strata: dict[int, list[int]] = {
        layer: [] for layer in range(manifest.num_layers)
    }
    records: list[CaptureRecord] = []
    for step in steps:
        payload = _safe_load_shard(step)
        required = {"prompt_origin_queries", "metadata"}
        if not required.issubset(payload):
            raise ValueError(f"raw capture shard {step} has invalid fields")
        queries = payload["prompt_origin_queries"]
        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(f"raw capture shard {step} has invalid metadata")
        if str(metadata.get("split")) != "calibration":
            continue
        if "layer_ids" in payload:
            layer_ids_tensor = payload["layer_ids"]
            if not isinstance(layer_ids_tensor, Tensor) or layer_ids_tensor.ndim != 1:
                raise ValueError(f"raw capture shard {step} has invalid layer IDs")
            layer_ids = [int(layer) for layer in layer_ids_tensor.tolist()]
        else:
            layer_ids = list(range(manifest.num_layers))
        expected_shape = (len(layer_ids), manifest.num_q_heads, manifest.head_dim)
        if not isinstance(queries, Tensor) or queries.shape != expected_shape:
            raise ValueError(
                f"raw capture shard {step} query shape {getattr(queries, 'shape', None)} "
                f"does not match {expected_shape}"
            )
        if len(set(layer_ids)) != len(layer_ids) or any(
            layer < 0 or layer >= manifest.num_layers for layer in layer_ids
        ):
            raise ValueError(f"raw capture shard {step} has invalid layer IDs")
        offset = int(metadata["decode_offset"])
        if not 0 <= offset <= manifest.max_decode_offset:
            raise ValueError(f"raw capture offset {offset} is outside the manifest horizon")
        for local_layer, layer_idx in enumerate(layer_ids):
            raw_queries[layer_idx].append(queries[local_layer].to(torch.float32))
            offsets[layer_idx].append(offset)
            strata[layer_idx].append(int(metadata["stratum_id"]))
        records.append(
            CaptureRecord(
                sequence_id_hash=int(metadata["sequence_id_hash"]),
                task_index=int(metadata["task_index"]),
                split=str(metadata["split"]),
            )
        )
    validate_no_split_overlap(records)

    if normalized_coresets:
        raise ValueError(
            "response-space coreset finalization requires residual feature shards; "
            "finalize the full/centroid pool first"
        )

    group_size = manifest.num_q_heads // manifest.num_kv_heads
    layers: list[QueryPoolLayer] = []
    for layer_idx in range(manifest.num_layers):
        if not raw_queries[layer_idx]:
            raise ValueError(f"raw capture is missing calibration layer {layer_idx}")
        stacked = torch.stack(raw_queries[layer_idx], dim=0)
        offset_tensor = torch.tensor(offsets[layer_idx], dtype=torch.long)
        stratum_per_step = torch.tensor(strata[layer_idx], dtype=torch.long)
        centroid_heads: list[Tensor] = []
        full_heads: list[Tensor] = []
        weight_heads: list[Tensor] = []
        for kv_head in range(manifest.num_kv_heads):
            q_start = kv_head * group_size
            q_stop = q_start + group_size
            candidates = stacked[:, q_start:q_stop].reshape(-1, manifest.head_dim)
            candidate_heads = torch.arange(q_start, q_stop).repeat(stacked.shape[0])
            candidate_offsets = offset_tensor.repeat_interleave(group_size)
            candidate_strata = stratum_per_step.repeat_interleave(group_size)
            selection = _balanced_selection_indices(
                candidate_heads, candidate_strata, pool_size, seed + layer_idx * 1009 + kv_head
            )
            selected_queries = candidates[selection]
            selected_heads = candidate_heads[selection]
            selected_strata = candidate_strata[selection]
            selected_weights = balanced_empirical_weights(selected_heads, selected_strata)
            full_heads.append(selected_queries.to(torch.bfloat16))
            weight_heads.append(selected_weights.to(torch.float32))

            horizon_centroids: list[Tensor] = []
            for horizon in range(1, manifest.max_decode_offset + 2):
                mask = candidate_offsets < horizon
                if not bool(mask.any()):
                    raise ValueError(f"no captured queries cover decode horizon {horizon}")
                horizon_weights = balanced_empirical_weights(
                    candidate_heads[mask], candidate_strata[mask]
                )
                horizon_centroids.append(
                    horizon_weights.to(torch.float32) @ candidates[mask].to(torch.float32)
                )
            centroid_heads.append(torch.stack(horizon_centroids, dim=0))
        layers.append(
            QueryPoolLayer(
                centroid_by_horizon=torch.stack(centroid_heads, dim=0),
                coreset_queries={},
                coreset_weights={},
                full_queries=torch.stack(full_heads, dim=0),
                full_weights=torch.stack(weight_heads, dim=0),
            )
        )
    finalized_manifest = replace(
        manifest,
        pool_size_per_kv_group=pool_size,
        coreset_sizes=normalized_coresets,
    )
    return write_query_pool(finalized_manifest, layers, output_dir)
