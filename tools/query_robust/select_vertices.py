"""Select Q-Filters-guided support vertices for Query-Robust assets.

The first directions are obtained from an uncentered query SVD, following
Q-Filters.  Actual captured queries are then selected at both support extrema
of those directions.  Spherical FPS fills the remaining vertex budget so the
asset still covers residual query variation needed by the Query-Robust hull.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
            if "meta" in payload:
                raw_meta = payload["meta"].item() if payload["meta"].ndim == 0 else payload["meta"]
                payload["meta"] = json.loads(str(raw_meta))
            return payload
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Query capture {path} must contain a mapping payload.")
    return payload


def _normalize_direction(direction: torch.Tensor) -> torch.Tensor | None:
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or float(norm) <= torch.finfo(torch.float32).eps:
        return None
    return direction / norm


def _qfilters_directions(
    queries: torch.Tensor,
    *,
    q_head_ids: torch.Tensor | None,
    num_components: int,
) -> list[torch.Tensor]:
    """Return a Q-Filters-style principal direction plus residual SVD axes.

    Q-Filters computes an uncentered SVD per query head and averages the
    sign-aligned first right vector within a GQA group.  Query-Robust still
    needs a hull rather than one direction, so the remaining directions come
    from a pooled SVD and are orthogonalized against that group direction.
    """

    if num_components <= 0:
        return []
    per_head: list[torch.Tensor] = []
    if q_head_ids is not None:
        q_head_ids = q_head_ids.to(device=queries.device, dtype=torch.long).reshape(-1)
        if q_head_ids.numel() != queries.shape[0]:
            raise ValueError("q_head_ids must have one entry per query sample.")
        head_values = torch.unique(q_head_ids[q_head_ids.ge(0)], sorted=True)
    else:
        head_values = torch.empty(0, dtype=torch.long, device=queries.device)

    if head_values.numel() == 0:
        head_values = torch.tensor([-1], dtype=torch.long, device=queries.device)

    for head_id in head_values.tolist():
        head_queries = queries if head_id < 0 else queries[q_head_ids == head_id]
        if head_queries.shape[0] < 2:
            continue
        _, _, vh = torch.linalg.svd(head_queries.float(), full_matrices=False)
        direction = _normalize_direction(vh[0])
        if direction is None:
            continue
        mean_projection = torch.dot(head_queries.mean(dim=0), direction)
        if float(mean_projection) < 0:
            direction = -direction
        per_head.append(direction)

    directions: list[torch.Tensor] = []
    if per_head:
        qfilter = _normalize_direction(torch.stack(per_head).sum(dim=0))
        if qfilter is not None:
            directions.append(qfilter)

    _, _, pooled_vh = torch.linalg.svd(queries.float(), full_matrices=False)
    for candidate in pooled_vh[:num_components]:
        residual = candidate.float()
        if directions:
            basis = torch.stack(directions)
            residual = residual - (residual @ basis.transpose(0, 1)) @ basis
        residual = _normalize_direction(residual)
        if residual is not None:
            directions.append(residual)
        if len(directions) >= num_components:
            break
    return directions[:num_components]


def select_group(
    queries: torch.Tensor,
    num_vertices: int,
    *,
    q_head_ids: torch.Tensor | None = None,
    qfilter_components: int = 4,
) -> tuple[torch.Tensor, int]:
    """Select Q-Filters support extrema, then fill with spherical FPS."""

    if queries.ndim != 2 or queries.shape[0] <= 0:
        raise ValueError(f"Expected non-empty [samples, D] queries, got {tuple(queries.shape)}.")
    queries = queries.float().contiguous()
    if not torch.isfinite(queries).all():
        raise ValueError("Query captures contain non-finite values.")
    if num_vertices < 2:
        raise ValueError("num_vertices must be at least 2.")
    if qfilter_components < 0:
        raise ValueError("qfilter_components must be non-negative.")
    mean = queries.mean(dim=0)
    centered = queries - mean
    radii = torch.linalg.vector_norm(centered, dim=-1)
    directions = centered / radii.clamp_min(torch.finfo(torch.float32).eps).unsqueeze(-1)
    selected_vertices: list[int] = []
    selected_set: set[int] = set()
    remaining = torch.ones(
        queries.shape[0], dtype=torch.bool, device=queries.device
    )
    selected_direction_vectors: list[torch.Tensor] = []

    def add_support_points(direction: torch.Tensor) -> None:
        if len(selected_vertices) >= num_vertices:
            return
        direction = direction.to(dtype=queries.dtype, device=queries.device)
        selected_direction_vectors.append(direction)
        support_scores = queries @ direction
        for support_index in (
            int(torch.argmax(support_scores).item()),
            int(torch.argmin(support_scores).item()),
        ):
            if len(selected_vertices) >= num_vertices:
                break
            if support_index not in selected_set:
                selected_set.add(support_index)
                selected_vertices.append(support_index)
                remaining[support_index] = False

    for direction in _qfilters_directions(
        queries,
        q_head_ids=q_head_ids,
        num_components=qfilter_components,
    ):
        add_support_points(direction)

    if not selected_direction_vectors and bool(remaining.any()):
        first = int(torch.argmax(radii).item())
        selected_direction_vectors.append(directions[first])
        remaining[first] = False

    while len(selected_vertices) < num_vertices and bool(remaining.any()):
        selected = torch.stack(selected_direction_vectors)
        max_cosine = directions @ selected.transpose(0, 1)
        min_max_cosine = max_cosine.max(dim=-1).values
        min_max_cosine = min_max_cosine.masked_fill(~remaining, float("inf"))
        direction_index = int(torch.argmin(min_max_cosine).item())
        direction = directions[direction_index]
        selected_direction_vectors.append(direction)
        remaining[direction_index] = False
        support_index = int(torch.argmax(queries @ direction).item())
        if support_index not in selected_set:
            selected_set.add(support_index)
            selected_vertices.append(support_index)
            remaining[support_index] = False

    if not selected_vertices:
        raise RuntimeError("Vertex selection did not produce a support point.")
    valid_count = len(selected_vertices)
    while len(selected_vertices) < num_vertices:
        # Repeating a real support point preserves the convex hull.  Zero
        # padding would create a fake query and inflate every certificate.
        selected_vertices.append(selected_vertices[-1])
    return queries[selected_vertices], valid_count


def select_vertices(
    input_path: Path,
    output_path: Path,
    *,
    num_vertices: int,
    model_id: str | None,
    model_fingerprint: str | None,
    tp_world_size: int | None,
    rope_config: Any | None,
    device: str = "cpu",
) -> None:
    payload = _load_payload(input_path)
    queries = payload.get("queries")
    if isinstance(queries, np.ndarray):
        queries = torch.from_numpy(queries)
    if not isinstance(queries, torch.Tensor) or queries.ndim != 4:
        raise ValueError("Collected queries must have shape [layers, kv_heads, samples, D].")
    try:
        selection_device = torch.device(device)
    except RuntimeError as error:
        raise ValueError(f"Invalid selection device {device!r}.") from error
    if selection_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA query selection was requested but CUDA is unavailable.")
    queries = queries.float().to(selection_device)
    num_samples = payload.get("num_samples")
    if num_samples is None:
        num_samples = torch.full(queries.shape[:2], queries.shape[2], dtype=torch.int32)
    elif isinstance(num_samples, np.ndarray):
        num_samples = torch.from_numpy(num_samples)
    if not isinstance(num_samples, torch.Tensor) or tuple(num_samples.shape) != tuple(queries.shape[:2]):
        raise ValueError("num_samples must have shape [layers, kv_heads].")
    layers, kv_heads, _, dim = map(int, queries.shape)
    q_head_ids = payload.get("q_head_ids")
    if isinstance(q_head_ids, np.ndarray):
        q_head_ids = torch.from_numpy(q_head_ids)
    if q_head_ids is not None and (
        not isinstance(q_head_ids, torch.Tensor)
        or tuple(q_head_ids.shape) != tuple(queries.shape[:3])
    ):
        raise ValueError("q_head_ids must have shape [layers, kv_heads, samples].")

    vertices = torch.empty(
        (layers, kv_heads, num_vertices, dim), dtype=torch.bfloat16
    )
    valid = torch.empty((layers, kv_heads), dtype=torch.int32)
    for layer in range(layers):
        for kv_head in range(kv_heads):
            count = int(num_samples[layer, kv_head])
            if count < 2:
                raise ValueError(
                    f"Need at least two query samples for layer={layer}, kv_head={kv_head}; got {count}."
                )
            selected, valid_count = select_group(
                queries[layer, kv_head, :count],
                num_vertices,
                q_head_ids=(
                    q_head_ids[layer, kv_head, :count].to(selection_device)
                    if q_head_ids is not None
                    else None
                ),
            )
            if valid_count < 2:
                raise ValueError(
                    f"Query samples for layer={layer}, kv_head={kv_head} do not contain "
                    "two distinct support vertices."
                )
            vertices[layer, kv_head] = selected.to(device="cpu", dtype=torch.bfloat16)
            valid[layer, kv_head] = valid_count

    input_meta = payload.get("meta", {})
    if not isinstance(input_meta, dict):
        raise TypeError("Collected query metadata must be a mapping.")
    meta = {
        **input_meta,
        "num_layers": layers,
        "num_kv_heads": kv_heads,
        "head_dim": dim,
        "M": num_vertices,
        "vertex_selection": "qfilters_svd_support_fps",
        "qfilter_components": 4,
        "selection_device": str(selection_device),
        "tp_world_size": int(tp_world_size)
        if tp_world_size is not None
        else input_meta.get("tp_world_size", 1),
    }
    if model_id is not None:
        meta["model_id"] = model_id
    if model_fingerprint is not None:
        meta["model_fingerprint"] = model_fingerprint
    if rope_config is not None:
        meta["rope_config"] = rope_config
    if "model_id" not in meta:
        raise ValueError(
            "A Query-Robust asset must include model_id; pass --model-id or provide it in the capture metadata."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vertices": vertices,
        "num_valid_vertices": valid,
        "meta": meta,
    }
    if output_path.suffix.lower() == ".npz":
        np.savez_compressed(
            output_path,
            vertices=vertices.view(torch.uint16).numpy(),
            num_valid_vertices=valid.numpy(),
            meta=json.dumps(meta, sort_keys=True),
        )
    else:
        torch.save(payload, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-vertices", type=int, default=64)
    parser.add_argument("--model-id")
    parser.add_argument("--model-fingerprint")
    parser.add_argument("--tp-world-size", type=int)
    parser.add_argument("--rope-config-json", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rope_config = (
        json.loads(args.rope_config_json.read_text())
        if args.rope_config_json is not None
        else None
    )
    select_vertices(
        args.input,
        args.output,
        num_vertices=args.num_vertices,
        model_id=args.model_id,
        model_fingerprint=args.model_fingerprint,
        tp_world_size=args.tp_world_size,
        rope_config=rope_config,
        device=args.device,
    )


if __name__ == "__main__":
    main()
