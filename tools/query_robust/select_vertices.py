"""Select deterministic support-extreme vertices for Query-Robust assets."""

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


def select_group(queries: torch.Tensor, num_vertices: int) -> tuple[torch.Tensor, int]:
    """Run spherical FPS and raw-query support extrema for one KV group."""

    if queries.ndim != 2 or queries.shape[0] <= 0:
        raise ValueError(f"Expected non-empty [samples, D] queries, got {tuple(queries.shape)}.")
    queries = queries.float().contiguous()
    if not torch.isfinite(queries).all():
        raise ValueError("Query captures contain non-finite values.")
    if num_vertices < 2:
        raise ValueError("num_vertices must be at least 2.")
    mean = queries.mean(dim=0)
    centered = queries - mean
    radii = torch.linalg.vector_norm(centered, dim=-1)
    directions = centered / radii.clamp_min(torch.finfo(torch.float32).eps).unsqueeze(-1)
    first = int(torch.argmax(radii).item())
    selected_vertices: list[int] = []
    selected_directions: list[int] = []
    selected_set: set[int] = set()
    remaining = torch.ones(queries.shape[0], dtype=torch.bool)

    while len(selected_vertices) < num_vertices and bool(remaining.any()):
        if not selected_directions:
            direction_index = first
        else:
            selected = directions[selected_directions]
            max_cosine = directions @ selected.transpose(0, 1)
            min_max_cosine = max_cosine.max(dim=-1).values
            min_max_cosine = min_max_cosine.masked_fill(~remaining, float("inf"))
            direction_index = int(torch.argmin(min_max_cosine).item())
        selected_directions.append(direction_index)
        remaining[direction_index] = False
        support_scores = queries @ directions[direction_index]
        support_index = int(torch.argmax(support_scores).item())
        if support_index not in selected_set:
            selected_set.add(support_index)
            selected_vertices.append(support_index)

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
) -> None:
    payload = _load_payload(input_path)
    queries = payload.get("queries")
    if isinstance(queries, np.ndarray):
        queries = torch.from_numpy(queries)
    if not isinstance(queries, torch.Tensor) or queries.ndim != 4:
        raise ValueError("Collected queries must have shape [layers, kv_heads, samples, D].")
    queries = queries.float()
    num_samples = payload.get("num_samples")
    if num_samples is None:
        num_samples = torch.full(queries.shape[:2], queries.shape[2], dtype=torch.int32)
    elif isinstance(num_samples, np.ndarray):
        num_samples = torch.from_numpy(num_samples)
    if not isinstance(num_samples, torch.Tensor) or tuple(num_samples.shape) != tuple(queries.shape[:2]):
        raise ValueError("num_samples must have shape [layers, kv_heads].")
    layers, kv_heads, _, dim = map(int, queries.shape)
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
                queries[layer, kv_head, :count], num_vertices
            )
            if valid_count < 2:
                raise ValueError(
                    f"Query samples for layer={layer}, kv_head={kv_head} do not contain "
                    "two distinct support vertices."
                )
            vertices[layer, kv_head] = selected.to(torch.bfloat16)
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
    )


if __name__ == "__main__":
    main()
