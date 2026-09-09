"""Validate a Query-Robust BF16 vertex asset before serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
            if "meta" in payload:
                raw_meta = payload["meta"].item() if payload["meta"].ndim == 0 else payload["meta"]
                payload["meta"] = json.loads(str(raw_meta))
            return payload
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Asset {path} must contain a mapping payload.")
    return payload


def validate(
    path: Path,
    *,
    expected_model_id: str | None = None,
    expected_layers: int | None = None,
    expected_kv_heads: int | None = None,
    expected_head_dim: int | None = None,
    expected_vertices: int | None = None,
    expected_tp_world_size: int | None = None,
) -> dict[str, Any]:
    payload = _load(path)
    vertices = payload.get("vertices")
    valid = payload.get("num_valid_vertices")
    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices)
        if vertices.dtype == torch.uint16:
            vertices = vertices.view(torch.bfloat16)
    if isinstance(valid, np.ndarray):
        valid = torch.from_numpy(valid)
    if not isinstance(vertices, torch.Tensor) or vertices.ndim != 4:
        raise ValueError("vertices must have shape [layers, kv_heads, M, D].")
    if not isinstance(valid, torch.Tensor) or valid.ndim != 2:
        raise ValueError("num_valid_vertices must have shape [layers, kv_heads].")
    layers, kv_heads, num_vertices, dim = map(int, vertices.shape)
    if tuple(valid.shape) != (layers, kv_heads):
        raise ValueError("num_valid_vertices shape does not match vertices.")
    if vertices.dtype != torch.bfloat16:
        raise TypeError(f"vertices must be stored as BF16, got {vertices.dtype}.")
    if not torch.isfinite(vertices.float()).all():
        raise ValueError("vertices contain non-finite values.")
    if bool((valid < 2).any()) or bool((valid > num_vertices).any()):
        raise ValueError("num_valid_vertices must be in [2, M].")
    for layer in range(layers):
        for kv_head in range(kv_heads):
            count = int(valid[layer, kv_head])
            if count == num_vertices:
                continue
            real_vertices = vertices[layer, kv_head, :count]
            padded_vertices = vertices[layer, kv_head, count:]
            is_repeat = (padded_vertices[:, None, :] == real_vertices[None, :, :]).all(dim=-1).any(dim=-1)
            if not bool(is_repeat.all()):
                raise ValueError(
                    "padded vertices must repeat one of the valid support vertices "
                    f"at layer={layer}, kv_head={kv_head}."
                )
    checks = {
        "layers": expected_layers,
        "kv_heads": expected_kv_heads,
        "head_dim": expected_head_dim,
        "vertices": expected_vertices,
    }
    actual = {
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": dim,
        "vertices": num_vertices,
    }
    for name, expected in checks.items():
        if expected is not None and actual[name] != int(expected):
            raise ValueError(f"{name} mismatch: expected={expected} got={actual[name]}.")
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise TypeError("asset meta must be a mapping.")
    if "tp_world_size" not in meta:
        raise ValueError("asset metadata must declare tp_world_size.")
    model_id = meta.get("model_id")
    if expected_model_id is not None and model_id is not None:
        if str(model_id).rstrip("/").split("/")[-1] != str(expected_model_id).rstrip("/").split("/")[-1]:
            raise ValueError(f"model_id mismatch: expected={expected_model_id!r} got={model_id!r}.")
    if expected_tp_world_size is not None:
        actual_tp = int(meta["tp_world_size"])
        if actual_tp != int(expected_tp_world_size):
            raise ValueError(
                f"tp_world_size mismatch: expected={expected_tp_world_size} got={actual_tp}."
            )
    return {
        "path": str(path),
        "shape": [layers, kv_heads, num_vertices, dim],
        "dtype": str(vertices.dtype),
        "min_valid_vertices": int(valid.min()),
        "max_valid_vertices": int(valid.max()),
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--num-vertices", type=int)
    parser.add_argument("--tp-world-size", type=int)
    args = parser.parse_args()
    report = validate(
        args.asset,
        expected_model_id=args.model_id,
        expected_layers=args.layers,
        expected_kv_heads=args.kv_heads,
        expected_head_dim=args.head_dim,
        expected_vertices=args.num_vertices,
        expected_tp_world_size=args.tp_world_size,
    )
    print(report)


if __name__ == "__main__":
    main()
