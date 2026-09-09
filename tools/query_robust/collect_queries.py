"""Collect post-RoPE decode queries for Query-Robust vertex construction.

Inputs are torch/NumPy payloads containing ``queries`` with shape
``[samples, layers, query_heads, head_dim]``.  The collector keeps a bounded,
deterministic reservoir for every ``(layer, kv_head, q_head, bucket)`` group
and writes the pooled samples plus their provenance to a torch asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            payload: dict[str, Any] = {key: data[key] for key in data.files}
            if "meta" in payload and isinstance(payload["meta"], np.ndarray):
                raw = payload["meta"].item() if payload["meta"].ndim == 0 else payload["meta"]
                payload["meta"] = json.loads(raw) if isinstance(raw, str) else raw
            return payload
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"Query capture {path} must contain a mapping payload.")
    return loaded


def _as_tensor(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (list, tuple)):
        return torch.as_tensor(value)
    raise TypeError(f"Capture field {name!r} must be a tensor or ndarray.")


def _bucket_label(context_length: int) -> str:
    length = int(context_length)
    if length <= 8192:
        return "8k"
    if length <= 16384:
        return "16k"
    if length <= 32768:
        return "32k"
    return "64k+"


def _stable_rng(seed: int, key: tuple[int, int, int, str]) -> np.random.Generator:
    encoded = repr((int(seed), *key)).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def collect(
    inputs: list[Path],
    *,
    output: Path,
    num_kv_heads: int,
    max_per_group: int,
    seed: int,
    extra_meta: dict[str, Any],
) -> None:
    if num_kv_heads <= 0 or max_per_group <= 0:
        raise ValueError("num_kv_heads and max_per_group must be positive.")

    groups: dict[tuple[int, int, int, str], list[tuple[torch.Tensor, int, str, int]]] = {}
    group_counts: dict[tuple[int, int, int, str], int] = {}
    group_rngs: dict[tuple[int, int, int, str], np.random.Generator] = {}
    global_meta: dict[str, Any] = {}
    num_layers: int | None = None
    query_heads: int | None = None
    head_dim: int | None = None

    for path in inputs:
        payload = _load_payload(path)
        queries = _as_tensor(payload.get("queries"), name="queries").float()
        if queries.ndim != 4:
            raise ValueError(
                f"{path}: queries must have shape [samples, layers, q_heads, D], "
                f"got {tuple(queries.shape)}."
            )
        samples, layers, q_heads, dim = map(int, queries.shape)
        if q_heads % num_kv_heads:
            raise ValueError(
                f"{path}: query heads={q_heads} must be divisible by KV heads={num_kv_heads}."
            )
        if num_layers is None:
            num_layers, query_heads, head_dim = layers, q_heads, dim
        elif (layers, q_heads, dim) != (num_layers, query_heads, head_dim):
            raise ValueError(
                f"{path}: query shape changed across captures: "
                f"expected=({num_layers}, {query_heads}, {head_dim}) "
                f"got=({layers}, {q_heads}, {dim})."
            )
        context_lengths = payload.get("context_lengths")
        if context_lengths is None:
            context_lengths = np.full((samples,), 0, dtype=np.int64)
        context_lengths = _as_tensor(context_lengths, name="context_lengths").reshape(-1)
        if int(context_lengths.numel()) != samples:
            raise ValueError(f"{path}: context_lengths must have one entry per sample.")
        bucket_values = payload.get("context_buckets")
        if bucket_values is not None:
            if isinstance(bucket_values, torch.Tensor):
                bucket_values = bucket_values.detach().cpu().tolist()
            elif isinstance(bucket_values, np.ndarray):
                bucket_values = bucket_values.tolist()
            bucket_values = [str(value) for value in bucket_values]
            if len(bucket_values) != samples:
                raise ValueError(f"{path}: context_buckets must have one entry per sample.")
        else:
            bucket_values = [_bucket_label(int(value)) for value in context_lengths.tolist()]
        decode_steps = payload.get("decode_steps")
        if decode_steps is None:
            decode_steps = torch.arange(samples, dtype=torch.int64)
        decode_steps = _as_tensor(decode_steps, name="decode_steps").reshape(-1)
        if int(decode_steps.numel()) != samples:
            raise ValueError(f"{path}: decode_steps must have one entry per sample.")

        global_meta.update(
            {
                key: value
                for key, value in dict(payload.get("meta", {})).items()
                if key not in {"num_layers", "num_kv_heads", "head_dim"}
            }
        )
        for sample in range(samples):
            bucket = bucket_values[sample]
            step = int(decode_steps[sample])
            for layer in range(layers):
                for q_head in range(q_heads):
                    kv_head = q_head // (q_heads // num_kv_heads)
                    key = (layer, kv_head, q_head, bucket)
                    values = groups.setdefault(key, [])
                    seen = group_counts.get(key, 0)
                    value = (queries[sample, layer, q_head], q_head, bucket, step)
                    if len(values) < max_per_group:
                        values.append(value)
                    else:
                        rng = group_rngs.setdefault(key, _stable_rng(seed, key))
                        replacement = int(rng.integers(0, seen + 1))
                        if replacement < max_per_group:
                            values[replacement] = value
                    group_counts[key] = seen + 1

    assert num_layers is not None and query_heads is not None and head_dim is not None
    pooled: dict[tuple[int, int], list[tuple[torch.Tensor, int, str, int]]] = {}
    for (layer, kv_head, _q_head, _bucket), values in sorted(groups.items()):
        pooled.setdefault((layer, kv_head), []).extend(values)

    max_samples = max(len(values) for values in pooled.values()) if pooled else 0
    if max_samples == 0:
        raise ValueError("No query samples were collected.")
    queries_out = torch.zeros(
        (num_layers, num_kv_heads, max_samples, head_dim), dtype=torch.float32
    )
    num_samples = torch.zeros((num_layers, num_kv_heads), dtype=torch.int32)
    q_heads_out = torch.full(
        (num_layers, num_kv_heads, max_samples), -1, dtype=torch.int32
    )
    buckets_out = np.full(
        (num_layers, num_kv_heads, max_samples), "", dtype="U16"
    )
    steps_out = torch.full(
        (num_layers, num_kv_heads, max_samples), -1, dtype=torch.int64
    )
    for (layer, kv_head), values in sorted(pooled.items()):
        count = len(values)
        num_samples[layer, kv_head] = count
        for index, (query, q_head, bucket, step) in enumerate(values):
            queries_out[layer, kv_head, index] = query
            q_heads_out[layer, kv_head, index] = q_head
            buckets_out[layer, kv_head, index] = bucket
            steps_out[layer, kv_head, index] = step

    meta = {
        **global_meta,
        **extra_meta,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "num_query_heads": query_heads,
        "head_dim": head_dim,
        "max_per_group": max_per_group,
        "seed": seed,
        "collection_buckets": sorted({bucket for bucket in buckets_out.reshape(-1).tolist() if bucket}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "queries": queries_out,
            "num_samples": num_samples,
            "q_head_ids": q_heads_out,
            "context_buckets": buckets_out,
            "decode_steps": steps_out,
            "meta": meta,
        },
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--max-per-group", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id")
    parser.add_argument("--model-fingerprint")
    parser.add_argument("--tp-world-size", type=int)
    parser.add_argument("--tp-rank", type=int)
    parser.add_argument("--rope-config-json", type=Path)
    args = parser.parse_args()
    extra_meta: dict[str, Any] = {}
    for name in ("model_id", "model_fingerprint", "tp_world_size", "tp_rank"):
        value = getattr(args, name)
        if value is not None:
            extra_meta[name] = value
    if args.rope_config_json is not None:
        extra_meta["rope_config"] = json.loads(args.rope_config_json.read_text())
    collect(
        args.input,
        output=args.output,
        num_kv_heads=args.num_kv_heads,
        max_per_group=args.max_per_group,
        seed=args.seed,
        extra_meta=extra_meta,
    )


if __name__ == "__main__":
    main()
