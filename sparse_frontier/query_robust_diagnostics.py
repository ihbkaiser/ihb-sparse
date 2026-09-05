"""Exact offline routing diagnostics for Query-Robust affine summaries."""

from __future__ import annotations

import math
import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def balanced_task_head_centroid(
    queries: Tensor,
    task_ids: Sequence[str],
    query_head_ids: Tensor,
) -> Tensor:
    """Average samples within head, then heads within task, then tasks."""
    if queries.ndim != 2 or query_head_ids.ndim != 1:
        raise ValueError("centroid inputs must be a query matrix and head vector")
    if queries.shape[0] != len(task_ids) or queries.shape[0] != query_head_ids.numel():
        raise ValueError("centroid inputs have inconsistent sample counts")
    if queries.shape[0] == 0:
        raise ValueError("cannot construct an empirical centroid without queries")
    work = queries.detach().cpu().to(torch.float64)
    grouped: dict[str, dict[int, list[Tensor]]] = defaultdict(lambda: defaultdict(list))
    for index, task in enumerate(task_ids):
        grouped[str(task)][int(query_head_ids[index].item())].append(work[index])
    task_means = []
    for task in sorted(grouped):
        head_means = [
            torch.stack(grouped[task][head]).mean(dim=0)
            for head in sorted(grouped[task])
        ]
        task_means.append(torch.stack(head_means).mean(dim=0))
    return torch.stack(task_means).mean(dim=0)


def _llama3_inv_freq(
    head_dim: int,
    base: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    inv = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim
        )
    )
    wavelength = 2.0 * math.pi / inv
    low_wavelength = original_max_position_embeddings / low_freq_factor
    high_wavelength = original_max_position_embeddings / high_freq_factor
    smooth = (
        original_max_position_embeddings / wavelength - low_freq_factor
    ) / (high_freq_factor - low_freq_factor)
    return torch.where(
        wavelength < high_wavelength,
        inv,
        torch.where(
            wavelength > low_wavelength,
            inv / factor,
            (1.0 - smooth) * inv / factor + smooth * inv,
        ),
    )


def apply_llama3_rope(
    query: Tensor,
    position: int,
    *,
    base: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> Tensor:
    """Apply the checkpoint's NeoX-style Llama-3 rotary map at one position."""
    if query.shape[-1] % 2:
        raise ValueError("Llama RoPE requires an even head dimension")
    work_dtype = (
        query.dtype if query.dtype in (torch.float32, torch.float64) else torch.float32
    )
    work = query.to(work_dtype)
    inv = _llama3_inv_freq(
        query.shape[-1],
        base,
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_position_embeddings,
        device=query.device,
        dtype=work_dtype,
    )
    angle = inv * int(position)
    cos = angle.cos()
    sin = angle.sin()
    first, second = work.chunk(2, dim=-1)
    rotated = torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    )
    return rotated.to(query.dtype)


def apply_rope(
    query: Tensor,
    position: int,
    *,
    rope_type: str,
    parameters: dict[str, object],
) -> Tensor:
    """Apply a supported NeoX rotary map from versioned checkpoint metadata."""
    if rope_type == "llama3":
        return apply_llama3_rope(
            query,
            position,
            base=float(parameters["rope_theta"]),
            factor=float(parameters["factor"]),
            low_freq_factor=float(parameters["low_freq_factor"]),
            high_freq_factor=float(parameters["high_freq_factor"]),
            original_max_position_embeddings=int(
                parameters["original_max_position_embeddings"]
            ),
        )
    if rope_type != "default":
        raise ValueError(f"unsupported Query-Robust RoPE type {rope_type!r}")
    if query.shape[-1] % 2:
        raise ValueError("NeoX RoPE requires an even head dimension")
    work_dtype = (
        query.dtype if query.dtype in (torch.float32, torch.float64) else torch.float32
    )
    work = query.to(work_dtype)
    inv = 1.0 / (
        float(parameters["rope_theta"])
        ** (
            torch.arange(0, query.shape[-1], 2, device=query.device, dtype=work_dtype)
            / query.shape[-1]
        )
    )
    angle = inv * int(position)
    cos, sin = angle.cos(), angle.sin()
    first, second = work.chunk(2, dim=-1)
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    ).to(query.dtype)


def build_chunk_summaries(
    keys: Tensor,
    empirical_centroid: Tensor,
    *,
    chunk_size: int,
    mode: str,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build one vector/scalar summary per logical chunk and KV head."""
    if keys.ndim != 3:
        raise ValueError("keys must have shape [tokens, kv_heads, head_dim]")
    if empirical_centroid.shape != keys.shape[1:]:
        raise ValueError("empirical centroid must have shape [kv_heads, head_dim]")
    if chunk_size < 1 or keys.shape[0] < 1:
        raise ValueError("chunk size and token count must be positive")
    if mode not in {"uniform", "centroid_entropy"}:
        raise ValueError(f"unsupported diagnostic summary mode {mode!r}")
    chunks = (keys.shape[0] + chunk_size - 1) // chunk_size
    summary_key = torch.empty(
        chunks, keys.shape[1], keys.shape[2], dtype=torch.float32, device=keys.device
    )
    summary_bias = torch.empty(
        chunks, keys.shape[1], dtype=torch.float32, device=keys.device
    )
    valid_tokens = torch.empty(chunks, dtype=torch.long, device=keys.device)
    keys32 = keys.float()
    centroid32 = empirical_centroid.float()
    for chunk in range(chunks):
        start = chunk * chunk_size
        stop = min(start + chunk_size, keys.shape[0])
        chunk_keys = keys32[start:stop].permute(1, 0, 2)
        valid_tokens[chunk] = stop - start
        if mode == "uniform":
            summary_key[chunk] = chunk_keys.mean(dim=1)
            summary_bias[chunk].fill_(math.log(stop - start))
            continue
        logits = torch.einsum("hd,hnd->hn", centroid32, chunk_keys) * scale
        probabilities = torch.softmax(logits, dim=-1)
        summary_key[chunk] = torch.einsum(
            "hn,hnd->hd", probabilities, chunk_keys
        )
        summary_bias[chunk] = -(
            probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        ).sum(dim=-1)
    return summary_key, summary_bias, valid_tokens


def select_gqa_chunks(
    query: Tensor,
    summary_key: Tensor,
    summary_bias: Tensor,
    valid_tokens: Tensor,
    *,
    token_budget: int,
    chunk_size: int,
    scale: float,
    sink_chunks: int = 1,
    recent_chunks: int = 1,
) -> Tensor:
    """Select deterministic shared chunks for each GQA KV-head group."""
    chunks, kv_heads, head_dim = summary_key.shape
    if query.ndim != 2 or query.shape[1] != head_dim:
        raise ValueError("query must have shape [q_heads, head_dim]")
    q_heads = query.shape[0]
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if summary_bias.shape != (chunks, kv_heads) or valid_tokens.shape != (chunks,):
        raise ValueError("summary tensors have inconsistent shapes")
    total_tokens = int(valid_tokens.sum().item())
    if token_budget < 1:
        raise ValueError("token budget must be positive")
    selected = torch.zeros(kv_heads, chunks, dtype=torch.bool, device=query.device)
    if total_tokens <= token_budget:
        selected.fill_(True)
        return selected

    full_chunks = [
        index for index, count in enumerate(valid_tokens.tolist()) if count == chunk_size
    ]
    partial_chunks = [
        index for index, count in enumerate(valid_tokens.tolist()) if count < chunk_size
    ]
    mandatory = set(full_chunks[:sink_chunks])
    mandatory.update(full_chunks[-recent_chunks:] if recent_chunks else [])
    mandatory.update(partial_chunks)
    mandatory_tokens = sum(int(valid_tokens[index].item()) for index in mandatory)
    if mandatory_tokens > token_budget:
        raise ValueError("mandatory chunks exceed the exact token budget")
    routed_count = (token_budget - mandatory_tokens) // chunk_size

    group = q_heads // kv_heads
    query32 = query.float().view(kv_heads, group, head_dim)
    for kv_head in range(kv_heads):
        logits = torch.einsum(
            "gd,cd->gc", query32[kv_head], summary_key[:, kv_head].float()
        ) * scale
        logits = logits + summary_bias[:, kv_head].float().unsqueeze(0)
        group_score = torch.softmax(logits, dim=-1).sum(dim=0)
        candidates = [index for index in full_chunks if index not in mandatory]
        ordered = sorted(
            candidates, key=lambda index: (-float(group_score[index].item()), index)
        )
        chosen = mandatory | set(ordered[:routed_count])
        selected[kv_head, list(sorted(chosen))] = True
    return selected


def retained_attention_mass(
    query: Tensor,
    keys: Tensor,
    selected_chunks: Tensor,
    valid_tokens: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Return exact retained softmax mass for every query head."""
    if keys.ndim != 3 or query.ndim != 2:
        raise ValueError("query/keys have invalid dimensions")
    q_heads, head_dim = query.shape
    kv_heads = keys.shape[1]
    if keys.shape[2] != head_dim or q_heads % kv_heads:
        raise ValueError("query and key head geometry is inconsistent")
    if selected_chunks.shape != (kv_heads, valid_tokens.numel()):
        raise ValueError("selected chunk mask has an invalid shape")
    group = q_heads // kv_heads
    masses = torch.empty(q_heads, dtype=torch.float32, device=query.device)
    for q_head in range(q_heads):
        kv_head = q_head // group
        logits = torch.einsum(
            "d,td->t", query[q_head].float(), keys[:, kv_head].float()
        ) * scale
        probabilities = torch.softmax(logits, dim=0)
        token_mask = torch.cat(
            [
                torch.full(
                    (int(count),),
                    bool(selected_chunks[kv_head, chunk].item()),
                    dtype=torch.bool,
                    device=query.device,
                )
                for chunk, count in enumerate(valid_tokens.tolist())
            ]
        )
        masses[q_head] = probabilities[token_mask].sum()
    return masses


def _chunk_log_masses(logits: Tensor, valid_tokens: Tensor) -> Tensor:
    masses = []
    start = 0
    for count in valid_tokens.tolist():
        stop = start + int(count)
        masses.append(torch.logsumexp(logits[:, start:stop], dim=-1))
        start = stop
    if start != logits.shape[1]:
        raise ValueError("chunk token counts do not cover the logits")
    return torch.stack(masses, dim=-1)


def _select_from_log_mass(
    log_mass: Tensor,
    valid_tokens: Tensor,
    *,
    kv_heads: int,
    token_budget: int,
    chunk_size: int,
    sink_chunks: int,
    recent_chunks: int,
) -> Tensor:
    q_heads, chunks = log_mass.shape
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    selected = torch.zeros(kv_heads, chunks, dtype=torch.bool, device=log_mass.device)
    if int(valid_tokens.sum().item()) <= token_budget:
        selected.fill_(True)
        return selected
    full = [index for index, count in enumerate(valid_tokens.tolist()) if count == chunk_size]
    partial = [index for index, count in enumerate(valid_tokens.tolist()) if count < chunk_size]
    mandatory = set(full[:sink_chunks])
    mandatory.update(full[-recent_chunks:] if recent_chunks else [])
    mandatory.update(partial)
    mandatory_tokens = sum(int(valid_tokens[index].item()) for index in mandatory)
    routed_count = (token_budget - mandatory_tokens) // chunk_size
    if routed_count < 0:
        raise ValueError("mandatory chunks exceed the token budget")
    group = q_heads // kv_heads
    normalized = torch.softmax(log_mass.float(), dim=-1).view(
        kv_heads, group, chunks
    ).sum(dim=1)
    for kv_head in range(kv_heads):
        candidates = [index for index in full if index not in mandatory]
        ordered = sorted(
            candidates,
            key=lambda index: (-float(normalized[kv_head, index].item()), index),
        )
        chosen = mandatory | set(ordered[:routed_count])
        selected[kv_head, list(sorted(chosen))] = True
    return selected


def route_step_metrics(
    query: Tensor,
    keys: Tensor,
    empirical_centroid: Tensor,
    *,
    mode: str,
    token_budget: int,
    chunk_size: int,
    scale: float,
    sink_chunks: int = 1,
    recent_chunks: int = 1,
) -> dict[str, Tensor]:
    """Compute exact per-head router metrics for one decode layer and step."""
    summary_key, summary_bias, valid = build_chunk_summaries(
        keys,
        empirical_centroid,
        chunk_size=chunk_size,
        mode=mode,
        scale=scale,
    )
    q_heads, head_dim = query.shape
    kv_heads = keys.shape[1]
    group = q_heads // kv_heads
    summary_q = summary_key.permute(1, 0, 2).repeat_interleave(group, dim=0)
    bias_q = summary_bias.T.repeat_interleave(group, dim=0)
    predicted_log_mass = (
        torch.einsum("hd,hcd->hc", query.float(), summary_q.float()) * scale
        + bias_q.float()
    )
    expanded_keys = keys.permute(1, 0, 2).repeat_interleave(group, dim=0)
    token_logits = torch.einsum(
        "hd,htd->ht", query.float(), expanded_keys.float()
    ) * scale
    exact_log_mass = _chunk_log_masses(token_logits, valid)
    selected = _select_from_log_mass(
        predicted_log_mass,
        valid,
        kv_heads=kv_heads,
        token_budget=token_budget,
        chunk_size=chunk_size,
        sink_chunks=sink_chunks,
        recent_chunks=recent_chunks,
    )
    oracle = _select_from_log_mass(
        exact_log_mass,
        valid,
        kv_heads=kv_heads,
        token_budget=token_budget,
        chunk_size=chunk_size,
        sink_chunks=sink_chunks,
        recent_chunks=recent_chunks,
    )
    token_selected = torch.cat(
        [
            selected[:, chunk : chunk + 1].expand(-1, int(count))
            for chunk, count in enumerate(valid.tolist())
        ],
        dim=1,
    ).repeat_interleave(group, dim=0)
    retained = (torch.softmax(token_logits, dim=-1) * token_selected).sum(dim=-1)
    error = predicted_log_mass - exact_log_mass
    recall_by_group = (selected & oracle).sum(dim=-1).float() / oracle.sum(
        dim=-1
    ).clamp_min(1)
    return {
        "retained_attention_mass": retained,
        "log_mass_signed_error_mean": error.mean(dim=-1),
        "log_mass_abs_error_mean": error.abs().mean(dim=-1),
        "log_mass_abs_error_p95": torch.quantile(error.abs(), 0.95, dim=-1),
        "topk_recall": recall_by_group.repeat_interleave(group),
        "selected_chunks": selected.sum(dim=-1).repeat_interleave(group),
        "num_chunks": torch.full(
            (q_heads,), valid.numel(), dtype=torch.long, device=query.device
        ),
    }


def _safe_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot load capture shard {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"capture shard is not a dictionary: {path}")
    return payload


def _request_metadata(request_dir: Path) -> dict[str, Any]:
    prompts = sorted(request_dir.glob("prompt_layer_*_rank_000.pt"))
    if not prompts:
        raise ValueError(f"capture request has no prompt shards: {request_dir}")
    metadata = _safe_load(prompts[0]).get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"capture request has invalid metadata: {request_dir}")
    return metadata


def build_capture_centroids(
    capture_dir: str | Path,
    *,
    split: str = "calibration",
    horizon: int | None = None,
) -> dict[int, Tensor]:
    """Build exact balanced prompt-origin centroids from calibration traces."""
    root = Path(capture_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    group = int(manifest["num_q_heads"]) // int(manifest["num_kv_heads"])
    records: dict[tuple[int, int], list[tuple[Tensor, str, int]]] = defaultdict(list)
    found = 0
    for request in sorted(root.glob("request_*")):
        metadata = _request_metadata(request)
        if metadata.get("split") != split:
            continue
        for step_path in sorted(request.glob("step_*_rank_000.pt")):
            step = _safe_load(step_path)
            offset = int(step["metadata"]["decode_offset"])
            if horizon is not None and offset >= horizon:
                continue
            layer_ids = step["layer_ids"].tolist()
            queries = step["prompt_origin_queries"].float()
            for local_layer, layer_idx in enumerate(layer_ids):
                for kv_head in range(int(manifest["num_kv_heads"])):
                    for q_head in range(kv_head * group, (kv_head + 1) * group):
                        records[(int(layer_idx), kv_head)].append(
                            (queries[local_layer, q_head], str(metadata["task"]), q_head)
                        )
            found += 1
    if not found:
        raise ValueError(f"capture contains no {split!r} decode queries")
    centroids: dict[int, list[Tensor]] = defaultdict(list)
    layers = sorted({layer for layer, _ in records})
    for layer in layers:
        for kv_head in range(int(manifest["num_kv_heads"])):
            samples = records[(layer, kv_head)]
            if not samples:
                raise ValueError(f"missing centroid samples for layer {layer}, KV head {kv_head}")
            centroids[layer].append(
                balanced_task_head_centroid(
                    torch.stack([sample[0] for sample in samples]),
                    [sample[1] for sample in samples],
                    torch.tensor([sample[2] for sample in samples]),
                ).float()
            )
    return {layer: torch.stack(heads) for layer, heads in centroids.items()}


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in rows})
    tasks = sorted({str(row["task"]) for row in rows})
    aggregates: dict[str, Any] = {}
    for method in methods:
        task_metrics: dict[str, Any] = {}
        for task in tasks:
            subset = [row for row in rows if row["method"] == method and row["task"] == task]
            task_metrics[task] = {
                metric: sum(float(row[metric]) for row in subset) / len(subset)
                for metric in (
                    "retained_attention_mass",
                    "log_mass_signed_error_mean",
                    "log_mass_abs_error_mean",
                    "log_mass_abs_error_p95",
                    "topk_recall",
                )
            }
        overall = {
            metric: sum(task_metrics[task][metric] for task in tasks) / len(tasks)
            for metric in next(iter(task_metrics.values()))
        }
        aggregates[method] = {"overall": overall, "tasks": task_metrics}
    if {"uniform", "centroid_entropy"}.issubset(aggregates):
        gains = {
            task: aggregates["centroid_entropy"]["tasks"][task][
                "retained_attention_mass"
            ]
            - aggregates["uniform"]["tasks"][task]["retained_attention_mass"]
            for task in tasks
        }
        aggregates["comparison"] = {
            "retained_mass_gain": aggregates["centroid_entropy"]["overall"][
                "retained_attention_mass"
            ]
            - aggregates["uniform"]["overall"]["retained_attention_mass"],
            "task_gains": gains,
            "tasks_improved": sum(gain > 0 for gain in gains.values()),
        }
    return aggregates


def evaluate_capture(
    capture_dir: str | Path,
    output_json: str | Path,
    *,
    rows_jsonl: str | Path | None = None,
    calibration_split: str = "calibration",
    evaluation_split: str = "validation",
    token_budget: int = 1024,
    chunk_size: int = 16,
    horizon: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Evaluate uniform and centroid summaries on disjoint dense traces."""
    root = Path(capture_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    centroids = build_capture_centroids(
        root, split=calibration_split, horizon=horizon
    )
    rope = dict(manifest["rope_parameters"])
    if manifest["rope_type"] not in {"llama3", "default"}:
        raise ValueError(
            "the offline evaluator supports only Llama-3 or default NeoX RoPE"
        )
    target = torch.device(device)
    rows: list[dict[str, Any]] = []
    evaluated_requests = 0
    for request in sorted(root.glob("request_*")):
        metadata = _request_metadata(request)
        if metadata.get("split") != evaluation_split:
            continue
        steps = [_safe_load(path) for path in sorted(request.glob("step_*_rank_000.pt"))]
        if horizon is not None:
            steps = [step for step in steps if int(step["metadata"]["decode_offset"]) < horizon]
        if not steps:
            continue
        layer_ids = [int(layer) for layer in steps[0]["layer_ids"].tolist()]
        generated_by_layer = {
            layer: torch.cat(
                [step["generated_keys"][local_layer] for step in steps], dim=0
            )
            for local_layer, layer in enumerate(layer_ids)
        }
        for local_layer, layer in enumerate(layer_ids):
            prompt_path = request / f"prompt_layer_{layer:03d}_rank_000.pt"
            prompt_keys = _safe_load(prompt_path)["keys"]
            request_centroid = apply_rope(
                centroids[layer].to(target),
                int(metadata["prompt_length"]),
                rope_type=str(manifest["rope_type"]),
                parameters=rope,
            )
            for step_number, step in enumerate(steps):
                keys = torch.cat(
                    [prompt_keys, generated_by_layer[layer][: step_number + 1]], dim=0
                ).to(target)
                query = step["post_rope_queries"][local_layer].to(target)
                for mode in ("uniform", "centroid_entropy"):
                    metrics = route_step_metrics(
                        query,
                        keys,
                        request_centroid,
                        mode=mode,
                        token_budget=token_budget,
                        chunk_size=chunk_size,
                        scale=float(manifest["attention_scale"]),
                    )
                    for q_head in range(query.shape[0]):
                        rows.append(
                            {
                                "sequence_id_hash": int(metadata["sequence_id_hash"]),
                                "task": str(metadata["task"]),
                                "task_index": int(metadata["task_index"]),
                                "split": str(metadata["split"]),
                                "decode_offset": int(step["metadata"]["decode_offset"]),
                                "layer": layer,
                                "q_head": q_head,
                                "kv_head": q_head
                                // (int(manifest["num_q_heads"]) // int(manifest["num_kv_heads"])),
                                "method": mode,
                                **{
                                    name: float(value[q_head].item())
                                    for name, value in metrics.items()
                                },
                            }
                        )
        evaluated_requests += 1
    if evaluated_requests == 0:
        raise ValueError(f"capture contains no {evaluation_split!r} evaluation requests")
    result = {
        "capture_dir": str(root),
        "calibration_split": calibration_split,
        "evaluation_split": evaluation_split,
        "token_budget": token_budget,
        "chunk_size": chunk_size,
        "horizon": horizon,
        "device": str(target),
        "evaluated_requests": evaluated_requests,
        "metric_rows": len(rows),
        "aggregates": _aggregate_rows(rows),
    }
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows_jsonl is not None:
        row_path = Path(rows_jsonl)
        row_path.parent.mkdir(parents=True, exist_ok=True)
        row_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows_jsonl")
    parser.add_argument("--token_budget", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = evaluate_capture(
        args.capture,
        args.output,
        rows_jsonl=args.rows_jsonl,
        token_budget=args.token_budget,
        chunk_size=args.chunk_size,
        horizon=args.horizon,
        device=args.device,
    )
    print(json.dumps(result["aggregates"]["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
