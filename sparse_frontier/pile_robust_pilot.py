"""Measure minimax/CVaR routing on held-out The Pile activations.

This is intentionally a bounded pilot: it evaluates selected layers and a
configurable number of chunks so the expensive empirical robust solve is
observable on a single GPU before attempting a full prompt sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from .pile_query_capture import load_pile_query_pool, select_pile_sequences
from .modelling.attention.query_robust import (
    _transport_llama3_rope,
    select_balanced_pile_queries,
)
from .modelling.attention.query_robust_solver import batched_independent_active_fit


@torch.no_grad()
def run_pilot(
    model_path: str | Path,
    pool_path: str | Path,
    *,
    seed: int = 1043,
    layer_ids: Sequence[int] = (31,),
    num_chunks: int = 8,
    chunk_size: int = 16,
    objective: str = "minimax",
    alpha: float = 0.95,
    initial_support: int = 32,
    max_support: int = 128,
    tolerance: float = 1e-3,
    max_iterations: int = 64,
    violators_per_round: int = 1,
    empirical_query_budget: int | None = None,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    pool = load_pile_query_pool(pool_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    sequences, source = select_pile_sequences(
        tokenizer, num_sequences=1, sequence_tokens=2048, seed=seed
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
    )
    model.eval()
    config = model.config
    q_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // q_heads))
    device = next(model.parameters()).device
    captured: dict[int, dict[str, torch.Tensor]] = {}
    selected = set(int(x) for x in layer_ids)
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in selected:
            continue
        attention = layer.self_attn

        def hook(module, args, kwargs, *, _layer=layer_idx):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            cos, sin = kwargs["position_embeddings"]
            position_ids = kwargs.get("position_ids")
            q = module.q_proj(hidden).view(1, hidden.shape[1], q_heads, head_dim).transpose(1, 2)
            k = module.k_proj(hidden).view(1, hidden.shape[1], kv_heads, head_dim).transpose(1, 2)
            if getattr(module, "q_norm", None) is not None:
                q = module.q_norm(q)
            if getattr(module, "k_norm", None) is not None:
                k = module.k_norm(k)
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            k, _ = apply_rotary_pos_emb(k, k, cos, sin)
            positions = (
                position_ids.reshape(-1).to(torch.int32)
                if position_ids is not None
                else torch.arange(hidden.shape[1], device=hidden.device, dtype=torch.int32)
            )
            captured[_layer] = {
                "queries": q[0].detach().cpu().to(torch.bfloat16),
                "keys": k[0].transpose(0, 1).detach().cpu().to(torch.bfloat16),
                "positions": positions.detach().cpu(),
            }

        hooks.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        input_ids = torch.tensor([sequences[0].token_ids], dtype=torch.long, device=device)
        model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    parameters = dict(pool.manifest["rope_parameters"])
    group = q_heads // kv_heads
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for layer_idx in sorted(selected):
        activation = captured[layer_idx]
        keys = activation["keys"][: num_chunks * chunk_size].view(
            num_chunks, chunk_size, kv_heads, head_dim
        )
        # Evaluate one actual held-out query position (the final prompt token)
        # against keys, while fitting summaries at that same target position.
        target_position = int(activation["positions"][min(num_chunks * chunk_size - 1, 2047)].item())
        source_layer = pool.layers[layer_idx]
        source_queries, source_positions, source_weights = select_balanced_pile_queries(
            source_layer.queries_by_kv_head,
            source_layer.positions_by_kv_head,
            source_layer.query_head_ids_by_kv_head,
            source_layer.weights_by_kv_head,
            empirical_query_budget,
        )
        transported = _transport_llama3_rope(
            source_queries.to(device=device, dtype=torch.float32),
            source_positions.to(device=device),
            target_position,
            parameters,
        )
        for kv in range(kv_heads):
            q_pool = transported[kv]
            weights = source_weights[kv].to(device=device, dtype=torch.float32)
            key = keys[:, :, kv].to(device=device, dtype=torch.float32)
            logits = torch.einsum("md,cnd->cmn", q_pool, key) * (head_dim ** -0.5)
            f = torch.logsumexp(logits, dim=-1)
            p, lower, upper, gap, active_size, converged, iterations = batched_independent_active_fit(
                logits,
                f,
                objective=objective,
                weights=weights,
                alpha=alpha,
                initial_support=initial_support,
                max_support=max_support,
                tolerance=tolerance,
                max_iterations=max_iterations,
                active_additions_per_round=violators_per_round,
            )
            summary = torch.einsum("cn,cnd->cd", p, key)
            bias = -(p * p.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)
            q_head = kv * group
            heldout_q = activation["queries"][q_head, min(num_chunks * chunk_size - 1, 2047)].to(device=device, dtype=torch.float32)
            exact = torch.logsumexp(torch.einsum("d,cnd->cn", heldout_q, key) * (head_dim ** -0.5), dim=-1)
            predicted = torch.einsum("d,cd->c", heldout_q, summary) * (head_dim ** -0.5) + bias
            uniform = torch.full_like(key[:, :, 0], 1.0 / chunk_size)
            uniform_summary = torch.einsum("cn,cnd->cd", uniform, key)
            uniform_predicted = torch.einsum("d,cd->c", heldout_q, uniform_summary) * (head_dim ** -0.5) + torch.log(torch.tensor(float(chunk_size), device=device))
            k = min(4, num_chunks)
            exact_top = torch.topk(exact, k=k).indices
            rows.append(
                {
                    "layer": layer_idx,
                    "kv_head": kv,
                    "q_head": q_head,
                    "objective": objective,
                    "num_chunks": num_chunks,
                    "active_size_max": int(active_size.max().item()),
                    "converged_fraction": float(converged.float().mean().item()),
                    "iterations_max": int(iterations.max().item()),
                    "gap_max": float(gap.max().item()),
                    "gap_mean": float(gap.mean().item()),
                    "robust_topk_recall": float(torch.isin(torch.topk(predicted, k=k).indices, exact_top).float().mean().item()),
                    "uniform_topk_recall": float(torch.isin(torch.topk(uniform_predicted, k=k).indices, exact_top).float().mean().item()),
                    "robust_abs_log_mass_error": float((predicted - exact).abs().mean().item()),
                    "uniform_abs_log_mass_error": float((uniform_predicted - exact).abs().mean().item()),
                }
            )
    elapsed = time.perf_counter() - started
    return {
        "status": "ok",
        "model_path": str(Path(model_path).resolve()),
        "pool_path": str(Path(pool_path).resolve()),
        "heldout_source": source,
        "objective": objective,
        "alpha": alpha,
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "violators_per_round": violators_per_round,
        "empirical_query_budget": empirical_query_budget,
        "preprocessing_solver_seconds": elapsed,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--pool_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1043)
    parser.add_argument("--layers", type=int, nargs="+", default=[31])
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--objective", choices=("minimax", "cvar"), default="minimax")
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--initial_support", type=int, default=32)
    parser.add_argument("--max_support", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max_iterations", type=int, default=64)
    parser.add_argument("--violators_per_round", type=int, default=1)
    parser.add_argument("--empirical_query_budget", type=int)
    args = parser.parse_args(argv)
    result = run_pilot(
        args.model_path,
        args.pool_path,
        seed=args.seed,
        layer_ids=args.layers,
        num_chunks=args.num_chunks,
        chunk_size=args.chunk_size,
        objective=args.objective,
        alpha=args.alpha,
        initial_support=args.initial_support,
        max_support=args.max_support,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        violators_per_round=args.violators_per_round,
        empirical_query_budget=args.empirical_query_budget,
    )
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
