# Query-Robust Affine Chunk Routing: Implementation Note

## Current status

Query-Robust is implemented and tested as an experimental sparse-routing method. It is a quality-preserving prototype, not a production latency claim.

The real-model calibration uses `NousResearch/Meta-Llama-3.1-8B-Instruct`, revision `d10aef7999a2b5ba950ab3974312feeedbfe0b77`, TP=1, BF16, Transformers 4.57.1, and one RTX 3090. Calibration is now explicitly separated from evaluation: it uses The Pile, not RULER.

The online design retains the original robust formulation in the smallest practical representation:

```text
one minimax/CVaR-fitted tangent vector per completed 16-token key chunk
+ one entropy scalar per chunk
```

It never scans, uploads, or retains query samples during decode. The full empirical pool is offline-only but is streamed during prompt/chunk summary construction. Minimax is the default; CVaR is selectable only with the full empirical caps and tail bound. Mean-query/expectation routing is not a Query-Robust runtime mode. The request keeps the frozen pool in BF16 (about 0.8 GB for all 32 layers); only the current layer's batch is cast to FP32 for solving.

## Delivered implementation

| Area | Delivered behavior |
|---|---|
| Artifact | Versioned atomic shards, safe `weights_only` loading, and fail-closed model/revision/head/RoPE/scale validation. |
| Representation | Pile Q is captured after Q normalization and RoPE, before scale; source positions are recorded and vectors are transported to the target request position before fitting. |
| Summaries | Minimax/CVaR active-set tangent fit per complete 16-token chunk, then one BF16 vector and FP32 entropy scalar; full-pool gaps are rescanned. |
| Routing | Deterministic GQA mass aggregation; sink/recent/current chunks mandatory; exact budget and partial-current-chunk accounting. |
| Cache | Native vLLM dense prefill; direct canonical paged-cache FlashAttention for default shared-KV selection. |
| Safety | Artifact, horizon, geometry, RoPE, scale, and sliding-window mismatches fail closed. |

Core implementation: `sparse_frontier/modelling/attention/query_robust.py`, `query_robust_solver.py`, `pile_query_capture.py`, `pile_robust_pilot.py`, `query_pool.py`, `query_capture.py`, `query_robust_diagnostics.py`, `query_pool_cli.py`, attention registry/handler/vLLM patch, and the corresponding tests.

## Empirical-query construction and robust objective

The authoritative pool was generated with:

```text
dataset: monology/pile-uncopyrighted, train, Hub revision 3be90335b66f24456a5d6659d9c8d208c0357119
sequences: 20 random source windows, exactly 2,048 tokens each
seed: 43
pool: 3,000 actual queries per layer and Q head, sampled uniformly without replacement
representation: post-QK-normalization, post-RoPE, pre-scale
```

Each layer shard stores the BF16 query vectors, Q-head and KV-head IDs, absolute token positions, sequence indices, and normalized per-head weights. The manifest records source row hashes, tokenizer/model revisions, dataset revision, seed, and sequence construction. Priority reservoirs avoid materializing all 40,960×32×32 query vectors and do not replace them with an SVD direction. The generated artifact is approximately 836 MB on disk; this is offline state and must be included in memory accounting if loaded into a worker.

For every target chunk, the solver forms `S[q, token]` against the streamed empirical pool and solves either
`min_p max_q g_q(p)` (default) or weighted empirical CVaR. The active set starts at 32–64 queries for minimax and is expanded from the full-pool scan by up to four largest inactive violations per round. This is an active-set scheduling optimization only: each chunk still has its own support budget, every candidate is still scored on the complete pool, and the returned `U-L` is unchanged as the acceptance certificate. CVaR preserves the original caps `w/(1-alpha)`; for alpha=0.95 and 12,000 queries per GQA group, at least 600 supported queries are required for dual feasibility. The final fit reports dual lower bound, full-pool upper bound, `U-L`, active support, and convergence state. Summary quantization is rescanned separately.

The online default is fail-closed: if the configured support/iteration budget does not produce a converged certificate at the requested gap, the intact cache is sent through dense FlashAttention for that request and telemetry records the fallback. This keeps an inexpensive exploratory setting from silently becoming a heuristic sparse route. Set a support/gap pair only after measuring its fallback rate.

The bounded held-out Pile pilot (`pile_robust_pilot`) used layer 31, four 16-token chunks, one disjoint sequence (seed 1043), minimax, initial support 32, maximum support 64, and tolerance 0.01. The exact per-chunk-support implementation reduced absolute log-mass error for all eight KV heads relative to the uniform token baseline. In a matched schedule comparison, admitting four violators per scan reduced solver time from 4.80 s to 3.05 s (1.58x); mean held-out absolute error was 0.953 versus 0.964 for one-at-a-time admission (uniform: 2.109), and worst `U-L` was 0.451 versus 0.723. Top-k recall was saturated at 1.0 because only four chunks were evaluated, so this is a solver/ranking signal rather than a quality claim.

A one-chunk CVaR-0.95 sanity run on the same held-out layer used the required 600-query feasibility floor (`max_support=600`). It reduced held-out absolute log-mass error for all eight KV heads as well, but with only four dual iterations its certificates remained loose (maximum `U-L` 33.8). CVaR is therefore exposed as an offline/solver comparison, not an online default.

| State | Size |
|---|---:|
| Offline Pile pool (32 layers × 32 Q heads × 3,000 × 128 BF16) | ≈836 MB |
| Active request query tensor (all layers, BF16; 32 × 8 KV groups × 12,000 queries × 128) | ≈0.8 GB |
| Current-layer FP32 solver working set | ≈49 MB plus score/support workspace |
| Per-chunk summary state (one vector + scalar) | unchanged one-vector-plus-bias budget |

The artifact validates against the pinned checkpoint, exact Llama-3 RoPE parameters, and scale `1/sqrt(128)`.

## RULER status

The previous centroid/RULER confirmation is retained only as historical integration evidence and is not evidence for the minimax/CVaR objective. A full robust RULER run still requires a long prompt-time solve and has not been completed in this checkpoint; no RULER accuracy claim is made for the robust mode.

| Method | Scores: single, multikey, multiquery, VT, FWE | Text vs dense | Mean wall time |
|---|---|---|---:|
| Dense | 1.0, 1.0, 1.0, 0.2, 1.0 | reference | 2.40 s |
| Quest | 1.0, 1.0, 1.0, 0.2, 1.0 | identical | 3.50 s |
| Query-Robust (centroid prototype; historical) | 1.0, 1.0, 1.0, 0.2, 1.0 | identical on all five | 4.18 s |

The robust solver pilot is the current quality signal. A real vLLM TP=1 smoke on the pinned checkpoint completed a 34-token prompt and one generated token through the schema-2 minimax path. With the deliberately bounded `initial_support=32`, `max_support=64`, and 16 solver iterations, the matched one-at-a-time schedule built 64 summaries in 39.27 s (39.35 s end-to-end). The four-violator schedule built the same 64 summaries in 7.54 s (7.61 s end-to-end), a 5.2x prompt-side speedup. Both runs had unresolved full-pool certificates and therefore set the fail-closed dense flag; this is deliberately not presented as sparse decode speed. Larger support/tighter gaps are an explicit cost-quality trade-off and must be benchmarked before enabling sparse decode.

`query_robust_summary_build_ms` is queued wall time around state construction and includes CUDA-stream serialization with model work. It is a regression counter, not isolated kernel timing. Production timing needs CUDA-event phase instrumentation.

## Runtime contract

For chunk keys `K_c` and empirical query rows `q_i`:

```text
S_i = q_i K_c^T / sqrt(d)
p_c = fit_p(S, logsumexp(S, -1), objective=minimax|cvar)
summary_key_c = p_c K_c
summary_bias_c = H(p_c)
score(q, c) = q summary_key_c / sqrt(d) + summary_bias_c
```

Shared-KV mode selects one deterministic common chunk set from local KV-group scores and calls FlashAttention directly on selected physical pages. Dense prefill remains native vLLM; summary construction is a side effect. Exact attention means exact computation over selected tokens, not equivalence to dense attention.

Direct shared-KV decode requires:

```text
physical_block_size == chunk_size
token_budget % chunk_size == 0
token_budget / chunk_size >= sink_chunks + recent_chunks + 1
num_q_heads % num_kv_heads == 0
num_kv_heads % tp_size == 0
```

One vLLM sequence and TP=1 are the only experimentally validated real-model configuration. TP shard loading is tested, but multi-rank capture, finalization, and throughput are open.

## Reproduction

```bash
PYTHONPATH=. HF_HOME=/path/to/hf python -m sparse_frontier.pile_query_capture \
  --model_path /absolute/path/to/llama \
  --output_dir /tmp/pile-query-pool \
  --num_sequences 20 --sequence_tokens 2048 \
  --samples_per_head 3000 --seed 43 --device cuda

PYTHONPATH=. python -m sparse_frontier.query_pool_cli validate \
  --pool /tmp/pile-query-pool --model_path /absolute/path/to/llama

PYTHONPATH=. HF_HOME=/path/to/hf python -m sparse_frontier.pile_robust_pilot \
  --model_path /absolute/path/to/llama \
  --pool_path /tmp/pile-query-pool \
  --output /tmp/pile-minimax-pilot.json \
  --layers 31 --num_chunks 4 --objective minimax \
  --initial_support 32 --max_support 64 --tolerance 0.01 --max_iterations 32

# Locked RULER evaluation is separate from the Pile calibration above.
VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=. python -m sparse_frontier.ruler_runner \
  --data_path /absolute/path/to/ruler_locked.jsonl \
  --model_path /absolute/path/to/llama \
  --output_dir /tmp/eval --method query_robust --budget 1024 \
  --max_input_tokens 8192 --max_output_tokens 64 --task_indices 5 \
  --query_pool_path /tmp/pile-query-pool \
  --query_robust_generation_horizon 64 \
  --query_robust_objective minimax --query_robust_initial_support 64 \
  --query_robust_max_support 1024
```

## Before a stronger claim

1. Run full robust RULER evaluation with calibration/tuning kept disjoint from locked task rows.
2. Compare minimax and CVaR at fixed full-pool objective gaps, including the 600-point CVaR feasibility floor at alpha=0.95.
3. Add 16K/32K, multi-request, and TP>1 confirmation where hardware permits.
4. Use CUDA-event phase timing and fuse/profile the batched solver and cache accesses.
5. Do not reintroduce mean-query, expectation, heuristic weighting, or an SVD replacement as the robust method.
