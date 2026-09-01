# Query-Robust Affine Chunk Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ml:subagent-driven-development (recommended) or superpowers-ml:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, integrate, and experimentally validate the strongest online-feasible Query-Robust Affine Chunk Routing implementation for the pinned Llama 3.1 8B checkpoint.

**Architecture:** A versioned artifact stores exact prompt-origin-aligned empirical centroids and optional small real-query residual coresets. Dense prefill builds one affine key vector and scalar per complete chunk; decode scores those summaries, selects exact tokens per GQA group, and invokes the existing FlashAttention cache kernel. Dense capture and an offline evaluator establish an empirical signal before the production router is trusted.

**Tech Stack:** Python 3.12, PyTorch 2.8/CUDA 12.8, vLLM 0.11.0, BF16 Llama 3.1 8B, safetensors, pytest, and RULER JSONL.

---

## Locked card and feedback ladder

The protocol is fixed by [the approved design](../specs/2026-09-01-llama-query-pool-online-design.md). The checkpoint is `NousResearch/Meta-Llama-3.1-8B-Instruct` revision `d10aef7999a2b5ba950ab3974312feeedbfe0b77`, already complete at:

```text
/workspace/.hf_home/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77
```

The primary metric is equally task-weighted retained exact attention mass at chunk size 16 and an exact 1,024-token budget. Exploratory success requires positive aggregate gain over uniform for at least two of seeds 41/43/47 and improvement on at least three of five task families for the median seed. There is no minimum effect size; representation correctness and split integrity are hard gates.

| Rung | Gate | Proof artifact |
|---|---|---|
| R0 | Protocol/revision fixed | approved design and `run_card.json` |
| R1 | Imports, schemas, math, config | `pytest-r1.log` |
| R2 | Shape/dtype/device/determinism/finite GPU checks | `tensor-witness.json` |
| R3 | Synthetic known-chunk and exact-selected oracle | `pytest-r3.log` |
| R4 | One real dense capture request | raw shard and `capture-smoke.log` |
| R5 | Five-task 8K representative-layer signal | `pilot_8k/aggregate.json` |
| R6 | Frozen all-layer validation and online choice | seed aggregates and `online-choice.json` |
| R7 | Audited decision and evidence-backed note | `decision.md` and `ImplementationNote.md` |

No higher rung runs before the preceding proof exists. R3 is a synthetic routing oracle rather than a training overfit because this is an inference-only method.

## File map

- `query_pool.py`: fail-closed artifact schema, loading, weighting, and finalization.
- `query_capture.py`: paired pre/post-RoPE decode queries plus post-RoPE key traces needed by exact offline routing.
- `query_robust.py`: summary math, state lifecycle, GQA routing, selected attention.
- `query_pool_cli.py`: finalize/inspect/validate commands.
- `query_robust_diagnostics.py`: streamed exact offline metrics.
- Existing registry, handler, vLLM model, runner, evaluation, and telemetry files: narrow integrations only.
- Four new unit-test modules plus diagnostics tests: all production behavior starts RED.
- `.aris/compute/local.md`: reproducible local environment ledger.
- `experiments/query_robust/`: untracked large traces and tracked-safe JSON/log evidence.

### Task 1: Pin and witness the local environment

**Files:**
- Create: `.aris/compute/local.md`
- Create: `experiments/query_robust/run_card.json`
- Create: `experiments/query_robust/env/tensor-witness.json`

- [ ] Record Python 3.12, `torch==2.8.0`, `vllm==0.11.0`, `transformers==4.57.1`, CUDA 12.8, RTX 3090, requirements hash, checkpoint path/revision, and exact commands. State that `/workspace` is not persistent.
- [ ] Preserve the baseline with `/venv/main/bin/python -m pytest -q`; expected `10 passed, 3 skipped`.
- [ ] Replace the incompatible preinstalled Torch stack, then install in phases:

```bash
/venv/main/bin/python -m pip uninstall -y torch torchvision torchaudio torchcodec
uv pip install --python /venv/main/bin/python --no-cache torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /venv/main/bin/python --no-cache -r requirements.txt
uv pip install --python /venv/main/bin/python --no-cache -e .
```

- [ ] If space is needed, remove only `/root/.cache/pip`; never remove the pinned model blobs.
- [ ] Write `tensor-witness.json` containing package versions, GPU, deterministic CUDA matmul sum `3680.0`, finite BF16 softmax, and requirements SHA256; rerun documented commands verbatim.
- [ ] Commit only `.aris/compute/local.md` with `docs: pin query robust experiment environment`.

### Task 2: Add analytic affine-summary math with TDD

**Files:**
- Create: `tests/test_query_robust_math.py`
- Create: `sparse_frontier/modelling/attention/query_robust.py`

- [ ] Write failing tests for the reverse-KL tangent identity, lower bound, weighted centroid, raw entropy, fitted residual, FP64 reference, BF16 rescan, empty/nonfinite inputs, and invalid weights.
- [ ] Run `/venv/main/bin/python -m pytest tests/test_query_robust_math.py -q`; expected import failure.
- [ ] Implement immutable `SummaryAudit(num_samples, weighted_signed_mean, weighted_abs_mean, max_abs)` and `SummaryBuildResult(summary_key, summary_bias, entropy, quantized_audit)` dataclasses. Define `weighted_centroid(queries: Tensor, weights: Tensor) -> Tensor`, `tangent_gap(logits: Tensor, p: Tensor) -> Tensor`, and `build_mean_summaries(keys: Tensor, queries: Tensor, weights: Tensor, storage_dtype: torch.dtype, bias_mode: Literal["raw_entropy", "mean_residual"], query_batch_size: int = 128) -> SummaryBuildResult`.

- [ ] Normalize nonnegative weights explicitly, accumulate in FP32/FP64, stream residual query batches, cast the vector once, and rescan the cast vector before returning audit values.
- [ ] Run the new tests and then the full CPU suite; commit `feat: add analytic query robust summaries`.

### Task 3: Add the query-pool artifact and finalizer with TDD

**Files:**
- Create: `tests/test_query_pool.py`
- Create: `sparse_frontier/modelling/attention/query_pool.py`
- Create: `sparse_frontier/query_pool_cli.py`

- [ ] Write failing tests for manifest round-trip, equal Q-head weight per KV group, deterministic seeds, cumulative horizons, response-space coreset tensors, atomic writes, `weights_only=True`, TP slicing, revision/scale/RoPE mismatch, corrupt/missing shards, and task-index leakage.
- [ ] Run the module and observe import failure.
- [ ] Implement immutable `QueryPoolManifest`, `QueryPoolLayer`, `QueryPoolExpectations`, and `QueryPool`. Define `load_query_pool(path: Path, expected: QueryPoolExpectations, tp_rank: int) -> QueryPool`, `finalize_capture(input_dir: Path, output_dir: Path, pool_size: int, coreset_sizes: Sequence[int], seed: int) -> Path`, and `validate_no_split_overlap(records: Sequence[CaptureRecord]) -> None`.

- [ ] Each layer shard stores `centroid_by_horizon [local_kv_heads,H,d]` FP32 and optional `coreset_queries [local_kv_heads,R,d]` BF16, normalized weights, source Q-head IDs, offsets, prompt lengths, sequence hashes, and strata. Select actual-query coresets per layer/KV group by deterministic greedy kernel herding on calibration-chunk residual vectors, then refit nonnegative normalized weights with SciPy NNLS. Apply marginal head/task/length/offset quotas when the requested size permits. Write a temporary file and publish with `os.replace`.
- [ ] Add `capture-finalize`, `inspect`, and `validate` commands with JSON success output and nonzero invalid exit; run tests and commit `feat: add versioned empirical query pools`.

### Task 4: Add representation-verified dense query capture with TDD

**Files:**
- Create: `tests/test_query_capture.py`
- Create: `sparse_frontier/modelling/attention/query_capture.py`
- Modify: `sparse_frontier/modelling/models/vllm_model.py`
- Modify: `sparse_frontier/ruler_runner.py`

- [ ] Write fake in-place RoPE/attention tests proving pre-RoPE cloning, post-RoPE/pre-scale capture, prompt-origin `R(offset)q_pre`, `R(P+h)=R(P)R(h)` reconstruction, BF16 prompt/generated post-RoPE key capture, atomic last-layer step shards, and bitwise-identical dense output. Add failures for missing context, multi-sequence batches, head mismatch, and composition error.
- [ ] Run RED before adding the collector.
- [ ] Implement `QueryCaptureCollector.capture_pre_rope(query: Tensor, positions: Tensor, rope_forward: Callable) -> None`, `capture_attention(query: Tensor, key: Tensor, value: Tensor | None, is_prefill: bool, layer_idx: int) -> None`, and `reset_request(context: CaptureContext) -> None`.

- [ ] Read request context from the atomically replaced path in `SF_QUERY_CAPTURE_CONTEXT`; write prompt keys once per requested layer and buffer one decode step across requested layers in `request_<hash>/step_<offset>_rank_<rank>.pt`. Values are omitted by default and captured only for a configured diagnostic subset because retained mass and log-mass error require keys, not values.
- [ ] Enable the rotary hook when `SF_QUERY_CAPTURE_DIR` is set. In capture-only dense mode call the original FlashAttention forward exactly once and never route output through `AttentionHandler`.
- [ ] Add `--capture_query_dir`, `--capture_context_path`, and `--capture_layers`; keep normal dense numerics. Run collector/runner tests and commit `feat: capture position-aligned dense decode queries`.

### Task 5: Add request summaries and deterministic GQA routing with TDD

**Files:**
- Create: `tests/test_query_robust_attention.py`
- Modify: `sparse_frontier/modelling/attention/query_robust.py`

- [ ] Write failing tests for constructor validation, allocation, reset leakage, full/partial chunks, boundary finalization, lower-index ties, sink/recent/current uniqueness, mandatory budget accounting, exact partial length, normalized GQA aggregation, dense fallback, and noncontiguous physical blocks.
- [ ] The central `route_chunks` reference check returns no more than `token_budget/chunk_size` blocks and places the current chunk last.
- [ ] Run RED.
- [ ] Implement `QueryRobustAttention` buffers:

```text
summary_key   [layers,max_chunks,local_kv_heads,d] storage dtype
summary_bias  [layers,max_chunks,local_kv_heads]   FP32
summary_ready [layers,max_chunks,local_kv_heads]   BOOL
valid_tokens  [layers,max_chunks]                  INT16/INT32
```

- [ ] `reset()` clears control state only. Dense prefill chooses the artifact horizon, applies captured prompt-origin RoPE once, summarizes complete chunks, and leaves the partial chunk mandatory.
- [ ] Decode finalizes only a newly completed chunk, scores in FP32, normalizes per Q head, sums within each GQA group, selects deterministic top-k excluding mandatory chunks, repeats group tables to Q heads, places current last, and calls `flash_attn_with_kvcache` with true lengths.
- [ ] Run mocked selected-attention tests and all math/pool tests; commit `feat: route exact chunks with empirical affine summaries`.

### Task 6: Integrate registry, handler, configuration, and telemetry

**Files:**
- Modify: `tests/test_query_robust_attention.py`
- Modify: `tests/test_ruler_runner.py`
- Modify: `sparse_frontier/modelling/attention/registry.py`
- Modify: `sparse_frontier/modelling/attention/handler.py`
- Modify: `sparse_frontier/modelling/models/vllm_model.py`
- Modify: `sparse_frontier/utils/sparsity_server.py`
- Create: `sparse_frontier/configs/attention/query_robust.yaml`

- [ ] Write failing registry tests for every divisibility rule, missing/corrupt pool, scale/revision/TP mismatch, sliding window, chunked prefill, and absent RoPE context. Write telemetry reset tests.
- [ ] Run RED.
- [ ] Register `query_robust`; use `chunk_size` as handler block size and pass model layer/head/TP/physical-block dimensions.
- [ ] Reuse the handler's single token counter, generic cache fallback, and request reset. Add measured selected-token access, not a second estimate.
- [ ] Add bounded telemetry: ready, summaries built, build milliseconds, fallback count, selected-token sum, dense-token sum.
- [ ] Add online-first YAML: budget 1024, chunk 16, horizon 128, raw entropy, coreset size 0, one sink/recent chunk, FP32 scores, BF16 summaries, diagnostics off, fail closed.
- [ ] Run the full CPU suite and commit `feat: integrate query robust attention with vLLM`.

### Task 7: Add exact offline diagnostics and runner plumbing

**Files:**
- Create: `tests/test_query_robust_diagnostics.py`
- Create: `sparse_frontier/query_robust_diagnostics.py`
- Modify: `sparse_frontier/ruler_runner.py`
- Modify: `sparse_frontier/evaluation.py`

- [ ] Write failing tensor-oracle tests for exact log mass, retained mass, top-k recall, signed/absolute error, p95, selected-output L2, near-zero dense norm, and equal task weighting.
- [ ] Run RED.
- [ ] Implement `evaluate_trace(trace_dir: Path, pool: QueryPool, token_budget: int, chunk_size: int, modes: Sequence[str], seed: int, output_dir: Path) -> dict[str, object]` and `aggregate_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, object]`.

- [ ] Stream query/layer batches; compare `uniform`, `centroid_entropy`, every available `coreset_r*`, and `full_residual_oracle` with identical mandatory chunks and exact token counts. For all-layer runs, evaluate one sequence immediately after capture, append atomic metric rows, and discard only that generated raw key trace after its aggregate checksum is durable; never discard query-pool shards or raw metrics.
- [ ] Extend runner choices and manifests. Load 32/32/8/128 from model `config.json` instead of hard-coding. Merge diagnostic aggregate paths in evaluation.
- [ ] Run diagnostics/runner/evaluation tests and commit `feat: evaluate query robust routing diagnostics`.

### Task 8: Execute R0-R4 on the pinned checkpoint

**Files:**
- Create: `experiments/query_robust/r0/`
- Create: `experiments/query_robust/capture_smoke/`

- [ ] Validate config/revision: 32 layers, 32 Q heads, 8 KV heads, d=128, BF16, no bias, Llama3 RoPE, scale `1/sqrt(128)`.
- [ ] Save R1 and R3 logs from the query-robust unit modules.
- [ ] Launch one actual 8K task-index-0 dense capture, cap output at four tokens, capture layers 0/7/15/23/31, and save `capture-smoke.log`.
- [ ] Require exit zero, no generation failure, four step shards, all requested layers, finite composition/logit error, and unchanged dense output versus capture-disabled. Diagnose and rerun R4 until green.

### Task 9: Run R5/R6 empirical selection

**Files:**
- Create: `experiments/query_robust/pilot_8k/`
- Create: `experiments/query_robust/query_pools/llama31_ruler_calibration/`

- [ ] Capture task indices 0 and 5 for all five 8K tasks on representative layers; keep split hashes disjoint.
- [ ] Finalize pool size 2048 and coreset sizes 8/16/32/64 with seed 43.
- [ ] Evaluate uniform, centroid+entropy, full oracle, and coresets at budget 1024/chunk 16 under seeds 41/43/47.
- [ ] If centroid+entropy passes the directional rule, freeze it. Otherwise select the smallest coreset whose sign matches the positive oracle and improves three tasks. If centroid and oracle both fail, keep a negative tested prototype and skip expensive expansion.
- [ ] Only after R5 passes, expand calibration indices 0-4, validation 5-9, and all 32 layers without changing metric, budget, chunk, mandatory policy, or decision rule. Use capture-evaluate-discard streaming so the full-layer evidence fits the 32 GiB filesystem; record each removed generated trace and its retained checksum in the run manifest.

### Task 10: Run the production-router GPU smoke and feasible confirmation

**Files:**
- Create: `experiments/query_robust/router_smoke/`
- Create: `experiments/query_robust/confirmation/`

- [ ] Run GPU-marked selected-attention integration tests.
- [ ] Run one 8K task-index-10 QueryRobust request with the frozen artifact/mode and diagnostic stride 16. Require state ready, summaries built, sparse decode seen, zero silent fallback, budget-respecting access, and nonempty diagnostics.
- [ ] Run matched dense/Quest/QueryRobust indices 10-14: diagnostics first, then diagnostics-disabled latency. Report prefill, summary build, first token, median/p95 decode, boundary spikes, and peak memory.
- [ ] Attempt 16K then 32K one-example smoke only after 8K is green. Treat 24 GiB OOM as hardware-limited, not method-negative.

### Task 11: Audit, document, and verify

**Files:**
- Create: `experiments/query_robust/decision.md`
- Modify: `ImplementationNote.md`
- Modify: `README.md` only after a working GPU smoke

- [ ] Write the R7 memo with highest green rung, hashes, metric by task/seed, accepted mode, failures, fallbacks, memory, build/decode latency, and stopped expansions. Label positive/mixed/negative/infrastructure-limited by the locked rule.
- [ ] Revise `ImplementationNote.md` with measured checkpoint/data provenance, prompt-origin RoPE transport, exact centroid state, selected coreset if any, model-derived dimensions, working commands, costs, and remaining cache-copy limitation. Leave CVaR/minimax optional and unvalidated.
- [ ] Use verification-before-completion: run `/venv/main/bin/python -m pytest -q`, `git diff --check`, inspect `git status --short`, and rerun the smallest successful GPU smoke from a fresh directory.
- [ ] Commit only query-robust files and safe small evidence. Never add checkpoint shards, prompt text, raw capture tensors, or credentials.
