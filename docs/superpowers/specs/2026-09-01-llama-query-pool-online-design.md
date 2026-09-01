# Llama 3.1 Query-Pool Study for Online Affine Chunk Routing

## Status and scope

This design records the approved experiment before checkpoint download or code changes. Its purpose is to obtain a real-model signal and use that evidence to revise `ImplementationNote.md`. It is not yet an implementation plan for the complete `query_robust` attention path and it does not claim that the method works on Llama 3.1.

The target checkpoint is the ungated `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror, pinned to revision:

```text
d10aef7999a2b5ba950ab3974312feeedbfe0b77
```

Before any measurement, the downloaded config and tensor index must be checked against the expected 32 layers, 32 query heads, 8 KV heads, head dimension 128, BF16 weights, and standard scaled dot-product attention. The model revision, tokenizer revision, resolved config, package versions, GPU, and git commit must be written to each run manifest.

The study may use the existing RULER pilot data. It must not turn a weak exploratory signal into a production or accuracy claim.

## Decision: an online-first empirical-query ladder

The production candidate will not keep or scan a large empirical query pool. The study compares four representations at the same one-vector/one-scalar chunk-summary budget and selects the cheapest one that gives a held-out directional routing gain:

1. **Uniform baseline.** Uniform token weights, mean key, and `log(valid_tokens)` bias. This uses no empirical queries.
2. **Centroid + entropy.** Compute the tangent from the exact weighted empirical query centroid and use only its analytic entropy intercept. Online state contains request-adjusted centroids, not sampled queries. This is the preferred online form if it works.
3. **Centroid + coreset residual.** Keep the same exact centroid slope, but fit the per-chunk scalar using a small, nonnegative-weight coreset of real queries. This is used only if it gives a reproducible held-out gain over centroid + entropy.
4. **Centroid + full residual oracle.** Use the full frozen pool to fit the scalar. This is an offline quality reference and must not be proposed as the online default.

CVaR and minimax are out of scope for this first real-model study. They add optimization cost without answering the prior question of whether empirical mean-query information helps Llama routing at all.

This ladder is chosen over two alternatives:

- A full online pool is exact for the chosen empirical objective but has unacceptable prompt-time query-key work and avoidable GPU memory.
- Synthetic sigma points or a low-rank Gaussian approximation are compact, but their nonlinear log-sum-exp accuracy and behavior after position transport are less transparent than a small set of actual queries. They remain later ablations, not the default.

## Empirical query representation

### Capture point

Capture decode queries at the exact score representation used by the backend: after any Q/K normalization and after RoPE, before multiplication by the attention scale. Llama 3.1 is expected to have no Q/K normalization, but this must be established from the resolved model code rather than assumed.

For every captured query, record:

- model revision, layer, global query-head ID, mapped global KV-head ID, and TP rank;
- absolute query position, prompt length, and decode offset;
- task family, context-length regime, and a stable sequence hash;
- the backend attention scale and RoPE configuration.

No prompt or generated text is stored in the query artifact.

### Position transport

Raw post-RoPE queries from one absolute position are not directly reusable at another prompt length. The finalizer therefore inverse-rotates every captured query using its recorded absolute position and the checkpoint's exact Llama RoPE implementation. It then applies RoPE at the relative decode offset and stores this prompt-origin-aligned query plus the offset. For a compositional rotary map this gives `R(P + h)q = R(P)[R(h)q]`, so request setup needs one batched rotation by prompt length rather than a separate absolute-position transformation for every sample.

For a new request of prompt length `P`, the pool loader applies `R(P)` to the stored prompt-origin-aligned centroid or coreset. This reconstructs the required `R(P + decode_offset)` representation while moving offset work out of request setup. The request-adjusted representation is used to compute centroids and, when enabled, residuals. The implementation must stop rather than silently use the pool if any of these checks fails:

- inverse-RoPE then forward-RoPE does not reconstruct the captured query within dtype tolerance;
- `R(P + h)q` and `R(P)[R(h)q]` disagree over the supported position range;
- reconstructed query-key logits disagree with the backend's dense logits;
- RoPE type, parameters, attention scale, head mapping, or model revision differs from the manifest;
- the requested decode horizon is outside the artifact's supported offsets.

The first artifact supports decode offsets 0 through 127 because the current NIAH outputs are capped at 128 tokens. Offset weights in a request are derived from the declared evaluation horizon, not from future generated content. A shorter request renormalizes over its supported offsets.

### Balanced empirical measure

The empirical measure is defined per layer and KV-head group. It gives equal total weight to each of the four mapped query heads, then balances task family, context-length regime, and decode-offset bucket within each query head. Samples missing from one stratum do not transfer uncontrolled weight to a different task or head; the finalizer records the realized weights and coverage.

The exact centroid is computed from all accepted calibration queries in FP64 and is never approximated by the coreset. The finalizer stores FP32 cumulative horizon centroids after relative-offset alignment. Supporting all horizons from 1 through 128 costs about 16 MiB on the host for the full 32-layer model. At request setup, the declared generation horizon selects one `[layers, KV heads, d]` table, applies the prompt-origin rotation in FP32, and uploads about 128 KiB. Only this request-adjusted table is needed for the entropy-only path. Therefore the online tangent remains the analytic optimizer of the declared weighted empirical-mean objective even when residual fitting is compressed or disabled.

## Residual coreset

The coreset exists only to approximate the empirical scalar residual

```text
logsumexp(q K^T / sqrt(d)) - q summary_key / sqrt(d) - entropy
```

for each chunk. It does not determine the tangent slope.

Candidates are real calibration queries in prompt-origin-aligned form. Coreset selection is performed separately for each layer and KV-head group using response features measured on calibration key chunks. The feature for one candidate is its residual over a stratified bank of key chunks. Selection uses deterministic greedy kernel herding followed by nonnegative least squares on the selected support. Weights are nonnegative and normalized; marginal task, head, context-length, and offset-bucket coverage are enforced as far as the requested coreset size permits.

Candidate sizes are 8, 16, 32, and 64 queries per layer/KV-head group. Size is selected only on validation key chunks from disjoint sequence indices. The full-pool residual is the offline oracle. If no coreset size through 64 preserves the direction of the oracle's gain, the online design falls back to centroid + entropy; it does not ship a larger hidden pool.

At 32 layers, 8 KV heads, dimension 128, and BF16 query storage, the query tensors occupy approximately:

| Residual representation | Query tensor memory |
|---|---:|
| Full pool, N=2,048 | 128 MiB |
| Coreset, R=64 | 4 MiB |
| Coreset, R=32 | 2 MiB |
| Coreset, R=16 | 1 MiB |
| Coreset, R=8 | 0.5 MiB |
| Centroid + entropy | no residual-query tensor; about 128 KiB active centroid |

The full cumulative centroid table is about 16 MiB of host artifact state and is not included in the active-centroid row. Metadata and temporary workspace must be measured separately. The report must also measure summary-build latency because memory alone does not determine online suitability.

## Data split and leakage control

The three existing RULER pilot files contain five task families: `niah_single`, `niah_multikey`, `niah_multiquery`, `vt`, and `fwe`, at nominal 8K, 16K, and 32K contexts. The same `task_index` reuses a random seed across length files, so a row-wise random split would leak related examples.

All context lengths are split together by `task_index`:

- calibration: indices 0-4;
- validation/model selection: indices 5-9;
- pilot confirmation: indices 10-14;
- locked follow-up: indices 15-49.

The initial cheap rung uses 8K only, one calibration and one validation sequence per task, layers 0, 7, 15, 23, and 31, and all query/KV heads in those layers. It is intended to validate the capture path and expose gross failure, not support a model-wide claim. Only after this rung passes do we capture all layers and additional split rows. The 16K and 32K regimes are expansion rungs; inability to fit 32K on the 24 GB GPU is reported as a hardware limitation, not a method failure.

No example, task index, or derived key chunk used for coreset construction may appear in validation, pilot, or locked evaluation. Query samples and key chunks inherit the sequence's split.

## Experiment card A: is there a method signal?

**Question.** On held-out Llama 3.1 decode traces, does an empirical-query tangent improve fixed-budget chunk routing over the uniform mean-key baseline?

**Primary comparison.** Uniform baseline versus centroid + entropy, with chunk size 16 and an exact selected-token budget of 1,024. Mandatory sink, recent, and partial chunks are identical and included inside the budget.

**Primary metric.** Mean exact attention mass retained by the selected tokens, aggregated first within query head and task, then equally across the five task families. This metric directly tests router selection and does not require a full sparse decode implementation.

**Directional success rule.** Treat the result as a useful signal, not a final accuracy result, if the empirical method has positive aggregate gain over uniform in at least two of three deterministic resampling seeds and improves at least three of five task families in the median-seed report. There is no minimum effect-size threshold and no requirement that a confidence interval exclude zero.

**Guardrails.** Report, but do not use as hard primary thresholds:

- exact chunk top-k recall;
- mean and p95 absolute chunk log-mass error;
- signed chunk log-mass error;
- relative L2 error of exact-selected attention output, with a stated near-zero norm convention;
- worst-layer and worst-head retained mass;
- peak memory, trace-capture time, and offline scoring time.

A single pooled improvement that is driven by one task, layer, or head is recorded as inconclusive rather than positive.

## Experiment card B: what is acceptable online?

**Question.** What is the cheapest residual mode that preserves the directional method signal?

**Comparisons.** Centroid + entropy, centroid + coreset residual at R in {8, 16, 32, 64}, and centroid + full residual oracle. All variants use the exact same empirical centroid, chunk budget, mandatory set, tie breaking, and selected-attention calculation.

**Primary metric.** The same retained exact attention mass as card A.

**Selection rule.** Choose centroid + entropy if its directional card-A signal survives. Otherwise choose the smallest R whose median empirical-versus-uniform gain has the same positive sign as the full oracle and improves at least three of five task families. This deliberately weak rule is for selecting an implementation candidate, not claiming equivalence to the oracle.

**Online guardrails.** The chosen mode must:

- use no full-pool tensor during request setup or decode;
- report request-adjustment, prompt-summary, and periodic finalization time separately;
- preserve nonnegative normalized residual weights;
- produce finite summaries for every accepted chunk;
- fall back explicitly to dense attention on artifact or representation failure;
- never substitute the full pool after a coreset failure.

If R=64 does not preserve a positive signal, the implementation note will recommend centroid + entropy or stop the online empirical-residual path. It will not conceal the full-pool cost in prefill latency.

## Trace and evaluation pipeline

The first implementation adds a diagnostic trace path rather than the full production router:

1. Load the pinned checkpoint and run dense greedy generation with the existing vLLM patch enabled only for capture side effects.
2. Save rank-local tensor shards containing captured queries and the K/V data or sufficient chunk statistics needed for exact offline evaluation.
3. Verify sampled captured logits and outputs against the dense backend before accepting a shard.
4. Finalize calibration queries deterministically and generate the exact centroid, candidate coresets, weights, coverage report, and manifest.
5. Evaluate uniform, entropy, coreset, and full-oracle variants on disjoint traces using identical mandatory chunks and exact token accounting.
6. Write per-layer/head/task metrics plus a compact aggregate JSON. Preserve raw run manifests and logs.

The trace format must be append-safe and rank-local. A complete shard is published by atomic rename. Partial or corrupt shards are rejected. CUDA synchronization must be confined to explicitly timed capture sections so capture overhead is not mistaken for production latency.

## Staged run order and stop conditions

### R0: representation tests

- Resolve and pin checkpoint/tokenizer revisions.
- Test GQA head mapping.
- Test inverse/forward RoPE round trips and prompt-origin composition across 8K, 16K, and 32K positions.
- Compare sampled reconstructed logits with dense backend logits.

Stop on any mismatch; routing metrics are invalid until representation correctness is established.

### R1: cheapest 8K trace

- Five calibration and five validation sequences, one per task.
- Five representative layers, all heads.
- Uniform versus centroid + entropy only.

Continue if metrics are finite, splits are clean, and there is a direction worth expanding. A negative result here permits one capture/debug audit but not parameter fishing on validation.

### R2: model-wide method signal

- All 32 layers.
- Expand within the fixed calibration and validation index ranges.
- Run deterministic resampling seeds 41, 43, and 47.

Stop the empirical-query implementation if centroid + entropy and the full residual oracle both fail the relaxed card-A directional rule. More complex robust solvers are not justified.

### R3: online residual selection

- Build and evaluate R in {8, 16, 32, 64}.
- Select the cheapest accepted mode using card B.
- Measure its serialized size and estimated prompt-build work.

### R4: confirmation and longer contexts

- Confirm the frozen choice on task indices 10-14 without retuning.
- Attempt 16K, then 32K if memory permits.
- Use indices 15-49 only after the implementation and reporting protocol are frozen.

Downstream RULER task accuracy is exploratory at this stage. It may corroborate the router signal, but failure to move accuracy on a tiny pilot does not erase a clear retained-mass improvement, and a task-accuracy fluctuation does not replace router diagnostics.

## Required revision to `ImplementationNote.md`

After the measured study, revise the implementation note to include:

- the exact pinned checkpoint and empirical-data provenance;
- the position-transported query representation and fail-closed checks;
- the latency-first choice between centroid + entropy and a selected coreset size;
- measured pool/coreset memory and prompt-summary cost;
- the fixed data split and directional evidence rule;
- tables for card A and card B, including negative or mixed results;
- explicit separation of offline oracle results from the proposed online path.

No numeric result is inserted without a corresponding run manifest and raw aggregate. If the study stops at a failed rung, the note must say so and recommend the cheapest sound fallback rather than projecting an unmeasured improvement.
