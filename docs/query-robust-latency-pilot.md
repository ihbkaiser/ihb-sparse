# Query-Robust online-latency pilot

This is a narrow, reproducible signal check, not a benchmark claim.  Every
Query-Robust run below used the pinned NousResearch Meta-Llama-3.1-8B-Instruct
checkpoint, seed 43, 16-token chunks, a 512-token total attention budget, and
the same RULER pilot examples as its comparator.  At 32K, all runs additionally
used three GiB CPU offload on the 24 GiB RTX 3090.

The initial Pile-artifact figures remain useful ablations, but the current
online recommendation uses a versioned **context-matched RULER decode-query
pool**.  The 16K and 32K pools were captured independently with dense
attention on calibration-only examples, then deterministically sampled to 32
queries per Q head (128 per KV group before the online subpool).  A 16K pool
did not transfer cleanly to 32K, so a production artifact must be tied to its
declared context regime rather than treated as universal.

## Online setting selected

Use a deterministic balanced empirical subpool of 64 queries per KV group
(16 retained queries from each of the four mapped Q heads), a 64-query active
support, and one batched minimax dual update.  This keeps the original
empirical minimax affine-summary objective and GQA aggregation; it only caps
the online empirical measure and the solver work.  The selected command-line
settings are:

```text
--query_robust_empirical_query_budget 64
--query_robust_initial_support 64
--query_robust_max_support 64
--query_robust_solver_max_iterations 1
--query_robust_solver_chunk_batch_size 64
--query_robust_solver_violators_per_round 4
```

At 16K, a 256-chunk construction batch is the faster scheduling choice on
this RTX 3090.  At 32K with CPU offload it is a regression, so retain 64
there.  This is a hardware/context scheduling parameter, not a method
parameter: it does not alter the empirical measure, minimax objective, or
selected-token budget.

This is an **exploratory accuracy/latency mode**, not a certified deployment
configuration: the current full-pool certificate is not met with one update,
so it requires `--no-query_robust_solver_fail_closed`.  The default
fail-closed configuration remains the sound setting and falls back to dense
attention whenever a certificate is unavailable.

For this already exploratory, fixed-support, one-update route only, adding
`--no-query_robust_solver_armijo` removes the host synchronization used to
decide a backtracking step.  It is rejected by the code unless support is
fixed, solver audit is disabled, and fail-closed routing is disabled; the
certified configuration always retains Armijo backtracking.

The decode hot path now scores a mandatory partial chunk by its exact
current-query log-mass instead of solving an empirical affine fit for it at
every generated token.  It remains mandatory and exact-selected attention is
unchanged.  This removes both an unnecessary per-token solve and a full
summary-table clone.  Summary construction for newly complete chunks remains
the same empirical minimax computation.

The 16-query (four-per-Q-head) ablation produced the same 32K FWE answer but
had no measurable latency advantage (89.99 s versus 90.00 s), so it is not
selected: it weakens the empirical measure without buying runtime.

## Matched results

At 16K, one fixed index from each of five RULER task families gives the
following sum of their task metrics (maximum 5.0):

| Method | Score | Mean end-to-end latency |
| --- | ---: | ---: |
| Query-Robust, 64 queries, 8 updates | 4.20 | 14.81 s |
| Query-Robust, 64 queries, 1 update | 4.20 | 9.90 s |
| Quest | 3.87 | 5.97 s |
| ShadowKV | 4.20 | 10.20 s |

Thus the one-update setting preserves this pilot's score while reducing
Query-Robust latency by 33%.  It remains slower than Quest on the five-task
16K slice, but is slightly faster than ShadowKV there.

At 32K on the same FWE example, the matched 512-token comparison is:

| Method | FWE metric | End-to-end latency | Outcome |
| --- | ---: | ---: | --- |
| Query-Robust, 64 queries, 1 update | 1.00 | 90.00 s | completed |
| Quest | 0.667 | 94.71 s | completed |
| ShadowKV | — | — | engine failed before decode |

The 32K Query-Robust result is a quality and latency signal against Quest on
that example, not proof of aggregate superiority.  ShadowKV's failed result
is deliberately not counted as zero accuracy.

## Context-matched-pool update

On a held-out 16K FWE slice (three examples), the 16K empirical pool gave the
same accuracy for all methods: Query-Robust 0.889, Quest 0.889, and ShadowKV
0.889.  Mean end-to-end latency was 6.18 s, 5.34 s, and 6.60 s respectively.
This is a quality recovery relative to the generic pool, but not a claim that
Query-Robust is faster than Quest.

On one additional held-out example from each of the five 16K RULER families,
all methods again reached the same aggregate metric, 4.2/5.  Query-Robust's
per-example latency was 5.95, 6.99, 10.90, 7.15, and 7.91 s for FWE,
multi-key NIAH, multi-query NIAH, single-needle NIAH, and VT.  Quest took
4.50, 5.62, 7.12, 6.67, and 5.61 s; ShadowKV took 7.62, 8.14, 11.30, 6.97,
and 9.85 s.  Thus the current route is consistently faster than ShadowKV on
this small matched slice, while Quest remains the 16K latency leader.

The current raw-entropy serving path also skips residual/error rescans and
host-side audit extrema that cannot affect its entropy intercept, summary, or
route.  Enable `collect_solver_audit` only for calibration/diagnostic runs.
On the five-example 16K slice this reduced the mean latency from 7.78 s to
7.69 s at a 64-chunk build batch; with the validated 256-chunk batch the same
score was reached in **7.05 s** (9.4% lower than the prior 64-chunk path).

The prompt-construction contraction now uses an explicit strided batched GEMM
rather than relying on the generic einsum contraction planner.  On the same
16K hard FWE example with the deliberately larger 128-query fixed support,
the pre-change run took 5.489 s end-to-end; the GEMM path took 5.399 s with
the identical answer (0.667).  A further certificate-free, fixed-support
exploratory run took 5.387 s.  The latter only omits a final full-pool
certificate scan after the tangent has already been fit; it is enabled solely
when all of the following hold: fail-closed is disabled, diagnostics are off,
and `max_support == initial_support`.  It cannot alter the fitted `p`, summary,
or selected chunks, but it also cannot be presented as a certificate.  The
default configuration never takes this shortcut.

On one held-out 32K FWE example, the independently captured 32K pool produced
the following matched results:

| Method | Empirical queries | Chunk-build batch | FWE metric | End-to-end latency |
| --- | ---: | ---: | ---: | ---: |
| Query-Robust | 128 | 16 | 0.667 | 103.23 s |
| Query-Robust, bounded online setting | 64 | 16 | 0.667 | 99.22 s |
| Query-Robust, bounded + batched | 64 | 64 | 0.667 | 94.64 s |
| Quest | — | — | 0.667 | 93.25 s |

The larger independent-fit batch preserved the answer and brought the current
implementation within 1.5% of Quest on this sample.  It is a batch-scheduling
change only; it does not alter the empirical objective, pool, or selection
rule.  This is the selected pilot setting, while the default certified
configuration remains conservative.

The full held-out 32K FWE block (five examples, indices 10--14) gives
Query-Robust and Quest the same aggregate accuracy, **0.867**.  Their mean
end-to-end latencies are 93.10 s and 92.92 s respectively.  The three fixed
subsets were 0.667 (one example), 0.833 (two examples), and 1.000 (two
examples) for both methods.  This rules out the earlier one-example quality
win as a robust aggregate claim, while showing that the context-matched pool
attains quality parity at this budget.

ShadowKV was also retried on the final two examples with the same three-GiB
CPU offload.  Its engine died before producing any decode token for either
request; these failures are reported separately and are **not** interpreted
as zero-quality observations.  Query-Robust therefore has an observed
long-context feasibility advantage over this ShadowKV configuration, but the
pilot does not establish a numeric accuracy comparison to it at 32K.

## Fair-access refresh

A subsequent fixed, held-out 16K FWE refresh used all indices 10--14 and
matched the code-level base token access rather than treating the three
methods' CLI budget identically.  Query-Robust used an exact 512-token total
budget; Quest's 512 setting includes its extra current 16-token page (528
tokens); ShadowKV used a 96-token sparse budget in addition to its fixed
416-token local/outlier reserve.  The results were:

| Method | FWE accuracy | Mean end-to-end latency | Outcome |
| --- | ---: | ---: | --- |
| Query-Robust | 0.867 | 5.55 s | completed, no fallback |
| Quest | 0.867 | 5.06 s | completed |
| ShadowKV (96 + fixed reserve) | 0.867 | 6.13 s | completed |

Thus this fairer five-example slice establishes quality parity and a latency
advantage over ShadowKV at approximately equal base access, but **not** a
quality win over either baseline.  At 32K, Query-Robust and Quest remain tied
at 0.867 on the corresponding five-example FWE block.  ShadowKV was retried
there at the comparable 96-token sparse setting with both six and eight GiB
of CPU offload; the engine terminated before its first decode token each time.
That is a real feasibility observation on this 24 GiB RTX 3090, not a zero
accuracy result.

The same fair-access accounting was also applied to two five-family 16K
single-index slices.  At index 10, all three methods scored 4.2/5.  At index
1, Query-Robust scored 4.2/5, Quest 3.867/5, and ShadowKV 4.2/5.  This is a
repeatable positive signal against Quest, but it is still only parity with
ShadowKV; it must not be presented as a three-way accuracy win.  The separate
32K one-example signal (Query-Robust 1.0, Quest 0.667, ShadowKV engine
failure) remains a feasibility/quality signal on this 24 GiB machine, not a
numeric ShadowKV comparison.

At the stricter 128-token 32K point, one context-matched held-out FWE example
also tied at 0.667.  Query-Robust completed in 93.20 s (67.79 s summary
construction) and Quest in 98.56 s.  This is a latency/feasibility signal for
the current implementation, not a quality superiority claim from one example.

## Rejected refinements

- Transporting the pool to a 50-token rather than 128-token horizon did not
  change the 16K five-family score.
- Raising the mandatory recent set from one to four chunks reduced held-out
  FWE accuracy from 1.000 to 0.667 at the same total token budget.
- Raising the balanced online pool from 64 to 128 queries, or taking eight
  minimax updates, did not improve quality; the latter roughly doubled
  latency.
- Removing the mandatory recent chunk, switching shared-KV aggregation from
  mass sum to max, or transporting the empirical pool to a 16-token horizon
  did not improve the five-case 16K FWE slice.  The 16-token horizon reduced
  accuracy to 0.800.  The default sum aggregation and configured horizon are
  retained; `max` remains an explicitly named ablation only.
- Raising Query-Robust from 512 to the 528-token access point that exactly
  matches Quest's configured-512 current-page behavior also left accuracy at
  0.867.  It is therefore not added as a routine sweep budget.
- On the same five held-out 16K FWE sequences, mean-residual and minimax-
  midpoint scalar calibration, per-KV-group cache tables, a full 128-query
  FWE-only empirical pool, and eight fixed-support minimax updates all left
  accuracy at 0.867 while increasing latency.  The broader 25-sequence
  cross-task pool and the raw-score GQA aggregation each reduced it to 0.800.
  The selected serving configuration therefore remains the 64-query,
  one-update, raw-entropy, normalized-mass, shared-cache route.
- Reserving four evenly spaced, query-independent exact chunks inside the
  same 512-token budget reduced this slice from 0.867 to 0.800 and increased
  latency to 5.70 s.  It was removed rather than retained as a serving knob.
- A bounded key-only exact rerank of eight extra affine-nominated chunks did
  not change either a correct case or a known 16K FWE error, while increasing
  latency by roughly 5%.  It too was removed; the reported route is strictly
  one-stage affine-summary selection followed by exact selected attention.
- A deterministic position-stratified, 16-query-per-Q-head FWE pool was also
  worse than uniform sampling (0.800 versus 0.867 on the five held-out cases)
  and was removed.  Coverage of captured decode offsets alone is not a useful
  proxy for the empirical minimax routing objective here.
- Extending the 16K FWE calibration traces for indices 1--4 to a requested 50
  decode tokens produced 56 additional realized query steps.  A fresh,
  uniform 32-query-per-Q-head pool still scored 0.867 on held-out indices
  10--14 (5.50 s mean latency), with identical predictions to the incumbent
  pool.  The original short calibration traces are therefore not the apparent
  cause of the remaining errors on this slice.

The pool finalizer now supports an explicit task-stratum filter.  It admits
only shards whose recorded split is `calibration`, and when a task filter is
given it rejects unlabeled shards rather than guessing.  This made the FWE-
only ablation reproducible and also demonstrated that the runner's indices
5--9 validation split cannot accidentally enter a calibration pool.
- A 16K CVaR-0.5 runtime attempt lost its engine and left an orphaned CUDA
  context, so it is excluded from online candidates pending a separate solver
  stability fix.
- On the five held-out 16K FWE cases, the explicit unguarded one-step
  exploratory schedule produced exactly the same five strings and the same
  0.867 accuracy as the Armijo route.  Mean end-to-end latency improved from
  5.547 s to 5.482 s (1.2%), while mean summary-build time improved from
  4.213 s to 4.142 s.  This is the selected low-risk serving ablation for the
  already non-certified pilot only; it is not a claim of solver convergence.
- CUDA-stream prefill overlap preserved the same answer on the corresponding
  16K case but regressed latency (5.77 s versus 5.46 s).  It remains an
  opt-in scheduling experiment with event-fenced state and delayed timing
  telemetry, not a selected setting for this RTX 3090.

## Remaining bottleneck

For the 32K Query-Robust FWE run, summary construction consumed 64.9 s of
the 90.0 s runtime.  The next safe latency milestone is overlapping or
amortizing this construction without marking an unbuilt chunk ready; reducing
the balanced subpool below 64 is not justified by this pilot.
