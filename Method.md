# Query-Robust Affine Chunk Routing

## Decision and claim status

This document specifies a training-free router for exact sparse decode attention. For every `(layer, local KV head, finalized chunk)` it stores one head-dimensional vector and one scalar. The stored summary is independent of the live decode query; the live query is used only to score summaries and choose chunks. Attention over the chosen chunks is unchanged and uses the original KV cache.

The robust implementation uses the **empirical minimax tangent** as its primary objective and supports weighted empirical CVaR as a tail-risk alternative. Both solve for a simplex tangent vector against the explicit query pool; neither replaces the worst-case/tail objective with an expectation or mean-query surrogate. The minimax solve is active-set based, warm-started, and certified by a full-pool violation scan. The online decode representation remains one vector plus one scalar per chunk, while the solve is paid during prompt/chunk summary construction.

Claim labels used below:

- **Identity / proposition**: proved here from the stated assumptions.
- **Algorithmic guarantee**: exact up to the reported numerical optimization gap and summary quantization.
- **Pilot observation**: synthetic CPU evidence only.
- **Paper-reported**: attributed to a primary source and not reproduced here.
- **Hypothesis**: must pass the evaluation gates in this document.

The likely contribution is not the KL identity by itself—closely related entropy-corrected landmarks already appear in HiLS—but the combination of a frozen empirical query distribution, a post-fit bias scalar, auditable mean/tail/worst-case objectives, explicit GQA routing, and integration into this exact-attention evaluation stack. This is a working novelty hypothesis, not a novelty claim.

The decisive alternatives are:

| Candidate | Decision |
|---|---|
| Analytic mean tangent + mean residual | Analysis-only comparator. It is not the robust runtime objective. |
| Raw analytic tangent | Keep only as a lower-bound diagnostic; it is not a permitted robust runtime objective. |
| Uniform mean key + $\log n$ | Mandatory baseline; reject as primary because it ignores the empirical query distribution. |
| CVaR tangent | Permitted tail-risk runtime objective when the full empirical caps and tail bound are retained. |
| Minimax tangent | **Primary robust runtime objective.** Active-set solve with full-pool scan and an explicit $U-L$ certificate. |
| Unconstrained affine ridge/least squares | Keep as a same-budget ablation; reject as primary because it lost tangent structure and underperformed calibrated tangent routing in the matched pilot. |
| Direct listwise ranking | Defer; it couples chunks/documents, loses the separable convex build, and lacks evidence sufficient to justify that complexity. |

## Deployment contract

Let a finalized chunk contain post-position-encoding keys $K=(k_1,\ldots,k_n)$ with $k_j\in\mathbb R^d$. For a live post-position-encoding query $q$, define

$$
s_j(q)=\frac{q^\top k_j}{\sqrt d},\qquad
f_K(q)=\log\sum_{j=1}^n \exp s_j(q).
$$

$f_K(q)$ is the chunk's exact log unnormalized attention mass. The router approximates it by

$$
\widehat f_K(q)=\frac{q^\top\bar k_K}{\sqrt d}+b_K,
$$

where $\bar k_K\in\mathbb R^d$ and $b_K\in\mathbb R$ are computed from the fixed keys and an offline query pool. Runtime selection ranks or normalizes these scores; the selected tokens then receive exact attention. Values are never summarized or reconstructed.

The primary scope is batch-one autoregressive decode with dense prefill, matching the repository's current vLLM path. Finalized historical chunks are routable. The current partial chunk, configured recent chunks, and any required sink chunks are kept exactly and count against the token budget. Causal prefill, sliding-window/hybrid layers, cross-request prefix sharing, and multi-request batching are out of the first implementation.

“Query-blind” therefore means **summary construction never observes the live request's future query**. It does not mean that routing ignores the current query. Methods such as Quest are current-query-aware in exactly this runtime sense; SnapKV additionally uses queries from the current prompt to decide what to retain, which is outside this contract.

## The affine family and what it can represent

Choose $p\in\Delta_n$, the probability simplex, and define

$$
\bar k(p)=\sum_{j=1}^n p_j k_j,\qquad
H(p)=-\sum_{j=1}^n p_j\log p_j,
$$

with $0\log 0=0$. The entropy-corrected affine score is

$$
T_p(q)=\frac{q^\top\bar k(p)}{\sqrt d}+H(p).
$$

Let

$$
a_j(q)=\frac{\exp s_j(q)}{\sum_\ell\exp s_\ell(q)}.
$$

Then the following is an exact identity:

$$
f_K(q)-T_p(q)=D_{\mathrm{KL}}\!\left(p\,\middle\|\,a(q)\right)\ge 0.
$$

Thus $T_p$ is a global affine lower bound. If $p=a(\bar q)$ for some $\bar q$, it is the tangent to $f_K$ at $\bar q$.

Uniform $p_j=1/n$ gives $\bar k=n^{-1}\sum_jk_j$ and $H(p)=\log n$, exactly the mean-key baseline with its chunk-size correction.

This family is not an arbitrary heuristic class. Put the scaled keys in the rows of $A\in\mathbb R^{n\times d}$, so $f_K(q)=\operatorname{LSE}(Aq)$. Its convex conjugate is

$$
f_K^*(v)=
\min_{p\in\Delta_n:\,A^\top p=v}\sum_jp_j\log p_j,
$$

for $v$ in the convex hull of the scaled keys, and $+\infty$ otherwise. Consequently, at a fixed feasible slope $v$, the strongest global affine lower bound is $v^\top q-f_K^*(v)$ and is realized by the maximum-entropy $p$ satisfying $A^\top p=v$. Any other $T_p$ with the same slope is dominated. Therefore all undominated global affine lower bounds are represented by maximum-entropy members of this family. This does **not** say they are the best affine fits on a finite calibration set: an unconstrained affine regression can fit that set better by giving up the global lower-bound and extrapolation structure.

For empirical queries $q_1,\ldots,q_N$, define

$$
S_{ij}=\frac{q_i^\top k_j}{\sqrt d},\qquad
f_i=\log\sum_j\exp S_{ij},\qquad
g_i(p)=f_i-(Sp)_i-H(p).
$$

Every $g_i(p)$ is the reverse-direction KL divergence $D_{\mathrm{KL}}(p\|a(q_i))$. It is convex in $p$. Reversing this KL direction would produce a different objective and would invalidate the closed forms below.

## Three fitting objectives

### Weighted mean: analysis-only comparator

For weights $w_i\ge0$, $\sum_iw_i=1$,

$$
\min_{p\in\Delta_n}\sum_iw_i g_i(p)
$$

has the unique interior solution

$$
p_{\mathrm{mean}}=\operatorname{softmax}(S^\top w)=a\!\left(\sum_iw_iq_i\right).
$$

This is cheap and stable, but it is not a robust objective. It is retained only as an explicitly labeled analysis comparator; it is forbidden as the Query-Robust runtime mode because it minimizes expected gap rather than worst-case or tail error.

For the stored intercept, fit the weighted calibration residual

$$
c_K=\sum_iw_i g_i(p),\qquad b_K=H(p)+c_K.
$$

For a fixed $p$, $c_K$ is the scalar that minimizes weighted squared log-mass error. It deliberately sacrifices the raw lower-bound property to remove per-chunk bias that can corrupt ranking. The build needs one query-by-key matrix product, reductions, and a softmax, with no iterative optimization. Store the unshifted gap statistics in the audit artifact so the raw lower bound remains reproducible.

### Worst case: primary exact dual and certificate mode

The empirical minimax problem is

$$
\epsilon^*=\min_{p\in\Delta_n}\max_i g_i(p).
$$

Its dual is

$$
\epsilon^*=
\max_{\lambda\in\Delta_N}
\left[
\lambda^\top f-\log\sum_j\exp\left((S^\top\lambda)_j\right)
\right].
$$

At a dual optimum,

$$
p^*=\operatorname{softmax}(S^\top\lambda^*).
$$

Equivalently, $\bar q^*=\sum_i\lambda_i^*q_i$ and $p^*=a(\bar q^*)$, so the recovered lower bound is a tangent at a convex combination of empirical queries.

Strong duality holds for the finite empirical problem: its epigraph form is convex, $\Delta_n$ is compact, and strict feasibility follows by choosing an interior $p$ and a $t$ larger than every $g_i(p)$. For any feasible $\lambda$ and recovered $p$, the computable bounds

$$
L(\lambda)=\lambda^\top f-\operatorname{LSE}(S^\top\lambda),
\qquad
U(p)=\max_i g_i(p)
$$

satisfy $L\le\epsilon^*\le U$. The difference $U-L$ is the optimization certificate; iteration count or gradient norm is not.

Because $g(q)=f_K(q)-T_p(q)$ is convex in $q$, $0\le g(q)\le U$ on the convex hull of the calibration queries. Storing $b=H(p)+U/2$ gives

$$
\left|f_K(q)-\widehat f_K(q)\right|\le U/2
$$

on that hull, at no additional runtime bytes. This shifted score is no longer a global lower bound. It is the best constant centering of this fixed minimax tangent bound, not necessarily the best unconstrained affine Chebyshev fit.

### Tail CVaR: risk-sensitive upgrade

Minimax gives one sample unlimited leverage. Weighted empirical CVaR interpolates between the mean and worst case:

$$
\min_{p\in\Delta_n}\operatorname{CVaR}_\alpha\{g_i(p);w_i\},
\qquad 0\le\alpha<1.
$$

Its dual has the same smooth objective as minimax but a capped simplex:

$$
\max_\lambda
\left[
\lambda^\top f-\operatorname{LSE}(S^\top\lambda)
\right]
$$

subject to

$$
\sum_i\lambda_i=1,\qquad
0\le\lambda_i\le\frac{w_i}{1-\alpha}.
$$

The recovery formula remains $p=\operatorname{softmax}(S^\top\lambda)$. At $\alpha=0$, the cap forces $\lambda=w$ and recovers the mean solution. As $\alpha\to1$, the cap disappears and recovers minimax. The proposed experimental value is $\alpha=0.95$, meaning that the empirical worst five percent drives the slope while no individual query can receive arbitrary mass. This value is a starting point, not an established model default; $\{0,0.9,0.95,0.99,\text{minimax}\}$ must be swept.

Use the same weighted calibration-residual rule

$$
c_K=\sum_iw_i g_i(p),\qquad b_K=H(p)+c_K.
$$

For a fixed $p$, $c_K$ is the scalar that minimizes weighted squared log-mass error. It deliberately sacrifices the raw lower-bound property to remove per-chunk bias that can corrupt ranking. Keep $U_K=\max_i g_i(p)$ in the offline audit artifact; since $0\le c_K\le U_K$, an a-posteriori hull bound is

$$
|f_K(q)-\widehat f_K(q)|\le\max(c_K,U_K-c_K).
$$

That bound is generally looser than the centered-minimax bound. CVaR is retained as a distributional-tail ablation; it becomes the default only if held-out model evidence justifies its iterative summary-build cost.

### Why direct ranking loss is not the primary solver

A listwise or pairwise loss can target top-$k$ recall more directly, but chunk rankings couple all chunks, depend on the competing document, and are discontinuous at the cutoff. A per-chunk optimizer would lose the clean convex problem, the one-chunk-at-a-time preprocessing path, and the minimax certificate. A differentiable listwise calibration of only the scalar $c_K$ is a useful later ablation. It should not replace the auditable per-chunk objectives before the basic approximation has passed held-out tests.

The objectives are genuinely different. In particular,

$$
\mathbb E[\log Z(q)],\qquad
\log\mathbb E[Z(q)],\qquad
\mathbb E[\operatorname{softmax}(qK^\top)]
$$

are not interchangeable. The mean-KL objective above optimizes the first expression up to its affine terms. Expected Attention models moments of $\exp(q^\top k)$ and therefore targets the second kind of quantity before normalization; it is not an estimator of expected softmax attention without further treatment of the random denominator.

## Routing rule and GQA

There is one summary per KV head, not per query head. Let $G(h_{kv})$ be the query heads sharing a KV head. Each live query head computes its summary scores $\widehat f_{hc}$. Raw scores from different query heads should not be summed: their additive scales can differ. Instead compute predicted chunk shares within each query head,

$$
\widehat m_{hc}=
\frac{\exp\widehat f_{hc}}{\sum_{c'}\exp\widehat f_{hc'}},
\qquad
r_c=\sum_{h\in G(h_{kv})}\widehat m_{hc},
$$

and select chunks with the largest $r_c$, after reserving budget for mandatory chunks. This follows the value-agnostic GQA relaxation derived in COBS; it is not an exact output-error optimum. Max aggregation, raw-score summation, and per-query-head summaries are required ablations. Per-query-head summaries multiply metadata by the GQA ratio and violate the strict one-vector-plus-one-scalar-per-KV-head budget.

If every chunk $c$ has a valid absolute log-score error bound $E_c$, then a true pairwise order $f_a>f_b$ is preserved whenever

$$
f_a-f_b>E_a+E_b.
$$

The analogous top-$k$ guarantee requires this separation for every selected/unselected pair at the boundary. This is a conditional ranking guarantee, not a claim that empirical-hull bounds cover deployment queries.

## Query-pool construction

The query pool is the most important experimental control. It must be captured once from the same frozen model revision and then held fixed while request-specific chunk summaries are built. No gradient update, learned landmark, or task-label supervision is used.

For each layer and query head, capture real queries after the model's Q/K normalization and RoPE—the representation passed by `vllm_patched_forward` to the attention implementation before the kernel applies the $1/\sqrt d$ scale. Do not L2-normalize these vectors. Store sequence ID, token position, prompt length, decode offset, domain/task stratum, model revision, RoPE configuration, dtype, TP degree, and the query-head-to-KV-head mapping.

Use a sequence-level split so queries from one document cannot enter both calibration and evaluation. Build the pool with equalized or explicit weights over:

- target-position buckets, especially prompt-end and later decode offsets;
- task/domain buckets represented in RULER and any target workloads;
- all query heads mapped to the KV head, retaining head labels for balanced sampling;
- ordinary and deliberately hard examples, with contamination tests run separately.

The first capacity sweep should use $N\in\{512,2048,4096,16384\}$ per `(layer, KV head)` after stratification. Start at 512 to bound resident pool memory and summary-build latency; increase only after plotting held-out saturation. None of these is an established production value.

RoPE makes positions part of the data distribution. The safest first path is to capture post-RoPE queries in target-position buckets and match a summary build to the plausible future-query buckets for that request. Rephasing a post-RoPE query by $R(t_{target})R(t_{source})^{-1}$ is valid only for the exact model's standard RoPE convention and must be checked against the model implementation. For partial, multidimensional, or otherwise modified RoPE, capture at the target positions or replay pre-RoPE vectors through the model's actual rotary operator. Never mix pre-RoPE queries with post-RoPE keys.

Distribution refresh is an offline event triggered by a model/checkpoint, tokenizer/context policy, RoPE/scaling, workload mixture, or statistically detected query-drift change. It is not per request. Chunk summaries are request-specific because their keys are request-specific, and are built when a chunk becomes final.

## Active constraint generation

For minimax, and for high-$\alpha$ CVaR with many zero-weight dual coordinates, use a working set of query constraints:

1. Initialize $M_0$ with stratified queries, not the first rows of the pool. Proposed pilot value: $M_0=64$.
2. Solve the dual on the working set in FP64, recovering $p=\operatorname{softmax}(S_M^\top\lambda)$.
3. Scan the full pool in FP32 with stable log-sum-exp and compute $U=\max_i g_i(p)$.
4. For minimax, compute the feasible dual lower bound $L$ and add the largest violating queries not already active. The online implementation admits up to four per chunk per scan; this reduces active-set rounds without changing the support budget, objective, or full-pool certificate.
5. Stop only when $U-L\le10^{-3}$ log units, or report the unresolved gap after 256 active constraints. Tighter tolerances are useful for solver tests but not justified for BF16 storage.

For certificate mode, active constraints are selected separately for each chunk: a shared support can miss a chunk-specific attention mode. A shared per-head support is allowed only as an initialization/cost baseline, followed by the same full-pool violation scan for every chunk. Compare random, stratum-balanced, farthest-point-in-logit-space, shared-active, and per-chunk-active initializations at a fixed objective and scan budget. This initialization ablation was not run without model activations; it is a required remaining pilot. Raw Euclidean distance or PCA in query space is only a baseline because the relevant geometry is $QK^\top$ and varies by chunk.

For CVaR, the dual cap can require at least roughly $(1-\alpha)N$ supported points under uniform weights. Use capped-simplex projected ascent with Armijo backtracking, or solve the convex primal epigraph form. The full-pool tail objective and maximum gap must be scanned after every candidate solve. A scan over only the active set is not a certificate.

The stored BF16/FP16 vector changes the mathematical score. After quantization, recompute all calibration residuals using the exact stored representation and stored FP32 scalar. Audit the resulting mean, tail, and maximum absolute errors. If a certificate is required, the certificate must be based on this quantized rescan, not the FP64 optimizer state.

Fallback is deterministic: if the solver fails, the certificate gap exceeds its threshold, the query-pool metadata does not match the model, or the summary is not ready, use dense attention over the intact cache. Never silently use a mismatched pool or an uncertified sparse summary.

## Masks, chunk boundaries, and positional edge cases

- Dense prefill is the primary path. A fixed chunk summary does not encode the query-dependent causal mask within a prefill block.
- The stated formulas assume pure dot-product logits scaled by $1/\sqrt d$. A model with a different attention scale, ALiBi/additive bias, logit soft-capping, or another logit transform must either incorporate that operation into the target and summary derivation or be rejected. The first implementation rejects it.
- Only complete historical chunks enter routing. The incomplete prompt tail and current generated chunk remain exact. When the generated chunk fills, enqueue summary construction; keep it in the recent exact set until the summary is ready.
- A summary is built with the chunk's true token count $n$. The entropy term automatically reflects $n$, but mixed chunk sizes should not be routed until the partial-chunk policy is tested.
- Sink and recent chunks are always selected and count against the total token budget. The pilot starts with one sink and one recent complete chunk; both values require ablation.
- Logical chunk size is independent of the physical vLLM cache block but must divide it cleanly in the first implementation. Start at 16 tokens to match the current Quest configuration and compare 8, 16, and 32.
- Reject sliding-window/hybrid layers initially, as the current ShadowKV path does. Their validity set changes independently of routing.
- Summary reuse under prefix caching is legal only when keys, absolute positions, layer/head identity, model metadata, and pool version all match. Prefix caching is currently disabled in this repository.

## What the synthetic pilots establish (historical solver checks)

These pilots used NumPy/SciPy on CPU with $d=32$, $n=16$, Gaussian keys/queries, stable log-sum-exp, and no model activations. They test formulas and failure modes, not language-model quality.

1. **Competing modes.** On a four-key/two-mode logit toy, the analytic mean tangent had maximum gap 2.20665 and minimax 2.20293. The small but correctly directed reduction checks that the objectives resolve competing attention modes differently without implying a useful routing gain.
2. **Active-set correctness.** With 512 calibration queries, a stratified 16-query start reached 19 active constraints in three solves. The final lower bound was $1.298066920$, the upper bound $1.298066953$, and the gap $3.32\times10^{-8}$. This supports the dual, recovery, scan, and stopping rule.
3. **Outlier sensitivity.** A 512-query pool with one extreme query gave calibration maximum gaps of 15.77 (mean), 14.43 (CVaR-0.95), and 3.48 (minimax). On 4,000 ordinary held-out queries, mean gaps were 0.421, 0.443, and 1.672 respectively. Minimax protected the outlier but paid about $4\times$ ordinary-query mean error. This is the reason it is not the default.
4. **Routing sanity check.** Across three seeds with 12 chunks, top-3 selection, 256 calibration queries, and 1,000 test queries, mean/CVaR/minimax top-3 recalls were 0.669/0.661/0.623 on IID queries and 0.661/0.652/0.614 after a coordinate shift. Adding the mean residual to the mean score gave recalls 0.670 IID and 0.660 shifted while reducing mean absolute log-score error from 0.458/0.491 to 0.160/0.170. CVaR used the same residual rule; minimax used $U/2$. The differences are illustrative only.
5. **Same-budget unconstrained affine fit.** Ordinary least squares over the same 256 queries stored one vector/scalar but gave IID/shifted top-3 recall 0.649/0.641 and mean absolute error 0.172/0.183, versus 0.670/0.660 and 0.160/0.170 for calibrated tangent. Ridge values 0.1 and 1 did not reverse the result. Under a deliberately extreme calibration contaminant, unconstrained regression fit that extreme regime far better but more than doubled ordinary-query error, illustrating the lost structural regularization.

These observations are retained as solver sanity checks, not as a reason to substitute expectation for the robust objective. They do not establish CVaR-0.95 as superior to minimax. Model-level held-out evidence must decide whether CVaR is worth its additional build cost.

## Real Pile calibration and robust pilot

The reproducible calibration artifact was captured from the frozen
`NousResearch/Meta-Llama-3.1-8B-Instruct` snapshot
`d10aef7999a2b5ba950ab3974312feeedbfe0b77`. Twenty sequences of exactly 2,048
tokens were selected from `monology/pile-uncopyrighted` (Hub revision
`3be90335b66f24456a5d6659d9c8d208c0357119`), using seed 43. The model was
forwarded in BF16, and a uniform-priority reservoir retained 3,000 actual
post-RoPE/pre-scale query vectors per layer and Q head without replacement.
Positions, sequence IDs, Q-to-KV mapping, source row hashes, and all sampling
metadata are recorded in the schema-2 artifact manifest. No SVD direction or
mean-query replacement is used.

On a separate Pile sequence (seed 1043), a bounded layer-31, four-chunk pilot
used the minimax active-set solver initialized at 32 queries and capped at 64
support points. Across all eight KV heads, robust summaries reduced held-out
absolute log-mass error relative to the uniform token baseline; the maximum
reported empirical optimization gap was 0.7227 in the deliberately bounded
support/iteration run (mean gap 0.109 across heads). This is a
directional pilot, not a downstream RULER claim. Full-prompt minimax timing is
reported separately because its per-chunk robust solves are the dominant setup
cost.

The eager online implementation batches all local KV heads and chunk groups on
the GPU, keeps the frozen Pile rows in BF16 between builds, and admits up to
four full-pool violators per independent chunk support at once. The latter is
only a solve schedule: every returned fit still gets a complete-pool `U-L`
certificate. On the held-out four-chunk layer-31 pilot, it reduced solve time
from 4.80 s to 3.05 s without degrading the measured held-out log-mass signal.
On the matched 34-token real-vLLM smoke, summary construction fell from 39.27 s
to 7.54 s. The deliberately small support cap still triggered the documented
dense fail-closed fallback in both runs. This establishes integration, safety,
and a setup-latency improvement—not production sparse decode latency.

## Evaluation gates

All comparisons use identical total selected-token budgets and exact attention over the selected KV entries. Report per layer/head and aggregate distributions, not only means.

The minimum diagnostic suite is:

- signed and absolute chunk log-mass error;
- calibration and held-out mean, 95th/99th percentile, CVaR, and maximum error;
- top-$k$ chunk recall against exact $f_K(q)$;
- retained exact attention mass;
- relative attention-output error using the real values;
- downstream task score;
- summary-build time, queries/second, active-set size, solver failures, and quantized certificate gap;
- per-token decode latency, peak memory, and summary bytes;
- results split by position, sequence, task/domain, layer, query head, and KV head.

Mandatory ablations are analytic mean raw, analytic mean plus residual, uniform mean key, unconstrained affine ridge, CVaR levels, minimax raw, minimax midpoint, no entropy scalar, key mean plus fitted scalar, UNIQUE-style dispersion scalar, Quest, ShadowKV, dense, query-pool size, active-set initialization/support sharing, chunk size, GQA aggregation, sink/recent policy, FP32 versus BF16 storage, and IID versus deliberately shifted/contaminated pools. Recent-query baselines appear only in a separately labeled query-visible setting.

The method advances only if, on held-out sequences, it improves either recall/retained mass at fixed latency and bytes or latency/bytes at fixed quality over the analytic mean and current Quest baseline. Minimax is retained only if its worst-tail gain survives without material ordinary-query or downstream regression. A calibration-hull certificate alone is not a quality result.

## Literature audit

No PDFs were present under `papers/` or `literature/`; the supplied local note was read in full. The table below uses primary paper/project pages. The configured deterministic `verify_papers.py` helper was unavailable, so every identity retains the required **⚠ UNVERIFIED (helper unavailable)** status even though its title and identifier were manually matched to a live arXiv or PMLR page. Results below are paper-reported and were not independently reproduced.

| Work | Objective and query visibility | Summary / relevance | Paper-reported result | Status and primary evidence |
|---|---|---|---|---|
| Quest | Current live query scores per-page min/max key bounds; no learned router. | Two extrema vectors per dimension, query-aware upper-bound selection; current baseline here. | Up to $2.23\times$ self-attention speedup and $7.03\times$ lower inference latency with negligible reported accuracy loss. | ⚠ UNVERIFIED; [paper abstract and §3](https://proceedings.mlr.press/v235/tang24l.html), [official code](https://github.com/mit-han-lab/Quest) |
| ShadowKV | Current query scores prompt-derived landmarks; low-rank key reconstruction and outlier retention. | Post-RoPE mean landmarks plus shared low-rank state; broader cache-compression system than this router. | Up to $6\times$ larger batches and $3.04\times$ throughput on A100 in the reported setup. | ⚠ UNVERIFIED; [paper abstract and §§3–4](https://arxiv.org/html/2410.21465), [official code](https://github.com/bytedance/ShadowKV) |
| SnapKV | Uses the current prompt's trailing observation-window queries to evict prompt KV. | Query-visible cache eviction, not frozen future-query-distribution fitting or exact attention over an intact cache. | At 16K input, $3.6\times$ generation speed and $8.2\times$ memory efficiency versus its baseline. | ⚠ UNVERIFIED; [paper abstract and §3.2](https://arxiv.org/html/2404.14469), [official code](https://github.com/FasterDecoding/SnapKV) |
| RetrievalAttention | Current live queries retrieve keys with an ANN index; analyzes query/key distribution mismatch. | Supports modeling query statistics, but stores an index rather than one affine chunk summary. | Near-full reported accuracy while accessing 1–3% of data; 128K/8B served on one RTX 4090 at 0.188 s/token. | ⚠ UNVERIFIED; [paper abstract and §§3.1–3.2](https://arxiv.org/html/2409.10516), [official code](https://github.com/microsoft/RetrievalAttention) |
| Q-Filters | Offline real-query activations define layer/head low-rank filters; inference is query-independent until filtered key scoring. | Direct precedent for frozen per-head query statistics; not a chunk log-mass approximation. | 99% needle accuracy at $32\times$ compression and up to 65% smaller perplexity degradation than StreamingLLM. | ⚠ UNVERIFIED; [paper abstract and §§3, 4.1](https://arxiv.org/html/2503.02812), [official code](https://github.com/NathanGodey/qfilters) |
| Expected Attention | Fits a Gaussian future-query distribution from available queries and estimates expected unnormalized attention. | Targets moments of exponentiated logits; distinguishes distributional scoring from mean log mass. | Reports outperforming its compression baselines in both prefill and decode without a numerical abstract-level aggregate. | ⚠ UNVERIFIED; [paper abstract and §3](https://arxiv.org/html/2510.00636), [official implementation in KVPress](https://github.com/NVIDIA/kvpress) |
| UNIQUE | Uses a key mean vector and a scalar dispersion correction. | Closest strict storage-budget comparator, but with a heuristic dispersion score rather than KL fitting. | Up to $11.4\times$ attention-kernel and at least $5.3\times$ end-to-end decode speedup over named dense baselines. | ⚠ UNVERIFIED; [paper abstract and §3](https://arxiv.org/html/2605.27740) |
| HiLS | Entropy-corrected affine landmark approximates log chunk mass; its full system learns landmarks/query subspaces. | Closest mathematical precedent for $q^\top\bar k+H(p)$; learned components differ from this solver. | More than $64\times$ training-length extrapolation with 90% reported retrieval accuracy. | ⚠ UNVERIFIED; [paper abstract and §3](https://arxiv.org/html/2607.02980), [project page](https://github.com/Tencent-Hunyuan/HiLS-Attention) |
| COBS | Derives mass-based block selection, GQA aggregation, cumulant approximations, and query subspaces. | Supports normalized GQA aggregation; second-order/subspace state exceeds the strict budget. | 32K RULER mean 0.8195 versus NSA 0.2999 and dense 0.9040, with $15.15\times$ less read traffic than dense. | ⚠ UNVERIFIED; [paper abstract and §§3–5](https://arxiv.org/html/2607.09052) |
| LOCKS | Reconstructs page logits from a page-local low-rank spectral representation. | Query-independent after finalization and higher capacity than one vector plus scalar. | At 1M tokens, reported $2.0\times$ per-token speedup at rank 8; about 2% tokens attended at a 2048-token budget in one setup. | ⚠ UNVERIFIED; [paper abstract and §§2, 4](https://arxiv.org/html/2607.24555) |

The relevant distinctions are therefore explicit: some methods see current prompt queries during compression (SnapKV, ShadowKV), some see the current decode query only during routing (Quest, RetrievalAttention, all affine-summary routers), some rely on frozen or estimated query distributions (Q-Filters, Expected Attention, this proposal), and some learn summary parameters (HiLS). Those regimes must not be compared as if they had identical information or storage budgets.

## Guarantees and non-guarantees

Guaranteed under exact arithmetic and the stated empirical problem:

- $T_p$ is a global lower bound and its gap is $D_{\mathrm{KL}}(p\|a(q))$.
- The finite minimax dual is exact; a feasible primal/dual pair supplies an optimization gap.
- A full calibration-pool scan supplies empirical tail and maximum errors.
- The centered minimax score has an absolute-error bound on the calibration-query convex hull.
- Attention over the selected tokens remains exact.

Not guaranteed:

- that deployment queries lie in or near the empirical convex hull;
- that lower log-mass error yields the best top-$k$, attention-output, or downstream quality;
- that CVaR-0.95 is the best risk level;
- that a query pool transfers across model, position, RoPE, head, or workload changes;
- that the affine form remains valid under unmodeled attention bias, soft-capping, or nonstandard scaling;
- that preprocessing cost is acceptable at production scale;
- that the proposal is novel relative to all concurrent work;
- any quality, latency, or memory improvement on a real model until the implementation and held-out pilots in `ImplementationNote.md` are run.
