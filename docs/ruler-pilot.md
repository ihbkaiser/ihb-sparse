# Llama-3.1 RULER pilot (8K/16K/32K)

This pilot uses exactly 50 examples from each of five official RULER
configurations, saved in one immutable mixed-task JSONL:

| Public row task | Official RULER configuration |
| --- | --- |
| `niah_single` | `niah_single_1` |
| `niah_multikey` | `niah_multikey_1` |
| `niah_multiquery` | `niah_multiquery` |
| `vt` | `vt` |
| `fwe` | `fwe` |

The task name is stored in every row. The file includes both `input_text` /
`gold_answer` and RULER-compatible `input` / `outputs` fields, along with the
task configuration, token accounting, seed, and official variant. The
generator refuses to overwrite an existing output.

## Prepare data

The essay variants require the official RULER Paul Graham corpus. A checkout of
RULER at a pinned commit can produce it with its `download_paulgraham_essay.py`
utility. Then run:

```bash
python -m sparse_frontier.ruler_pilot \
  --model_path /path/to/Llama-3.1-8B-Instruct \
  --essay_path /path/to/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json \
  --output experiments/data/ruler_8k_pilot.jsonl \
  --samples_per_task 50 --max_seq_length 8192 --seed 20260831
```

The output is generated once. To use a different tokenizer or seed, choose a
new output path and keep its run metadata with the resulting file.

To create same-seed, same-task 16K or 32K counterparts, use a new output path
and change only `--max_seq_length`, for example:

```bash
python -m sparse_frontier.ruler_pilot \
  --model_path /path/to/Llama-3.1-8B-Instruct \
  --essay_path /path/to/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json \
  --output experiments/data/ruler_16k_pilot.jsonl \
  --samples_per_task 50 --max_seq_length 16384 --seed 20260831
```

Use `32768` and a separate output path for 32K. The task/sample seeds and gold
answers remain aligned with the 8K pilot, while the prompt contexts are longer.

## Smoke and matrix

Run the one-example smoke pass before the full matrix. It uses batch size one,
BF16, 8K input capacity, deterministic temperature-zero generation, and the
same dataset row for all methods:

```bash
python -m sparse_frontier.ruler_runner \
  --data_path experiments/data/ruler_8k_pilot.jsonl \
  --model_path /path/to/Llama-3.1-8B-Instruct \
  --output_dir experiments/results/ruler_8k_pilot_smoke \
  --smoke
```

The full run has seven configurations: Dense once, and Quest and ShadowKV at
512, 1024, and 2048 tokens:

```bash
python -m sparse_frontier.ruler_runner \
  --data_path experiments/data/ruler_8k_pilot.jsonl \
  --model_path /path/to/Llama-3.1-8B-Instruct \
  --output_dir experiments/results/ruler_8k_pilot
```

Each run writes a dataset snapshot, per-example `predictions.jsonl`,
`aggregate.json`, `aggregate.csv`, and `run.json`. Failures are persisted and
counted; the command exits nonzero if any example fails. The generic evaluator
can also be called independently:

```bash
python -m sparse_frontier.evaluation \
  --data_path /path/to/ruler_8k_pilot.jsonl \
  --predictions_path /path/to/predictions.jsonl \
  --output_json /path/to/aggregate.json \
  --output_csv /path/to/aggregate.csv \
  --method shadowkv --budget 2048
```

ShadowKV here is the accuracy/correctness path described in the main README:
prefill is dense/exact, pre-RoPE K is captured for the global SVD, and the
existing vLLM cache remains dense. The upstream CPU-offloaded cache layout is
out of scope, so no memory-saving claim is made. An optional local CUDA
extension can fuse reconstruction and landmark scoring; it does not change
the cache layout or algorithm.

## Faster ShadowKV experiments

The default runner uses randomized SVD with the configured oversampling and two
power iterations. The decode path uses reusable assembled-token buffers and
direct varlen FlashAttention. To use the faithful official full SVD, pass
`--shadowkv_svd_backend exact`. To keep a full SVD while selecting cuSOLVER's
CUDA Jacobi driver, use:

```bash
/usr/bin/python3 -m sparse_frontier.ruler_runner \
  --data_path experiments/data/ruler_8k_pilot.jsonl \
  --model_path /path/to/Llama-3.1-8B-Instruct \
  --output_dir experiments/results/ruler_8k_pilot_gesvdj \
  --shadowkv_svd_backend gesvdj
```

`gesvdj` remains a full decomposition of the same pre-RoPE matrix, but its
floating-point factors are not promised to be bitwise identical to official
`torch.svd`; a cuSOLVER convergence failure emits a warning and falls back to
official `torch.svd`. On the current Llama-3.1 smoke, cuSOLVER did fail and this
option was slower, so it is hardware-dependent. To trade exact SVD equivalence
for more speed, use the explicit randomized backend:

```bash
/usr/bin/python3 -m sparse_frontier.ruler_runner \
  --data_path experiments/data/ruler_8k_pilot.jsonl \
  --model_path /path/to/Llama-3.1-8B-Instruct \
  --output_dir experiments/results/ruler_8k_pilot_randomized \
  --shadowkv_svd_backend randomized \
  --shadowkv_svd_oversample 32 \
  --shadowkv_svd_niter 2
```

This only changes how the low-rank factors are computed. Landmark scoring,
minimum-cosine outliers, GQA aggregation, selected-chunk ordering, original
position RoPE reconstruction, value gathering, and FlashAttention are
unchanged. The randomized mode is approximate and must not be presented as
bitwise-equivalent to the official full-SVD baseline.

To build and use the optional fused GPU stages, run:

```bash
SF_BUILD_SHADOWKV_CUDA=1 python setup.py build_ext --inplace
```

They are enabled automatically when built. Use `--no-shadowkv_fused_retrieval`
to retain Torch landmark scoring, or `SF_SHADOWKV_CUDA=off` to use the complete
verified PyTorch fallback. The fused implementation is benchmarked for the
current BF16 Llama shape; other GPUs should be measured independently.

On GPUs where the default vLLM utilization leaves insufficient KV-cache
capacity for 32K, set `SF_GPU_MEMORY_UTILIZATION=0.96` for that run. If the
full dense prefill runs out of activation memory, also set
`SF_MAX_NUM_BATCHED_TOKENS=4096`; the vLLM hook preserves exact prefill for
Quest in this chunked mode. ShadowKV still requires one non-chunked prefill for
its global pre-RoPE SVD and fails explicitly when that mode is unavailable. The
defaults remain `0.85` and the full requested token count, respectively.

For a 24 GiB card that cannot fit ShadowKV's full prefill with BF16 weights,
the runner also supports vLLM's opt-in weight offload via
`SF_CPU_OFFLOAD_GB=3`. This is weight offload only; it does not port
`ShadowKVCache_CPU` or offload the KV cache, and it may be much slower. The
setting is written to `run.json` so results are auditable.

The faster manual-cache variant uses `SF_KV_CACHE_MEMORY_BYTES` together with
a smaller weight offload. For the current 33,792-token engine,
`SF_KV_CACHE_MEMORY_BYTES=4429185024` reserves exactly 4.125GiB for the dense
KV cache; `SF_CPU_OFFLOAD_GB=0.5` then keeps most weights on GPU. This remains
the same correctness path, but leaves essentially no room for a second full
32K request, so increasing batch size is unsupported until ShadowKV state is
made per-request.
