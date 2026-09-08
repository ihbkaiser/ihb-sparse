# NVIDIA/kvpress-compatible RULER evaluation

`benchmark/kvpress_ruler/evaluate.py` evaluates the official processed RULER
artifact used by NVIDIA/kvpress (`simonjegou/ruler`) with the native
Sparse-vLLM engine. It uses the same `data_dir` context-length configuration,
greedy decoding, `answer_prefix`, and task scorer. The script requires an
explicit model path so local checkpoints are reproducible and never hardcoded.

The local Llama 3.1 checkpoint can be evaluated as follows:

```bash
MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct
python benchmark/kvpress_ruler/evaluate.py \
  --model-path "$MODEL_PATH" \
  --sparse-method quest \
  --data-dir 4096 \
  --output-dir results/kvpress-ruler/quest-4096

python benchmark/kvpress_ruler/evaluate.py \
  --model-path "$MODEL_PATH" \
  --sparse-method shadowkv \
  --batch-size 4 \
  --data-dir 4096 \
  --shadowkv-sparse-budget 2048 \
  --shadowkv-rank 160 \
  --shadowkv-chunk-size 8 \
  --output-dir results/kvpress-ruler/shadowkv-4096
```

`--batch-size` is the maximum concurrent decode batch and is passed to the
engine as `max_num_seqs_in_batch`, `max_decoding_seqs`, and
`max_num_seqs_in_gpu`. The default is 4; lower it when the local GPU cannot
hold the model plus the compressed ShadowKV workspace. Add `--decode-graph`
to capture one fixed graph for that batch bucket. ShadowKV's graph path uses
CUDA-graph-safe gather kernels for pinned host values and selected low-rank
factors. It keeps chunk landmark means in factorized form, avoiding a full
`[batch, heads, chunks, head_dim]` landmark tensor per layer; startup capture
can still consume substantial workspace at 128K, so graph and eager runs must
be reported separately. Exact full SVD is the default reference path. For a
throughput-oriented run, pass `--shadowkv-svd-method lowrank`; this uses
randomized low-rank SVD with the configured oversampling and iteration values,
so it must be reported separately because it changes the factorization basis.
Use `--gpu-memory-utilization` to expose the engine's memory budget in the
experiment manifest; this is useful for high-batch graph capture on smaller
GPUs, where the default `0.90` may leave insufficient room for the requested
KV/runtime workspace.

For a speed/memory trade-off, `--shadowkv-storage gpu_cache` keeps the full
configured prompt-capacity K/V payload for each active layer on the GPU. Set
`--shadowkv-gpu-cache-tokens` to at least `--max-model-len`; the runner fails
early when the capacity is insufficient. This is a separate experiment:
ShadowKV selection is retained, while mapped host reads and decode-time
reconstruction are removed. GPU-cache finalization computes landmark and
outlier metadata on the GPU from the exact cache copy, but skips the unused
low-rank SVD. This avoids turning long-context metadata construction into a
CPU TTFT bottleneck. `--shadowkv-kernel-backend cutlass` uses the
CUTLASS strided-batched GEMM backend and requires `--shadowkv-cutlass-root`
(or `SPARSEVLLM_CUTLASS_ROOT`) pointing to the pinned CUTLASS source tree.
For eager CPU-shadow ablations, `--no-shadowkv-multistream-gather` disables
copy-stream overlap and `--no-shadowkv-gather-copy-with-offsets` disables
selected-value chunk reuse; both are enabled by default and are recorded in
`run_info.json`.

On a B200/SM100, ShadowKV prefill uses FlashInfer's CuTe-DSL paged prefill
provider with 16-token pages. ShadowKV decode uses a dedicated FlashInfer
adapter in both eager and CUDA-graph modes: it flattens each `(request,
KV-head)` into one FlashInfer row and packs the per-head compact page table on
device. With `--shadowkv-flashinfer-backend auto`, SM100+ selects FlashInfer's
CuTe-DSL decoder because it reads graph-stable dynamic sequence lengths. The
CUDA-core FA2/FA3 path plans an exact page-count topology and is therefore
eager-only for this changing per-head payload. If FlashInfer is unavailable,
provider resolution fails explicitly; it does not silently reinterpret the
per-head payload as a shared page table.

The decode provider can be selected explicitly for matched ablations with
`--shadowkv-decode-backend {auto,flashinfer,triton}`. `auto` keeps FlashInfer
first in the ShadowKV portfolio; `flashinfer` requires the per-head FlashInfer
adapter; `triton` selects the graph-stable repository Triton per-head kernel.
An execution failure is not converted into a different provider after binding.
For FlashInfer itself, `--shadowkv-flashinfer-backend auto` selects CuTe DSL on
Blackwell; `fa2` and `fa3` remain available for eager experiments and re-plan
only when a compact row crosses a physical page boundary.

Before a long run, use `--fraction 0.01 --max-samples 2` as a smoke test.
`datasets` must be installed unless `--dataset-path` points to a local JSON or
JSONL file with the columns `context`, `question`, `answer_prefix`, `answer`,
`task`, and `max_new_tokens`.

## Paper-aligned 128K configuration

For the requested Llama-3.1-8B-Instruct RULER comparison, use a processed
`131072` artifact and keep all other protocol settings unchanged. The current
hosted `simonjegou/ruler` dataset exposes only the 4096/8192/16384 configs, so
`--data-dir 131072` is not a substitute for generating or supplying the local
128K artifact. Pass that artifact with `--dataset-path`; it must already use
the kvpress columns listed above. The official ShadowKV RULER command uses
`datalen=131072`, `sparse_budget=2048`, `rank=160`, and `chunk_size=8`
([ShadowKV repository](https://github.com/ByteDance-Seed/ShadowKV)). The
commands below use the same ShadowKV values and a matched 2048-token QuEST
budget:

```bash
MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct
RULER_128K_PATH=/path/to/processed-ruler-131072.jsonl

# QuEST: page/chunk size 16; 0 sink + 2016 scored + 32 recent = 2048 tokens.
python benchmark/kvpress_ruler/evaluate.py \
  --model-path "$MODEL_PATH" \
  --sparse-method quest \
  --batch-size 4 \
  --data-dir 131072 \
  --dataset-path "$RULER_128K_PATH" \
  --quest-chunk-size 16 \
  --sink-keep-tokens 0 \
  --decode-keep-tokens 2016 \
  --recent-keep-tokens 32 \
  --output-dir results/kvpress-ruler/quest-128k-budget2048

# ShadowKV: paper/repository configuration for 128K RULER.
python benchmark/kvpress_ruler/evaluate.py \
  --model-path "$MODEL_PATH" \
  --sparse-method shadowkv \
  --batch-size 4 \
  --data-dir 131072 \
  --dataset-path "$RULER_128K_PATH" \
  --shadowkv-sparse-budget 2048 \
  --shadowkv-rank 160 \
  --shadowkv-chunk-size 8 \
  --output-dir results/kvpress-ruler/shadowkv-128k-budget2048
```

For CUDA Graph throughput, repeat each command with `--decode-graph` and a
separate output directory. The evaluator records the selected batch and graph
settings in `run_info.json`; it does not silently fall back if graph capture
or the ShadowKV CUDA extension fails.

The current QuEST CLI defaults in this runner are `chunk_size=16`,
`sink=64`, `scored=2048`, and `recent=512`, which intentionally favor a
conservative 4096-token smoke configuration. They are not the paper-aligned
128K comparison: their effective retention budget is 2624 tokens. For a
quality-oriented QuEST ablation, the upstream example uses approximately a
1K token budget; that is a different, lower-budget experiment and should be
reported separately ([Quest repository](https://github.com/mit-han-lab/Quest),
[Quest paper](https://arxiv.org/abs/2406.10774)).

ShadowKV's primary paper knobs are `2048/160/8`; in this implementation the
default derived outlier count is 48 chunks. `shadowkv_local_chunks=4` and
`shadowkv_recent_tokens=512` are runtime view-construction details, not an
additional 2048-token budget.

Leave `--max-context-length` unset to preserve kvpress semantics: it truncates
the context portion to the tokenizer/model limit, then appends the question
and answer prefix. Set `--max-model-len` only to a value that covers the
resulting prompt plus the dataset's generated-token budget; for a local
Llama-3.1 checkpoint whose model limit is exactly 131072, verify this against
`run_info.json` before launching the full split.

The outputs are `run_info.json`, `raw_outputs.jsonl`, `parsed_outputs.jsonl`,
`per_sample_results.jsonl`, and `aggregate_metrics.json`. Every sample has an
explicit terminal status. Sparse-vLLM currently compresses after its combined
context/question prefill, whereas kvpress compresses context before the
question pass; this semantic difference is recorded in `run_info.json` and
must be included when interpreting comparisons.
