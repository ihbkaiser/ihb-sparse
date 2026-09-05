# Sparse Frontier

Sparse-attention evaluation for vLLM. The RULER runner evaluates Dense, Quest,
ShadowKV, and Query-Robust on the canonical [KVPress RULER dataset](https://huggingface.co/datasets/simonjegou/ruler).

## Installation

Python 3.10--3.12, Linux, an NVIDIA GPU, and a CUDA 12.8-compatible driver are
required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional ShadowKV CUDA kernels:

```bash
SF_BUILD_SHADOWKV_CUDA=1 python setup.py build_ext --inplace
```

Use a local Hugging Face checkpoint for `--model_path`. For gated models,
authenticate and download the checkpoint first:

```bash
hf auth login
hf download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir experiments/checkpoints/Llama-3.1-8B-Instruct
export MODEL_PATH="$PWD/experiments/checkpoints/Llama-3.1-8B-Instruct"
```

## RULER evaluation

The runner downloads `simonjegou/ruler` automatically. It supports KVPress's
`4096`, `8192`, and `16384` dataset configurations and all 13 tasks:
`niah_single_{1,2,3}`, `niah_multikey_{1,2,3}`, `niah_multivalue`,
`niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1`, and `qa_2`.

Prompts, task-specific generation limits, and string-match metrics follow
KVPress. Sparse execution remains this repository's vLLM backend.

Run a one-example smoke test:

```bash
python -m sparse_frontier.ruler_runner \
  --context_length 4096 \
  --model_path "$MODEL_PATH" \
  --output_dir experiments/results/ruler_4k_smoke \
  --tp 1 \
  --smoke
```

Run the default Dense/Quest/ShadowKV matrix at 16K:

```bash
python -m sparse_frontier.ruler_runner \
  --context_length 16384 \
  --model_path "$MODEL_PATH" \
  --output_dir experiments/results/ruler_16k \
  --max_input_tokens 16896 \
  --tp 1 \
  --seed 43
```

The default matrix is Dense plus Quest and ShadowKV at budgets 96, 128, 256,
512, 1024, and 2048. Run one method/budget with `--method` and `--budget`:

```bash
python -m sparse_frontier.ruler_runner \
  --context_length 8192 \
  --model_path "$MODEL_PATH" \
  --output_dir experiments/results/ruler_8k_quest_1024 \
  --method quest --budget 1024 \
  --max_input_tokens 8704 --tp 1
```

`--max_input_tokens` defaults to `context_length + 512`. Increase it if the
selected model template or task prompt requires more room.

Each method directory contains the canonical `predictions.csv` and
`metrics.json`, plus `dataset.jsonl`, `predictions.jsonl`, `aggregate.json`,
`aggregate.csv`, and `run.json` with sparse runtime telemetry.

### Locked 128K Table-1-compatible protocol

The short-context KVPress Hub export is not used at 128K.  Instead, create a
local JSONL from one pinned RULER/KVPress source revision and a sidecar
protocol manifest.  The manifest records the source repository and 40-hex Git
revision, data-file SHA256, seed, exact ten-task set, model/tokenizer revision,
chat-template SHA256, task output caps, and the hardware policy.  The runner
checks all of these before generation and writes them again to every `run.json`.

The locked task set is `niah_single_{1,2}`, `niah_multikey_{1,2}`,
`niah_multiquery`, `niah_multivalue`, `qa_{1,2}`, `vt`, and `fwe`.  Rows may be
KVPress-shaped (`context`, `question`, `answer_prefix`, `answer`) or generated
RULER rows (`input`, `outputs`); the latter are tokenized directly and never
receive a second chat template.  `runtime.prefill` must be `dense_exact`,
`dtype` must be BF16, decoding must be greedy with batch size one, and the
hardware policy's `max_num_batched_tokens` must fit the whole prefill.

Run the complete four-method matrix as follows:

```bash
python -m sparse_frontier.ruler_runner \
  --protocol ruler_128k_table1_recipe \
  --protocol_manifest experiments/ruler_128k/protocol.json \
  --model_path "$MODEL_PATH" \
  --query_pool_path experiments/query_pools/pile_8b \
  --output_dir experiments/results/ruler_128k_table1_recipe
```

The profile locks Dense, Quest, ShadowKV, and Query-Robust to one run each.
Quest uses `token_budget=2048`, `page_size=16`, and two dense decode layers;
ShadowKV uses `sparse_budget=2048`, `chunk_size=8`, rank 160, 48 outlier
chunks, exact SVD, and the verified PyTorch landmark retrieval path; Query-Robust uses `token_budget=2048`, `chunk_size=16`,
and a horizon equal to the manifest generation cap.  Its pool must be a direct
`pile_empirical_query_pool` captured from an external Pile `train` split;
request-derived RULER capture pools are rejected.

The locked 128K/Table-1 path retains its explicit `triton_fast_retry` pilot
settings for protocol compatibility; its telemetry must be inspected because
that fused mirror-ascent path is not the certified low-query implementation.
For the verified B200 low-query recipe (the current RULER smoke uses four
empirical queries), use `batched_newton`, `solver_chunk_batch_size=1024`,
`solver_max_iterations=8`, `solver_gap_tolerance=5e-4`, and fail-closed mode.
It solves each chunk independently in FP64 dual state, scans the full query
pool for a certificate, and uses bounded support enumeration only for hard
four-query boundary cases.  The B200 16K smoke completed in 47.82 s with
accuracy 1.0, `gap_max=4.98e-4`, zero nonconverged chunks, zero dense
fallbacks, and the same prediction as dense.  These figures are evidence for
that one-sample low-query smoke, not a claim about the full 128K matrix.
The per-example telemetry retains `query_robust_solver_retry_count`,
`query_robust_solver_nonconverged`, and `query_robust_solver_gap_max`; any
nonzero nonconverged count or fail-closed fallback must be reported.

This is a controlled accuracy protocol with Table-1-compatible nominal
recipes, not a reproduction of Table 1's native CPU-offload memory/throughput
measurements.

### Query-Robust

Query-Robust requires an empirical query pool matched to the checkpoint and
context length. Capture a dense subset, finalize it, then run a single budget:

```bash
python -m sparse_frontier.ruler_runner \
  --context_length 16384 \
  --model_path "$MODEL_PATH" \
  --output_dir experiments/results/ruler_16k_capture \
  --method dense \
  --tasks niah_single_1 niah_multikey_1 niah_multiquery vt fwe \
  --task_indices 0 1 2 3 4 \
  --capture_query_dir experiments/query_pools/raw_16k \
  --capture_max_tokens 128 \
  --capture_queries_only \
  --max_input_tokens 16896 --tp 1

python -m sparse_frontier.query_pool_cli capture-schema2-finalize \
  --input experiments/query_pools/raw_16k \
  --output experiments/query_pools/pool_16k \
  --samples_per_head 32 --seed 43

python -m sparse_frontier.ruler_runner \
  --context_length 16384 \
  --model_path "$MODEL_PATH" \
  --output_dir experiments/results/ruler_16k_query_robust_1024 \
  --method query_robust --budget 1024 \
  --query_pool_path experiments/query_pools/pool_16k \
  --query_robust_empirical_query_budget 4 \
  --query_robust_solver_backend batched_newton \
  --query_robust_solver_gap_tolerance 0.0005 \
  --query_robust_solver_max_iterations 8 \
  --query_robust_solver_chunk_batch_size 1024 \
  --query_robust_solver_fail_closed \
  --max_input_tokens 16896 --tp 1
```

The command above is the certified low-query route and requires at most 32
selected empirical queries (four is the validated B200 setting).  If the
full pool is intentionally retained, use the reference `active_set` backend
with its larger iteration budget and inspect the certificate telemetry; do
not silently substitute the fused Triton pilot.

## Verification

```bash
python -m pytest -q
```

## References

- [KVPress](https://github.com/NVIDIA/kvpress)
- [RULER](https://arxiv.org/abs/2404.06654)
- [vLLM](https://github.com/vllm-project/vllm)
