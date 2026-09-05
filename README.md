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
  --max_input_tokens 16896 --tp 1
```

## Verification

```bash
python -m pytest -q
```

## References

- [KVPress](https://github.com/NVIDIA/kvpress)
- [RULER](https://arxiv.org/abs/2404.06654)
- [vLLM](https://github.com/vllm-project/vllm)
