# Sparse Frontier

Evaluation framework for training-free sparse attention in transformer LLMs. The repository provides vLLM-based implementations and reproducible RULER evaluation for Dense, Quest, and ShadowKV.

## Requirements

- Linux, Python 3.10–3.12, NVIDIA GPU, and a CUDA 12.8-compatible driver.
- Enough GPU memory for the selected model, context length, and tensor parallelism (`tp`).
- Hugging Face access for gated checkpoints such as Llama and Gemma.

Install the pinned runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional ShadowKV CUDA kernels can be compiled in the same environment:

```bash
SF_BUILD_SHADOWKV_CUDA=1 python setup.py build_ext --inplace
```

The kernels are optional. The PyTorch implementation remains the fallback.

## Models

The RULER runner accepts a local Hugging Face checkpoint through `--model_path`; it does not accept a Hydra model name. Download the checkpoint first, or provide an existing cache directory:

| Model | Hugging Face repository | Default `tp` |
|---|---|---:|
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B-Instruct` | 1 |
| Qwen 2.5 14B | `Qwen/Qwen2.5-14B-Instruct` | 1 |
| Qwen 2.5 32B | `Qwen/Qwen2.5-32B-Instruct` | 2 |
| Qwen 2.5 72B | `Qwen/Qwen2.5-72B-Instruct` | 4 |
| Qwen 3 4B/8B | `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B` | 1 |
| Llama 3.1 8B/70B | `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct` | 1/4 |
| Gemma 3 4B/12B/27B | `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it` | 1/1/2 |

For a gated model, accept its license on Hugging Face and authenticate:

```bash
hf auth login
hf download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir experiments/checkpoints/Llama-3.1-8B-Instruct
```

Use the resulting local directory as `MODEL_PATH`. For ShadowKV, the model must use scalar RoPE positions and its KV-head count must be divisible by `--tp`.

## RULER workflow

The mixed-task pilot covers `niah_single`, `niah_multikey`, `niah_multiquery`, `vt`, and `fwe`. It generates 50 samples per task by default (250 total), with stable indexes, task metadata, gold answers, and prompt-length accounting.

### Prepare data

Obtain the official RULER `PaulGrahamEssays.json` and set its path. The JSON object must contain a string field named `text`.

```bash
export MODEL_PATH="$PWD/experiments/checkpoints/Llama-3.1-8B-Instruct"
export ESSAY_PATH="$PWD/data/PaulGrahamEssays.json"
mkdir -p experiments/data/ruler
```

Generate one immutable dataset per context length. `--max_seq_length` includes the task-specific generation allowance.

```bash
for length in 8192 16384 32768; do
  python -m sparse_frontier.ruler_pilot \
    --model_path "$MODEL_PATH" \
    --essay_path "$ESSAY_PATH" \
    --output "experiments/data/ruler/ruler_${length}.jsonl" \
    --samples_per_task 50 \
    --max_seq_length "$length" \
    --seed 20260831
done
```

The generator refuses to overwrite an existing file. Do not reuse an 8K dataset for a 16K or 32K run.

### Smoke test

The smoke matrix runs Dense, Quest-512, and ShadowKV-512 on one sample each:

```bash
python -m sparse_frontier.ruler_runner \
  --data_path experiments/data/ruler/ruler_8192.jsonl \
  --model_path "$MODEL_PATH" \
  --output_dir experiments/results/ruler_8k_smoke \
  --max_input_tokens 8192 \
  --max_output_tokens 1024 \
  --tp 1 \
  --seed 43 \
  --smoke
```

### Full matrix

Without `--method`, the runner executes seven isolated configurations:

| Method | Budgets |
|---|---|
| Dense | none |
| Quest | 512, 1024, 2048 |
| ShadowKV | 512, 1024, 2048 |

Run the matrix at any generated context length by keeping `--data_path` and `--max_input_tokens` equal:

```bash
python -m sparse_frontier.ruler_runner \
  --data_path experiments/data/ruler/ruler_16384.jsonl \
  --model_path "$MODEL_PATH" \
  --output_dir experiments/results/ruler_16k_full \
  --max_input_tokens 16384 \
  --max_output_tokens 1024 \
  --tp 1 \
  --seed 43

python -m sparse_frontier.ruler_runner \
  --data_path experiments/data/ruler/ruler_32768.jsonl \
  --model_path "$MODEL_PATH" \
  --output_dir experiments/results/ruler_32k_full \
  --max_input_tokens 32768 \
  --max_output_tokens 1024 \
  --tp 1 \
  --seed 43
```

For a single configuration, add `--method dense`, or add `--method quest|shadowkv` and `--budget 512|1024|2048`:

```bash
python -m sparse_frontier.ruler_runner \
  --data_path experiments/data/ruler/ruler_16384.jsonl \
  --model_path "$MODEL_PATH" \
  --output_dir experiments/results/ruler_16k_quest_b1024 \
  --method quest \
  --budget 1024 \
  --max_input_tokens 16384 \
  --max_output_tokens 1024 \
  --tp 1
```

### 32K memory settings

On a 24 GB GPU, increase vLLM capacity or offload model weights when required:

```bash
SF_GPU_MEMORY_UTILIZATION=0.96 \
SF_CPU_OFFLOAD_GB=3 \
python -m sparse_frontier.ruler_runner \
  --data_path experiments/data/ruler/ruler_32768.jsonl \
  --model_path "$MODEL_PATH" \
  --output_dir experiments/results/ruler_32k_full \
  --max_input_tokens 32768 \
  --max_output_tokens 1024 \
  --tp 1 \
  --seed 43
```

`SF_CPU_OFFLOAD_GB` offloads model weights, not the KV cache; vLLM's KV cache remains dense. `SF_MAX_NUM_BATCHED_TOKENS=4096` can reduce dense-prefill activation memory, but ShadowKV requires a non-chunked prefill. `SF_KV_CACHE_MEMORY_BYTES` can cap dense KV-cache allocation when a measured value is available. For Qwen, YaRN RoPE scaling is applied automatically when input plus output length exceeds 32,768 tokens.

### ShadowKV options

ShadowKV uses randomized SVD by default. Use exact full SVD, and optionally disable fused retrieval:

```bash
python -m sparse_frontier.ruler_runner \
  --data_path experiments/data/ruler/ruler_8192.jsonl \
  --model_path "$MODEL_PATH" \
  --output_dir experiments/results/ruler_8k_shadowkv_exact \
  --method shadowkv \
  --budget 2048 \
  --max_input_tokens 8192 \
  --shadowkv_svd_backend exact \
  --no-shadowkv_fused_retrieval
```

Each run writes `dataset.jsonl`, `predictions.jsonl`, `aggregate.json`, `aggregate.csv`, and `run.json`. A nonzero exit code indicates at least one failed example or run.

## Hydra workflow

For the registered attention implementations outside the mixed pilot runner, use Hydra. Attention configs: `dense`, `quest`, `snapkv`, `ada_snapkv`, `flexprefill`, `block_sparse`, `vertical_and_slash`, `tova`, `shadowkv`. RULER task configs: `ruler_niah`, `ruler_vt`, `ruler_cwe`.

```bash
python -m sparse_frontier.main \
  model=llama_8b attention=dense task=ruler_niah \
  max_input_tokens=8192 max_output_tokens=1024 samples=30 tp=1

python -m sparse_frontier.main \
  model=llama_8b attention=quest attention.args.token_budget=1024 \
  task=ruler_vt max_input_tokens=16384 max_output_tokens=1024 \
  samples=30 tp=1
```

Model names and default checkpoint paths are defined in [`sparse_frontier/configs/model`](sparse_frontier/configs/model). Override a path with `model.path=/absolute/path/to/checkpoint`. Global defaults are in [`default.yaml`](sparse_frontier/configs/default.yaml).

## Limitations and development

Evaluation is single-request (`batch_size=1`). Dense attention is exact; Quest sparsifies decoding; ShadowKV keeps dense prefill and a dense vLLM KV cache, so it makes no KV-cache memory-savings claim.

```bash
python -m pytest -q
```

Primary extension points are [`sparse_frontier/modelling/attention`](sparse_frontier/modelling/attention), [`sparse_frontier/tasks`](sparse_frontier/tasks), and their registries.

## References

- [The Sparse Frontier](https://arxiv.org/abs/2504.17768)
- [RULER](https://github.com/NVIDIA/RULER)
- [vLLM](https://github.com/vllm-project/vllm)
