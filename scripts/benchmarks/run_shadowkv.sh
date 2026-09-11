#!/usr/bin/env bash
set -euo pipefail

# Fair 32K RULER run at configured sparse compression rate 1/32:
#   ShadowKV sparse budget = 1024 tokens
#   paper knobs: rank 160, chunk size 8
#   runtime profile: GPU cache, 48 outlier chunks, 4 local chunks (32 tokens)
# The ShadowKV auxiliary outlier/local/recent payload is recorded by the
# evaluator; the matched primary sparse budget is --shadowkv-sparse-budget.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to the Llama checkpoint path}"
: "${DATASET_PATH:?Set DATASET_PATH to the processed 32K RULER JSONL}"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/kvpress-ruler/ruler-32768-budget1024}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/shadowkv}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  printf 'DATASET_PATH does not exist: %s\n' "${DATASET_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src" \
"${PYTHON_BIN}" "${REPO_ROOT}/benchmark/kvpress_ruler/evaluate.py" \
  --model-path "${MODEL_PATH}" \
  --sparse-method shadowkv \
  --data-dir 32768 \
  --dataset-path "${DATASET_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --max-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --shadowkv-sparse-budget 1024 \
  --shadowkv-rank 160 \
  --shadowkv-chunk-size 8 \
  --shadowkv-outlier-chunks 48 \
  --shadowkv-local-chunks 4 \
  --shadowkv-recent-tokens 512 \
  --shadowkv-storage gpu_cache \
  --shadowkv-svd-method exact \
  --output-dir "${OUT_DIR}"
