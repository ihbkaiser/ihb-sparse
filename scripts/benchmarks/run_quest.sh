#!/usr/bin/env bash
set -euo pipefail

# Fair 32K RULER run at compression rate 1/32:
#   effective KV budget = 1024 tokens
#   same dataset, batch size, GPU, and scheduler settings as ShadowKV/QR.

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
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/quest}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  printf 'DATASET_PATH does not exist: %s\n' "${DATASET_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src" \
"${PYTHON_BIN}" "${REPO_ROOT}/benchmark/kvpress_ruler/evaluate.py" \
  --model-path "${MODEL_PATH}" \
  --sparse-method quest \
  --data-dir 32768 \
  --dataset-path "${DATASET_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --max-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --quest-chunk-size 16 \
  --sink-keep-tokens 0 \
  --decode-keep-tokens 1024 \
  --recent-keep-tokens 0 \
  --output-dir "${OUT_DIR}"
