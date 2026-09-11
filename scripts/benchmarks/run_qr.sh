#!/usr/bin/env bash
set -euo pipefail

# Fair 32K RULER run at compression rate 1/32:
#   effective KV budget = 1024 tokens
#   frozen final calibration = Query-Robust M32

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
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/query-robust-m32}"
QR_VERTICES="${QR_VERTICES:-${REPO_ROOT}/artifacts/query_robust/llama31-8b-pile32k-m32-v1-20260911_200944/qr_vertices_m32.pt}"
MODEL_FINGERPRINT="${MODEL_FINGERPRINT:-d10aef7999a2b5ba950ab3974312feeedbfe0b77}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  printf 'DATASET_PATH does not exist: %s\n' "${DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -f "${QR_VERTICES}" ]]; then
  printf 'QR_VERTICES does not exist: %s\n' "${QR_VERTICES}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src" \
"${PYTHON_BIN}" "${REPO_ROOT}/benchmark/kvpress_ruler/evaluate.py" \
  --model-path "${MODEL_PATH}" \
  --sparse-method query_robust \
  --data-dir 32768 \
  --dataset-path "${DATASET_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --max-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --query-robust-vertices-path "${QR_VERTICES}" \
  --query-robust-model-fingerprint "${MODEL_FINGERPRINT}" \
  --query-robust-num-vertices 32 \
  --query-robust-chunk-size 16 \
  --query-robust-solver-iters 24 \
  --query-robust-solver-lr 0.25 \
  --query-robust-score-alpha 1.0 \
  --query-robust-skip-layers 0 \
  --no-query-robust-uniform-p \
  --sink-keep-tokens 0 \
  --decode-keep-tokens 1024 \
  --recent-keep-tokens 0 \
  --output-dir "${OUT_DIR}"
