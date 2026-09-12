#!/usr/bin/env bash
set -euo pipefail

# cd /workspace/Sparse-vLLM

MODEL_PATH="/workspace/storage-shared/models/Llama-3.1-8B-Instruct"
DATASET_PATH="/workspace/storage-shared/nlp/tungdd11/tungsparse/ruler-data/ruler-131072.jsonl"
QR_VERTICES="/workspace/Sparse-vLLM/scripts/benchmarks/qr_vertices_m32_128k.pt"
MODEL_FINGERPRINT="d10aef7999a2b5ba950ab3974312feeedbfe0b77"
OUT_ROOT="/workspace/storage-shared/nlp/tungdd11/tungsparse/results/kvpress-ruler-new/full-ruler128k-budget4096-b4"
GPU_ID=0

export SPARSEVLLM_MASTER_PORT=24339

mkdir -p "$OUT_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONPATH="$PWD:$PWD/src" \
python benchmark/kvpress_ruler/evaluate.py \
  --model-path "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --data-dir 131072 \
  --sparse-method query_robust \
  --fraction 1.0 \
  --seed 42 \
  --gpu-memory-utilization 0.90 \
  --batch-size 4 \
  --max-batched-tokens 65536 \
  --query-robust-vertices-path "$QR_VERTICES" \
  --query-robust-model-fingerprint "$MODEL_FINGERPRINT" \
  --query-robust-num-vertices 32 \
  --query-robust-chunk-size 16 \
  --query-robust-solver-iters 24 \
  --query-robust-solver-lr 0.25 \
  --query-robust-score-alpha 1.0 \
  --query-robust-skip-layers 0 \
  --no-query-robust-uniform-p \
  --sink-keep-tokens 0 \
  --decode-keep-tokens 4096 \
  --recent-keep-tokens 0 \
  --output-dir "$OUT_ROOT/query-robust-m32"
