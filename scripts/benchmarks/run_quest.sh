#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Sparse-vLLM

MODEL_PATH="/workspace/storage-shared/models/Llama-3.1-8B-Instruct"
DATASET_PATH="/workspace/storage-shared/nlp/tungdd11/tungsparse/ruler-data/ruler-32768.jsonl"
OUT_ROOT="/workspace/storage-shared/nlp/tungdd11/tungsparse/results/kvpress-ruler/full-ruler32k-budget1024-b2"
GPU_ID=0

mkdir -p "$OUT_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONPATH="$PWD:$PWD/src" \
python benchmark/kvpress_ruler/evaluate.py \
  --model-path "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --data-dir 32768 \
  --sparse-method quest \
  --fraction 1.0 \
  --seed 42 \
  --gpu-memory-utilization 0.85 \
  --batch-size 32 \
  --max-batched-tokens 65536 \
  --quest-chunk-size 16 \
  --sink-keep-tokens 0 \
  --decode-keep-tokens 1024 \
  --recent-keep-tokens 0 \
  --output-dir "$OUT_ROOT/quest"
