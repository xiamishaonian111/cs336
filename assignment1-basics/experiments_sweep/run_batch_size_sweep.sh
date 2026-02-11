#!/bin/bash
# Batch Size Sweep for TinyStories 17M model
# Fixed total tokens (~40M), varying batch size from 1 to GPU memory limit
# Steps are computed per batch size to keep total tokens constant

set -e

TRAIN_PATH="implementation/output/tinystories_train_tokens.bin"
VAL_PATH="implementation/output/tinystories_valid_tokens.bin"
BASE_DIR="experiments"

TOTAL_TOKENS=40960000
CONTEXT_LENGTH=256
LR="3e-3"

BATCH_SIZES=(1 2 4 8 16 32 64)

echo "========================================="
echo "Starting batch size sweep: ${BATCH_SIZES[*]}"
echo "Total tokens per run: ${TOTAL_TOKENS}"
echo "========================================="

for BS in "${BATCH_SIZES[@]}"; do
  MAX_ITERS=$((TOTAL_TOKENS / (BS * CONTEXT_LENGTH)))
  WARMUP=$((MAX_ITERS / 20))
  LOG_DIR="${BASE_DIR}/batch_size_${BS}"
  CKPT_DIR="checkpoints/batch_size_${BS}"

  echo ""
  echo "========================================="
  echo "Running BS=${BS} | Steps=${MAX_ITERS} | Warmup=${WARMUP}"
  echo "Start time: $(date)"
  echo "========================================="

  uv run --no-sync python -m implementation.train --vocab_size 10000 --context_length 256 --d_model 512 --d_ff 1344 --num_layers 4 --num_heads 16 --batch_size ${BS} --max_iters ${MAX_ITERS} --warmup_iters ${WARMUP} --min_lr 0 --weight_decay 0.1 --beta1 0.9 --beta2 0.999 --eps 1e-8 --grad_clip 1.0 --device cuda --train_path ${TRAIN_PATH} --val_path ${VAL_PATH} --val_interval 100 --log_interval 10 --lr ${LR} --log_dir ${LOG_DIR} --checkpoint_dir ${CKPT_DIR} --checkpoint_interval 2000 --overwrite_logs || echo "FAILED: BS=${BS} (likely OOM)"

  echo "End time: $(date)"
  echo "Finished BS=${BS}"
done

echo ""
echo "========================================="
echo "All sweeps complete!"
echo "Results saved in ${BASE_DIR}/batch_size_*"
echo "========================================="
