#!/bin/bash
# Learning Rate Sweep for TinyStories 17M model
# Low-resource setting: 40M tokens (batch_size=32, steps=5000, context=256)

set -e

TRAIN_PATH="implementation/output/tinystories_train_tokens.bin"
VAL_PATH="implementation/output/tinystories_valid_tokens.bin"
BASE_DIR="experiments"

LRS=("1e-4" "3e-4" "1e-3" "3e-3" "1e-2" "3e-2" "1e-1" "3e-1" "1")

echo "========================================="
echo "Starting LR sweep: ${LRS[*]}"
echo "========================================="

for LR in "${LRS[@]}"; do
  LOG_DIR="${BASE_DIR}/lr_sweep_${LR}"
  echo ""
  echo "========================================="
  echo "Running LR=${LR} | Logging to ${LOG_DIR}"
  echo "Start time: $(date)"
  echo "========================================="

  CKPT_DIR="checkpoints/lr_sweep_${LR}"
  uv run --no-sync python -m implementation.train --vocab_size 10000 --context_length 256 --d_model 512 --d_ff 1344 --num_layers 4 --num_heads 16 --batch_size 32 --max_iters 5000 --warmup_iters 250 --min_lr 0 --weight_decay 0.1 --beta1 0.9 --beta2 0.999 --eps 1e-8 --grad_clip 1.0 --device cuda --train_path ${TRAIN_PATH} --val_path ${VAL_PATH} --val_interval 100 --log_interval 10 --lr ${LR} --log_dir ${LOG_DIR} --checkpoint_dir ${CKPT_DIR} --checkpoint_interval 2000 --overwrite_logs

  echo "End time: $(date)"
  echo "Finished LR=${LR}"
done

echo ""
echo "========================================="
echo "All sweeps complete!"
echo "Results saved in ${BASE_DIR}/lr_sweep_*"
echo "========================================="
