#!/usr/bin/env bash
set -euo pipefail

# Simple training + evaluation entry for augmentation-based training.
# You can override defaults via environment variables or pass extra args to python.
# Example:
#   EPOCHS=50 BATCH_SIZE=48 bash scripts/train_aug.sh --num_workers 8

DATASET_ROOT=${DATASET_ROOT:-ButterflyClassificationDataset}
TRAIN_CSV=${TRAIN_CSV:-splits/train.csv}
VAL_CSV=${VAL_CSV:-splits/val.csv}
STATS=${STATS:-src/data/dataset_stats.json}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
SEED=${SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-}

DEVICE_ARG=()
if [[ -n "$DEVICE" ]]; then
  DEVICE_ARG=(--device "$DEVICE")
fi

echo "[train_aug] Starting training with augmentations..."
python train.py \
  --dataset_root "$DATASET_ROOT" \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --stats "$STATS" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  --aug \
  "${DEVICE_ARG[@]}" \
  "$@"

LATEST_CKPT=$(ls -t checkpoints/aug_best_*.pt 2>/dev/null | head -n 1 || true)
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "[train_aug] No augmented checkpoint found (checkpoints/aug_best_*.pt). Skipping eval."
  exit 0
fi

echo "[train_aug] Evaluating checkpoint: ${LATEST_CKPT}"
python evaluate.py \
  --checkpoint "${LATEST_CKPT}" \
  --dataset_root "$DATASET_ROOT" \
  --val_csv "$VAL_CSV" \
  --stats "$STATS" \
  --out_json analysis/metrics/val_metrics_aug.json \
  --out_report analysis/metrics/classification_report_aug.txt \
  --out_cm analysis/figures/confusion_matrix_aug.png "$@"

echo "[train_aug] Done. Metrics: analysis/metrics/val_metrics_aug.json; CM: analysis/figures/confusion_matrix_aug.png"
