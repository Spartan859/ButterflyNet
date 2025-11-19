#!/usr/bin/env bash
set -euo pipefail

# Evaluate the latest augmented-simple checkpoint (or a provided one).
# Usage:
#   bash scripts/eval_aug_simple.sh                # auto-pick latest aug-simple_best_*.pt
#   CHECKPOINT=checkpoints/aug-simple_best_20251119_200000.pt bash scripts/eval_aug_simple.sh
#   bash scripts/eval_aug_simple.sh --no_normalize_cm  # extra args passed to evaluate.py

DATASET_ROOT=${DATASET_ROOT:-ButterflyClassificationDataset}
VAL_CSV=${VAL_CSV:-splits/val.csv}
STATS=${STATS:-src/data/dataset_stats.json}

# Allow overriding checkpoint via env var
CKPT=${CHECKPOINT:-}
if [[ -z "${CKPT}" ]]; then
  CKPT=$(ls -t checkpoints/aug-simple_best_*.pt 2>/dev/null | head -n 1 || true)
fi

if [[ -z "${CKPT}" ]]; then
  echo "[eval_aug_simple] No simple-aug checkpoint found (checkpoints/aug-simple_best_*.pt) and CHECKPOINT not provided." >&2
  exit 1
fi

echo "[eval_aug_simple] Using checkpoint: ${CKPT}"
python evaluate.py \
  --checkpoint "${CKPT}" \
  --dataset_root "${DATASET_ROOT}" \
  --val_csv "${VAL_CSV}" \
  --stats "${STATS}" \
  --out_json analysis/metrics/val_metrics_aug_simple.json \
  --out_report analysis/metrics/classification_report_aug_simple.txt \
  --out_cm analysis/figures/confusion_matrix_aug_simple.png \
  "$@"

echo "[eval_aug_simple] Done. Metrics: analysis/metrics/val_metrics_aug_simple.json; CM: analysis/figures/confusion_matrix_aug_simple.png"
