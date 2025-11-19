#!/usr/bin/env bash
set -euo pipefail

# Evaluate the latest augmented checkpoint (or a provided one).
# Usage:
#   bash scripts/eval_aug.sh                # auto-pick latest aug_best_*.pt
#   CHECKPOINT=checkpoints/aug_best_20251119_135455.pt bash scripts/eval_aug.sh
#   bash scripts/eval_aug.sh --no_normalize_cm  # extra args passed to evaluate.py

DATASET_ROOT=${DATASET_ROOT:-ButterflyClassificationDataset}
VAL_CSV=${VAL_CSV:-splits/val.csv}
STATS=${STATS:-src/data/dataset_stats.json}

# Allow overriding checkpoint via env var
CKPT=${CHECKPOINT:-}
if [[ -z "${CKPT}" ]]; then
  CKPT=$(ls -t checkpoints/aug_best_*.pt 2>/dev/null | head -n 1 || true)
fi

if [[ -z "${CKPT}" ]]; then
  echo "[eval_aug] No augmented checkpoint found (checkpoints/aug_best_*.pt) and CHECKPOINT not provided." >&2
  exit 1
fi

echo "[eval_aug] Using checkpoint: ${CKPT}"
python evaluate.py \
  --checkpoint "${CKPT}" \
  --dataset_root "${DATASET_ROOT}" \
  --val_csv "${VAL_CSV}" \
  --stats "${STATS}" \
  --out_json analysis/metrics/val_metrics_aug.json \
  --out_report analysis/metrics/classification_report_aug.txt \
  --out_cm analysis/figures/confusion_matrix_aug.png \
  "$@"

echo "[eval_aug] Done. Metrics: analysis/metrics/val_metrics_aug.json; CM: analysis/figures/confusion_matrix_aug.png"
