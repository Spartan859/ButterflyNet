#!/usr/bin/env bash
set -euo pipefail

# Evaluate the latest dropout checkpoint (or a provided one).
# Usage:
#   bash scripts/eval_dropout.sh                # auto-pick latest *drop*_best_*.pt
#   CHECKPOINT=checkpoints/baseline_drop0.30_best_20251119_210000.pt bash scripts/eval_dropout.sh
#   bash scripts/eval_dropout.sh --no_normalize_cm  # extra args are passed to evaluate.py

DATASET_ROOT=${DATASET_ROOT:-ButterflyClassificationDataset}
VAL_CSV=${VAL_CSV:-splits/val.csv}
STATS=${STATS:-src/data/dataset_stats.json}

CKPT=${CHECKPOINT:-}
if [[ -z "${CKPT}" ]]; then
  CKPT=$(ls -t checkpoints/*drop*_best_*.pt 2>/dev/null | head -n 1 || true)
fi

if [[ -z "${CKPT}" ]]; then
  echo "[eval_dropout] No dropout checkpoint found (checkpoints/*drop*_best_*.pt) and CHECKPOINT not provided." >&2
  exit 1
fi

echo "[eval_dropout] Using checkpoint: ${CKPT}"
python evaluate.py \
  --checkpoint "${CKPT}" \
  --dataset_root "${DATASET_ROOT}" \
  --val_csv "${VAL_CSV}" \
  --stats "${STATS}" \
  --out_json analysis/metrics/val_metrics_drop.json \
  --out_report analysis/metrics/classification_report_drop.txt \
  --out_cm analysis/figures/confusion_matrix_drop.png \
  "$@"

echo "[eval_dropout] Done. Metrics: analysis/metrics/val_metrics_drop.json; CM: analysis/figures/confusion_matrix_drop.png"
