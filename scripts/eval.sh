#!/usr/bin/env bash
set -euo pipefail

# Evaluate the latest baseline checkpoint (or a provided one).
# Usage:
#   bash scripts/eval.sh                # auto-pick latest baseline_best_*.pt
#   CHECKPOINT=checkpoints/baseline_best_20251119_135455.pt bash scripts/eval.sh
#   bash scripts/eval.sh --no_normalize_cm  # extra args passed to evaluate.py

DATASET_ROOT=${DATASET_ROOT:-ButterflyClassificationDataset}
VAL_CSV=${VAL_CSV:-splits/val.csv}
STATS=${STATS:-src/data/dataset_stats.json}

# Allow overriding checkpoint via env var
CKPT=${CHECKPOINT:-}
if [[ -z "${CKPT}" ]]; then
  CKPT=$(ls -t checkpoints/baseline_best_*.pt 2>/dev/null | head -n 1 || true)
fi

if [[ -z "${CKPT}" ]]; then
  echo "[eval] No baseline checkpoint found (checkpoints/baseline_best_*.pt) and CHECKPOINT not provided." >&2
  exit 1
fi

echo "[eval] Using checkpoint: ${CKPT}"
python evaluate.py \
  --checkpoint "${CKPT}" \
  --dataset_root "${DATASET_ROOT}" \
  --val_csv "${VAL_CSV}" \
  --stats "${STATS}" \
  --out_json analysis/metrics/val_metrics.json \
  --out_report analysis/metrics/classification_report.txt \
  --out_cm analysis/figures/confusion_matrix.png \
  "$@"

echo "[eval] Done. Metrics: analysis/metrics/val_metrics.json; CM: analysis/figures/confusion_matrix.png"
