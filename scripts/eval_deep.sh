#!/usr/bin/env bash
set -euo pipefail

# Evaluate the latest deep-variant checkpoint (or a provided one).
# Usage:
#   bash scripts/eval_deep.sh                # auto-pick latest *deep*_best_*.pt
#   CHECKPOINT=checkpoints/deep_baseline_best_20251119_220000.pt bash scripts/eval_deep.sh
#   bash scripts/eval_deep.sh --no_normalize_cm  # extra args passed to evaluate.py

DATASET_ROOT=${DATASET_ROOT:-ButterflyClassificationDataset}
VAL_CSV=${VAL_CSV:-splits/val.csv}
STATS=${STATS:-src/data/dataset_stats.json}

CKPT=${CHECKPOINT:-}
if [[ -z "${CKPT}" ]]; then
  CKPT=$(ls -t checkpoints/*deep*_best_*.pt 2>/dev/null | head -n 1 || true)
fi

if [[ -z "${CKPT}" ]]; then
  echo "[eval_deep] No deep checkpoint found (checkpoints/*deep*_best_*.pt) and CHECKPOINT not provided." >&2
  exit 1
fi

echo "[eval_deep] Using checkpoint: ${CKPT}"
python evaluate.py \
  --checkpoint "${CKPT}" \
  --dataset_root "${DATASET_ROOT}" \
  --val_csv "${VAL_CSV}" \
  --stats "${STATS}" \
  --model_variant deep \
  --out_json analysis/metrics/val_metrics_deep.json \
  --out_report analysis/metrics/classification_report_deep.txt \
  --out_cm analysis/figures/confusion_matrix_deep.png \
  "$@"

echo "[eval_deep] Done. Metrics: analysis/metrics/val_metrics_deep.json; CM: analysis/figures/confusion_matrix_deep.png"
