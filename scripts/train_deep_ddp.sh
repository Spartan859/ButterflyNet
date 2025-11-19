#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_deep_ddp.sh 0,1         # 2 GPUs, deep model (no aug by default)
#   AUG=1 AUG_MODE=simple bash scripts/train_deep_ddp.sh 0 --epochs 80 --batch_size 48
#   DROPOUT_P=0.2 bash scripts/train_deep_ddp.sh 0,1 --epochs 100
# Notes:
#   - Uses train.py with --model_variant deep
#   - Combine with augmentation and/or dropout as needed via env vars and args

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <CUDA_VISIBLE_DEVICES> [--epochs 100 --batch_size 32 ...]"
  exit 1
fi

CUDA_DEVICES="$1"; shift || true
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# Determine number of processes
NPROC=$(( $(echo "${CUDA_DEVICES}" | awk -F',' '{print NF}') ))
if [[ -z "${NPROC}" || "${NPROC}" -le 0 ]]; then
  NPROC=1
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Defaults (override via env as needed)
DATASET_ROOT=${DATASET_ROOT:-ButterflyClassificationDataset}
TRAIN_CSV=${TRAIN_CSV:-splits/train.csv}
VAL_CSV=${VAL_CSV:-splits/val.csv}
STATS=${STATS:-src/data/dataset_stats.json}
DROPOUT_P=${DROPOUT_P:-0.0}
AUG=${AUG:-0}
AUG_MODE=${AUG_MODE:-full}

EXTRA_ARGS=(--model_variant deep)
if [[ "${AUG}" == "1" || "${AUG}" == "true" ]]; then
  EXTRA_ARGS+=(--aug --aug_mode "${AUG_MODE}")
fi

exec python -m torch.distributed.run \
  --nproc_per_node "${NPROC}" \
  train.py \
  --distributed \
  --device cuda \
  --dataset_root "${DATASET_ROOT}" \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --stats "${STATS}" \
  --epochs 50 \
  --dropout_p "${DROPOUT_P}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
