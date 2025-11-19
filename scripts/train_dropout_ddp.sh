#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_dropout_ddp.sh 0,1       # 2 GPUs, baseline + dropout
#   bash scripts/train_dropout_ddp.sh 0         # 1 GPU (still via torchrun)
#   DROPOUT_P=0.3 bash scripts/train_dropout_ddp.sh 0,1 --epochs 80 --batch_size 48
#   # You can still combine with augmentation by passing --aug and AUG_MODE env var if needed.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <CUDA_VISIBLE_DEVICES> [--epochs 100 --batch_size 32 ...]"
  exit 1
fi

CUDA_DEVICES="$1"; shift || true
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# Determine processes by device list count
NPROC=$(( $(echo "${CUDA_DEVICES}" | awk -F',' '{print NF}') ))
if [[ -z "${NPROC}" || "${NPROC}" -le 0 ]]; then
  NPROC=1
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Defaults
DATASET_ROOT=${DATASET_ROOT:-ButterflyClassificationDataset}
TRAIN_CSV=${TRAIN_CSV:-splits/train.csv}
VAL_CSV=${VAL_CSV:-splits/val.csv}
STATS=${STATS:-src/data/dataset_stats.json}
DROPOUT_P=${DROPOUT_P:-0.3}
AUG=${AUG:-0}
AUG_MODE=${AUG_MODE:-full}

EXTRA_ARGS=()
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
  --epochs 100 \
  --dropout_p "${DROPOUT_P}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
