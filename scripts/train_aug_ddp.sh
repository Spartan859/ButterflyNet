#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_aug_ddp.sh 0,1,2,3  # 4 GPUs with augmentations
#   bash scripts/train_aug_ddp.sh 0,1      # 2 GPUs with augmentations
#   bash scripts/train_aug_ddp.sh 0        # 1 GPU (still via torchrun)
#
# Extra args are forwarded to train.py, e.g.:
#   bash scripts/train_aug_ddp.sh 0,1 --epochs 60 --batch_size 48 --lr 5e-4

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <CUDA_VISIBLE_DEVICES> [--epochs 100 --batch_size 32 ...]"
  exit 1
fi

CUDA_DEVICES="$1"; shift || true
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# Determine number of processes from device list (e.g. "0,1,2" -> 3)
NPROC=$(( $(echo "${CUDA_DEVICES}" | awk -F',' '{print NF}') ))
if [[ -z "${NPROC}" || "${NPROC}" -le 0 ]]; then
  NPROC=1
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
printenv | grep -E '^(DATASET_ROOT|TRAIN_CSV|VAL_CSV|STATS|EPOCHS|BATCH_SIZE|SEED|NUM_WORKERS)=' || true

echo "Launching torchrun with nproc_per_node=${NPROC} (aug enabled)"

# Augmentation mode (simple or full). Default simple per new requirement.
AUG_MODE=${AUG_MODE:-simple}
echo "Augmentation mode: ${AUG_MODE}" 

# Defaults (override via env if needed)
DATASET_ROOT=${DATASET_ROOT:-ButterflyClassificationDataset}
TRAIN_CSV=${TRAIN_CSV:-splits/train.csv}
VAL_CSV=${VAL_CSV:-splits/val.csv}
STATS=${STATS:-src/data/dataset_stats.json}

# Run with torch.distributed.run to match existing train_ddp.sh style
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
  --aug \
  --aug_mode "${AUG_MODE}" \
  "$@"
