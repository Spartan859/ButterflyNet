#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_ddp.sh 0,1,2,3  # use 4 GPUs
#   bash scripts/train_ddp.sh 0,1      # use 2 GPUs
#   bash scripts/train_ddp.sh 0        # use 1 GPU (still via torchrun)
#
# Extra args are forwarded to train.py, e.g.:
#   bash scripts/train_ddp.sh 0,1 --epochs 30 --batch_size 64 --lr 5e-4

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <CUDA_VISIBLE_DEVICES> [--epochs 20 --batch_size 32 ...]"
  exit 1
fi

CUDA_DEVICES="$1"; shift || true
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# Count processes by number of devices
# e.g. "0,1,2,3" -> 4
NPROC=$(( $(echo "$CUDA_DEVICES" | awk -F',' '{print NF}') ))

# Fallback to 1 if parsing failed
if [[ -z "${NPROC}" || "${NPROC}" -le 0 ]]; then
  NPROC=1
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Launching torchrun with nproc_per_node=${NPROC}"

# Default dataset locations (adjust paths as needed)
DATASET_ROOT=${DATASET_ROOT:-ButterflyClassificationDataset}
TRAIN_CSV=${TRAIN_CSV:-splits/train.csv}
VAL_CSV=${VAL_CSV:-splits/val.csv}
STATS=${STATS:-src/data/dataset_stats.json}

# Run torchrun (PyTorch>=1.10)
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
  "$@"
