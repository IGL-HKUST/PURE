#!/usr/bin/env bash
# One GPU: bash scripts/train/run_train.sh --config ... --data ... --checkpoint ... --output ...
# DDP:     GPUS=4 bash scripts/train/run_train.sh --config ... --data ... --checkpoint ... --output ...
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
gpu_count="${GPUS:-1}"
if [[ ! "$gpu_count" =~ ^[1-9][0-9]*$ ]]; then
  echo 'GPUS must be a positive integer.' >&2
  exit 2
fi
if [[ "$gpu_count" == 1 ]]; then
  exec "${PYTHON:-python3}" scripts/train/train.py "$@"
else
  exec "${PYTHON:-python3}" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc-per-node="$gpu_count" scripts/train/train.py "$@"
fi
