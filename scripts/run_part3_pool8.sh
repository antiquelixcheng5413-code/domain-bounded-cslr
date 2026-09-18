#!/usr/bin/env bash
# Token-density sweep down K=8 (tri + pooling). Frozen settings identical to
# run_part3_pool.sh, only num_pool_tokens=8.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

PYTHONPATH=src venv/bin/python -m cslr.translation train-real \
  --config configs/translation_part3_pool8.yaml \
  --feature-root artifacts/part3_features \
  --train-split train --eval-split validation \
  --epochs 15 --lr 5e-4 --batch-size 16 --eval-limit 0