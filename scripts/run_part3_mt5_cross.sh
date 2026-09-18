#!/usr/bin/env bash
# mT5 mitigation (A): freeze LM path, train visual_proj + cross-attention only.
# Identical frozen settings to run_part3_mt5.sh and run_part3_pool.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=/mnt/d/part3_models/hf_home
export TRANSFORMERS_CACHE=/mnt/d/part3_models/hf_cache
export HF_ENDPOINT=https://hf-mirror.com

PYTHONPATH=src venv/bin/python -m cslr.translation train-real \
  --config configs/translation_part3_mt5_cross.yaml \
  --feature-root artifacts/part3_features \
  --train-split train --eval-split validation \
  --epochs 15 --lr 5e-4 --batch-size 16 --eval-limit 0