#!/usr/bin/env bash
# mT5 mitigation (B): visual<->text alignment aux weight=1.0 on plain mT5 run.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=/mnt/d/part3_models/hf_home
export TRANSFORMERS_CACHE=/mnt/d/part3_models/hf_cache
export HF_ENDPOINT=https://hf-mirror.com

PYTHONPATH=src venv/bin/python -m cslr.translation train-real \
  --config configs/translation_part3_mt5_vaux.yaml \
  --feature-root artifacts/part3_features \
  --train-split train --eval-split validation \
  --epochs 15 --lr 5e-4 --batch-size 16 --eval-limit 0