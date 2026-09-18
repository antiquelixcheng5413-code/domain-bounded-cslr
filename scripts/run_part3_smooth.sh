#!/usr/bin/env bash
# Route-2 experiment A: label smoothing (0.1) on tri+pool (beam greedy).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src venv/bin/python -m cslr.translation train-real \
  --config configs/translation_part3_smooth.yaml \
  --feature-root artifacts/part3_features \
  --train-split train --eval-split validation \
  --epochs 15 --lr 5e-4 --batch-size 16 --eval-limit 0