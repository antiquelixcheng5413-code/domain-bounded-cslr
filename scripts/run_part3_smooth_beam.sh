#!/usr/bin/env bash
# Route-2 experiment B: label smoothing (0.1) + beam search (beam=4) on tri+pool.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src venv/bin/python -m cslr.translation train-real \
  --config configs/translation_part3_smooth_beam.yaml \
  --feature-root artifacts/part3_features \
  --train-split train --eval-split validation \
  --epochs 15 --lr 5e-4 --batch-size 16 --eval-limit 0