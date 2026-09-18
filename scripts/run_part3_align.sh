#!/usr/bin/env bash
# Reproducible tri-modal + frame-alignment + pooling experiment (route #2).
# Same frozen settings as pool-only (route #1), but with align_frames=true.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=src venv/bin/python -m cslr.translation train-real \
  --config configs/translation_part3_align.yaml \
  --feature-root artifacts/part3_features \
  --train-split train --eval-split validation \
  --epochs 15 --lr 5e-4 --batch-size 16 --eval-limit 0