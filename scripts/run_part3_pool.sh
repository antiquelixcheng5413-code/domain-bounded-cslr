#!/usr/bin/env bash
# Reproducible tri-modal + token-pooling experiment (P3 improvement roadmap #1).
# Run from repo root. Same frozen settings as the tri concat baseline, but with
# SpaMo-style fixed pooling (num_pool_tokens=32) replacing raw concat.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=src venv/bin/python -m cslr.translation train-real \
  --config configs/translation_part3_pool.yaml \
  --feature-root artifacts/part3_features \
  --train-split train --eval-split validation \
  --epochs 15 --lr 5e-4 --batch-size 16 --eval-limit 0