#!/usr/bin/env bash
# P170 — dropout0.3 正则探针（VAC@160 RGB，单变量对照 P150）
# 与 P150 唯一差异：--dropout 0.3。从零 55ep，multistep[35,45]，不开 SWA/余弦。
set -u
cd "$(dirname "$0")/../.." || exit 1

TAG=p170-dropout03-rgb
LOG="artifacts/logs_P170_vac160_dropout.log"

./venv/bin/python -u tools/blank_gov/p78_train_official_tfnet.py \
  --tag "$TAG" \
  --module VAC \
  --img-size 160 \
  --batch 2 \
  --mem-budget 5.2e9 \
  --amp-bf16 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --hidden 1024 \
  --seed 42 \
  --official-pad \
  --word-head 0.3 \
  --word-boundary ctc_align \
  --warmup-ep 3 \
  --fix-rgb-order \
  --dropout 0.3 \
  --epochs 55 \
  --total-epochs 55 \
  2>&1 | tee "$LOG"

echo "EXIT=$?"
echo "=== ckpt 大小校验（<1MB 视为坏）==="
ls -l "artifacts/checkpoints/${TAG}-best.pt" "artifacts/checkpoints/${TAG}-last.pt" 2>/dev/null
