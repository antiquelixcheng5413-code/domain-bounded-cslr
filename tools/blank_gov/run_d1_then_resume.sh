#!/usr/bin/env bash
# 顺序执行：D1（官方权重消融，~16min）→ 自动续跑 P150（从 last.pt 恢复）
# 一次只跑一个 GPU 任务，中间不留空窗。
set -u
cd /home/su127/FYP/domain-bounded-cslr || exit 1
PY=./venv/bin/python

echo "=== [1/2] D1 官方权重消融（fast：160+RGB / 224+BGR / 160+BGR）==="
date "+  开始 %m-%d %H:%M:%S"
"$PY" -u tools/blank_gov/p156_official_ablation.py 0 fast \
  > artifacts/logs_D1_official_ablation.log 2>&1
echo "   D1 退出码 $? （$(date '+%H:%M:%S')）"

echo "=== [2/2] 续跑 P150（resume last.pt）==="
date "+  开始 %m-%d %H:%M:%S"
"$PY" -u tools/blank_gov/p78_train_official_tfnet.py \
  --tag p150-rgbfix \
  --module VAC \
  --img-size 160 \
  --batch 2 \
  --mem-budget 5.2e9 \
  --amp-bf16 \
  --epochs 55 --total-epochs 55 \
  --lr 1e-4 --weight-decay 1e-4 --hidden 1024 \
  --seed 42 \
  --official-pad \
  --word-head 0.3 --word-boundary ctc_align \
  --warmup-ep 3 \
  --workers 4 \
  --fix-rgb-order \
  --resume artifacts/checkpoints/p150-rgbfix-last.pt \
  > artifacts/logs_P150_resume.log 2>&1
echo "   P150 退出码 $? （$(date '+%H:%M:%S')）"
