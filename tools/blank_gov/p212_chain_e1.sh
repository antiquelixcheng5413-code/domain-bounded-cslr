#!/bin/bash
# P212 / E1 自动链：正式训练 → 判定（WER + uWER + 低频桶）→ 落收据
# 铁律：一次只跑一个训练；长任务 python -u；每步留日志；失败立即停。
set -u
cd /home/su127/FYP/domain-bounded-cslr
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
LOG=artifacts/logs_P212_chain.log
exec >> "$LOG" 2>&1
echo "===== E1 chain start $(date) ====="

COMMON="--module TFNet --img-size 160 --batch 2 --mem-budget 7e9 --amp-bf16
  --lr 2e-5 --weight-decay 1e-4 --hidden 1024 --seed 42 --official-pad
  --fix-rgb-order --freeze-backbone --no-resume-sched
  --resume artifacts/checkpoints/p184-init-tfnet160.pt"

echo "[chain] ==== E1 正式 6 轮 $(date) ===="
venv/bin/python -u tools/blank_gov/p78_train_official_tfnet.py \
  --tag p212-e1-csgc $COMMON \
  --csgc-weight 0.3 --csgc-momentum 0.99 --csgc-tau 0.07 --csgc-nneg 16 \
  --epochs 6 --total-epochs 6 \
  > artifacts/logs_P212_e1_train.log 2>&1 || { echo "[chain] ABORT: E1 训练失败"; exit 1; }
grep -E "ep0[0-9][0-9] ctc|最佳 WER_official" artifacts/logs_P212_e1_train.log | tail -8

echo "[chain] ==== uWER 评估（基线 + E1）$(date) ===="
venv/bin/python -u tools/blank_gov/p210_d4_eval.py \
  --ckpt artifacts/checkpoints/p184-tfnet-freeze-best.pt --tag p184-base \
  > artifacts/logs_P212_e1_eval_base.log 2>&1 || { echo "[chain] ABORT: base 评估失败"; exit 1; }
venv/bin/python -u tools/blank_gov/p210_d4_eval.py \
  --ckpt artifacts/checkpoints/p212-e1-csgc-best.pt --tag p212-e1 \
  > artifacts/logs_P212_e1_eval.log 2>&1 || { echo "[chain] ABORT: E1 评估失败"; exit 1; }

echo "[chain] ==== E1 判定 ===="
venv/bin/python tools/blank_gov/p212_judge.py || exit 1
echo "===== E1 chain done $(date) ====="
