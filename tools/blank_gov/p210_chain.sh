#!/bin/bash
# P210 自动串联：等 D2 结束 → 判定 → 打 D4 补丁 → 冒烟 → D4 正式 6 轮 → uWER 评估
# 铁律：一次只跑一个训练；长任务 python -u；每步留日志；失败立即停。
set -u
cd /home/su127/FYP/domain-bounded-cslr
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
LOG=artifacts/logs_P210_chain.log
exec >> "$LOG" 2>&1
echo "===== chain start $(date) ====="

D2PID_FILE=artifacts/cache/p208_d2.pid
if [ -f "$D2PID_FILE" ]; then
  D2PID=$(cat "$D2PID_FILE")
  echo "[chain] waiting D2 pid=$D2PID"
  tail --pid="$D2PID" -f /dev/null
fi

# 完成信号必须是 D2 日志里的最终汇总行，避免 PID 复用/中断导致误判
if ! grep -q "最佳 WER_official" artifacts/logs_P208_d2_train.log; then
  echo "[chain] ABORT: D2 未正常完成（日志无最终汇总行），不启动 D4 $(date)"
  exit 1
fi
echo "[chain] D2 完成 $(date)"
grep -E "ep0[0-9][0-9] ctc|最佳 WER_official" artifacts/logs_P208_d2_train.log | tail -8

echo "[chain] ==== D2 判定 ===="
venv/bin/python tools/blank_gov/p210_chain_judge.py d2 || exit 1

# ---- D4 补丁（D2 已结束，此时改 p78 不影响已结束的进程）----
if ! grep -q -- "--cb-alpha" tools/blank_gov/p78_train_official_tfnet.py; then
  cp tools/blank_gov/p78_train_official_tfnet.py /tmp/p78_pre_d4.py
  venv/bin/python tools/blank_gov/p210_patch_d4.py || exit 1
  venv/bin/python -m py_compile tools/blank_gov/p78_train_official_tfnet.py || exit 1
  venv/bin/python tools/blank_gov/p78_train_official_tfnet.py --help 2>&1 \
    | grep -q -- "--cb-alpha" || { echo "[chain] ABORT: --cb-alpha 未生效"; exit 1; }
  echo "[chain] D4 补丁 OK"
fi

COMMON="--module TFNet --img-size 160 --batch 2 --mem-budget 7e9 --amp-bf16
  --lr 2e-5 --weight-decay 1e-4 --hidden 1024 --seed 42 --official-pad
  --fix-rgb-order --freeze-backbone --no-resume-sched
  --resume artifacts/checkpoints/p184-init-tfnet160.pt"

echo "[chain] ==== D4 冒烟 $(date) ===="
venv/bin/python -u tools/blank_gov/p78_train_official_tfnet.py \
  --tag p210-d4-smoke $COMMON --cb-alpha 0.5 \
  --epochs 1 --total-epochs 6 --workers 0 --max-train 32 --max-dev 16 \
  > artifacts/logs_P210_d4_smoke.log 2>&1 || { echo "[chain] ABORT: D4 冒烟失败"; exit 1; }
tail -3 artifacts/logs_P210_d4_smoke.log

echo "[chain] ==== D4 正式 6 轮 $(date) ===="
venv/bin/python -u tools/blank_gov/p78_train_official_tfnet.py \
  --tag p210-d4-cb $COMMON --cb-alpha 0.5 \
  --epochs 6 --total-epochs 6 \
  > artifacts/logs_P210_d4_train.log 2>&1 || { echo "[chain] ABORT: D4 训练失败"; exit 1; }
grep -E "ep0[0-9][0-9] ctc|最佳 WER_official" artifacts/logs_P210_d4_train.log | tail -8

echo "[chain] ==== uWER 评估（基线 + D4）$(date) ===="
venv/bin/python -u tools/blank_gov/p210_d4_eval.py \
  --ckpt artifacts/checkpoints/p184-tfnet-freeze-best.pt --tag p184-base \
  > artifacts/logs_P210_d4_eval_base.log 2>&1 || { echo "[chain] ABORT: base 评估失败"; exit 1; }
venv/bin/python -u tools/blank_gov/p210_d4_eval.py \
  --ckpt artifacts/checkpoints/p210-d4-cb-best.pt --tag p210-d4 \
  > artifacts/logs_P210_d4_eval.log 2>&1 || { echo "[chain] ABORT: D4 评估失败"; exit 1; }

echo "[chain] ==== D4 判定 ===="
venv/bin/python tools/blank_gov/p210_chain_judge.py d4 || exit 1
echo "===== chain done $(date) ====="
