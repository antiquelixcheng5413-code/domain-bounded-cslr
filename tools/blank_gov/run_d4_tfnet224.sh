#!/usr/bin/env bash
# D4：TFNet + 224 + RGB —— 测「容量 35M→80M 值多少 pp」（队友 B）
#
# ⚠️ 务必先跑冒烟：TFNet 是 3D 主干（resnet34MAM），首次跑容易踩坑
#
# 用法
#   bash tools/blank_gov/run_d4_tfnet224.sh smoke      # 必须做（40 条 / 2 轮）
#   bash tools/blank_gov/run_d4_tfnet224.sh            # 正式 55 轮
#   RESUME=1 bash tools/blank_gov/run_d4_tfnet224.sh   # 🔴 从上次 last.pt 续跑
#
# ⚠️ p78 的 --resume 默认是空，**不会按 tag 自动续跑**；
#    这个任务要跑 50h 上下，几乎一定会被打断 ⇒ 中断后务必用 RESUME=1，否则从头再来。
set -u
cd "$(dirname "$0")/../.." || exit 1

PY=./venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3)

TAG=p158-tfnet224-rgb
LOG=artifacts/logs_D4_tfnet224.log
CKPT=artifacts/checkpoints/$TAG-last.pt

if [ "${1:-}" = "smoke" ]; then
  EXTRA="--max-train 40 --max-dev 20 --epochs 2 --total-epochs 55"
  LOG=artifacts/logs_D4_smoke.log
  echo "[D4] 冒烟模式：40 条 / 2 轮（强烈建议先跑这个）"
else
  EXTRA="--epochs 55 --total-epochs 55"
fi

if [ -n "${RESUME:-}" ]; then
  if [ -s "$CKPT" ]; then
    SZ=$(du -m "$CKPT" | cut -f1)
    if [ "$SZ" -lt 100 ]; then
      echo "[D4] 🔴 $CKPT 只有 ${SZ}MB，疑似损坏。请改用 best.pt 或从零重跑。"
      exit 1
    fi
    EXTRA="$EXTRA --resume $CKPT"
    echo "[D4] 续跑模式：从 $CKPT 恢复（${SZ}MB）"
  else
    echo "[D4] ⚠️ 找不到 $CKPT，本次按新跑处理"
  fi
fi

mkdir -p artifacts
echo "[D4] tag=$TAG  module=TFNet  img=224  batch=${BATCH:-1}  log=$LOG"
echo "──── [D4] $(date '+%F %T') 启动 mode=${1:-full} resume=${RESUME:-no} ────" >> "$LOG"

"$PY" -u tools/blank_gov/p78_train_official_tfnet.py \
  --tag "$TAG" \
  --module TFNet \
  --img-size 224 \
  --batch "${BATCH:-1}" \
  --mem-budget "${MEM:-6.5e9}" \
  --amp-bf16 \
  --lr 1e-4 --weight-decay 1e-4 --hidden 1024 \
  --seed 42 \
  --official-pad \
  --word-head 0.3 --word-boundary ctc_align \
  --warmup-ep 3 \
  --workers 4 \
  --fix-rgb-order \
  $EXTRA 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

echo "[D4] 退出码 $rc；日志 $LOG"
if [ "$rc" -ne 0 ]; then
  echo "[D4] 中断了 ⇒ 用下面这条续跑（不会丢已跑的轮次）："
  echo "       RESUME=1 bash tools/blank_gov/run_d4_tfnet224.sh"
fi
exit $rc
