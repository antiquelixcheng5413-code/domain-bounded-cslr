#!/usr/bin/env bash
# D3：VAC + 224 + RGB —— 测「分辨率 160→224 值多少 pp」（队友 A）
#
# 用法
#   bash tools/blank_gov/run_d3_vac224.sh            # 正式跑（55 轮）
#   bash tools/blank_gov/run_d3_vac224.sh smoke      # 冒烟（40 条 / 2 轮，先验证能跑通）
#   RESUME=1 bash tools/blank_gov/run_d3_vac224.sh   # 🔴 从上次 last.pt 续跑（中断后用这个）
#
# 3060 8GB 注意：若 OOM，把 BATCH 改成 1 再跑。
#
# ⚠️ p78 的 --resume 默认是空，**不会按 tag 自动续跑**；
#    中断后必须显式带这个参数，否则从第 1 轮重来。本脚本用 RESUME=1 封装了这个坑。
set -u
cd "$(dirname "$0")/../.." || exit 1

PY=./venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3)

TAG=p157-vac224-rgb
LOG=artifacts/logs_D3_vac224.log
CKPT=artifacts/checkpoints/$TAG-last.pt

if [ "${1:-}" = "smoke" ]; then
  EXTRA="--max-train 40 --max-dev 20 --epochs 2 --total-epochs 55"
  LOG=artifacts/logs_D3_smoke.log
  echo "[D3] 冒烟模式：40 条 / 2 轮"
else
  EXTRA="--epochs 55 --total-epochs 55"
fi

if [ -n "${RESUME:-}" ]; then
  if [ -s "$CKPT" ]; then
    SZ=$(du -m "$CKPT" | cut -f1)
    if [ "$SZ" -lt 100 ]; then
      echo "[D3] 🔴 $CKPT 只有 ${SZ}MB，疑似损坏（正常应 ~545MB）。请改用 best.pt 或从零重跑。"
      exit 1
    fi
    EXTRA="$EXTRA --resume $CKPT"
    echo "[D3] 续跑模式：从 $CKPT 恢复（${SZ}MB）"
  else
    echo "[D3] ⚠️ 找不到 $CKPT，本次按新跑处理"
  fi
fi

mkdir -p artifacts
echo "[D3] tag=$TAG  module=VAC  img=224  log=$LOG"
echo "──── [D3] $(date '+%F %T') 启动 mode=${1:-full} resume=${RESUME:-no} ────" >> "$LOG"

"$PY" -u tools/blank_gov/p78_train_official_tfnet.py \
  --tag "$TAG" \
  --module VAC \
  --img-size 224 \
  --batch "${BATCH:-2}" \
  --mem-budget "${MEM:-5.2e9}" \
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

echo "[D3] 退出码 $rc；日志 $LOG"
if [ "$rc" -ne 0 ]; then
  echo "[D3] 中断了 ⇒ 用下面这条续跑（不会丢已跑的轮次）："
  echo "       RESUME=1 bash tools/blank_gov/run_d3_vac224.sh"
fi
exit $rc
