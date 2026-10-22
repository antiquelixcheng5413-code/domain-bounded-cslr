#!/usr/bin/env bash
# P167：词级辅助头消融 —— VAC + 160 + RGB，--word-head 0
#
# 动机：--word-head 0.3 是**我们**加的，官方源码里没有这个词级辅助头。
#       它以 30% 权重与 CTC 抢梯度，可能是峰锐度偏低（0.6242 vs 官方 0.8641）的一个来源。
#       这是"我们的配置 vs 官方配置"之间**最后一个没验证过的偏离**。
#
# 对照：P150（VAC+160+RGB，word-head 0.3，best 53.73 @ ep49）
# 判据：与 P150 做逐轮配对（只用 ep32+），改善需 ≥ 1.556pp 才可判
#
# 用法
#   bash tools/blank_gov/run_p167_wh0.sh            # 正式跑（55 轮，约 26h）
#   bash tools/blank_gov/run_p167_wh0.sh smoke      # 冒烟（40 条 / 2 轮）
#   RESUME=1 bash tools/blank_gov/run_p167_wh0.sh   # 🔴 从上次 last.pt 续跑
#
# ⚠️ p78 的 --resume 默认为空、不会按 tag 自动续跑 ⇒ 中断后必须用 RESUME=1，
#    否则从第 1 轮重来。
set -u
cd "$(dirname "$0")/../.." || exit 1

PY=./venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3)

TAG=p167-vac160-rgb-wh0
LOG=artifacts/logs_P167_wh0.log
CKPT=artifacts/checkpoints/$TAG-last.pt

if [ "${1:-}" = "smoke" ]; then
  EXTRA="--max-train 40 --max-dev 20 --epochs 2 --total-epochs 55"
  LOG=artifacts/logs_P167_smoke.log
  # ★ 冒烟必须用独立 tag：否则会写坏正式 ckpt，
  #   正式跑若在 ep001 前崩掉，RESUME=1 会静默续到这个只训过 40 条的模型上
  TAG=${TAG}-smoke
  CKPT=artifacts/checkpoints/$TAG-last.pt
  echo "[P167] 冒烟模式：40 条 / 2 轮（tag=$TAG）"
else
  EXTRA="--epochs 55 --total-epochs 55"
fi

if [ -n "${RESUME:-}" ]; then
  if [ -s "$CKPT" ]; then
    SZ=$(du -m "$CKPT" | cut -f1)
    if [ "$SZ" -lt 100 ]; then
      echo "[P167] 🔴 $CKPT 只有 ${SZ}MB，疑似残file（正常应 ~520MB）。拒绝续跑。"
      exit 1
    fi
    EXTRA="$EXTRA --resume $CKPT"
    echo "[P167] 续跑模式：从 $CKPT 恢复（${SZ}MB）"
  else
    echo "[P167] ⚠️ 找不到 $CKPT，本次按新跑处理"
  fi
fi

mkdir -p artifacts
echo "[P167] tag=$TAG  module=VAC  img=160  word-head=0  log=$LOG"
echo "──── [P167] $(date '+%F %T') 启动 mode=${1:-full} resume=${RESUME:-no} ────" >> "$LOG"

"$PY" -u tools/blank_gov/p78_train_official_tfnet.py \
  --tag "$TAG" \
  --module VAC \
  --img-size 160 \
  --batch "${BATCH:-2}" \
  --mem-budget "${MEM:-5.2e9}" \
  --amp-bf16 \
  --lr 1e-4 --weight-decay 1e-4 --hidden 1024 \
  --seed 42 \
  --official-pad \
  --word-head 0 --word-boundary ctc_align \
  --warmup-ep 3 \
  --workers 4 \
  --fix-rgb-order \
  $EXTRA 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

echo "[P167] 退出码 $rc；日志 $LOG"
if [ "$rc" -ne 0 ]; then
  echo "[P167] 中断了 ⇒ 用下面这条续跑（不会丢已跑的轮次）："
  echo "       RESUME=1 bash tools/blank_gov/run_p167_wh0.sh"
fi
exit $rc
