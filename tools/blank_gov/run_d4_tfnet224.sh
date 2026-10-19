#!/usr/bin/env bash
# D4：TFNet + 224 + RGB —— 测「容量 35M→80M 值多少 pp」（队友 B）
#
# ⚠️ 务必先跑冒烟：TFNet 是 3D 主干（resnet34MAM），首次跑容易踩坑
#   bash tools/blank_gov/run_d4_tfnet224.sh smoke
#   bash tools/blank_gov/run_d4_tfnet224.sh            # 正式 55 轮
set -u
cd "$(dirname "$0")/../.." || exit 1

PY=./venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3)

TAG=p158-tfnet224-rgb
LOG=artifacts/logs_D4_tfnet224.log

if [ "${1:-}" = "smoke" ]; then
  EXTRA="--max-train 40 --max-dev 20 --epochs 2 --total-epochs 55"
  LOG=artifacts/logs_D4_smoke.log
  echo "[D4] 冒烟模式：40 条 / 2 轮（强烈建议先跑这个）"
else
  EXTRA="--epochs 55 --total-epochs 55"
fi

echo "[D4] tag=$TAG  module=TFNet  img=224  batch=1  log=$LOG"
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
  $EXTRA
rc=$?
echo "[D4] 退出码 $rc；日志 $LOG"
exit $rc
