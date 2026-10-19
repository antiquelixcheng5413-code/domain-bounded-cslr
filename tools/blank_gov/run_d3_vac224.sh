#!/usr/bin/env bash
# D3：VAC + 224 + RGB —— 测「分辨率 160→224 值多少 pp」（队友 A）
#
# 用法
#   bash tools/blank_gov/run_d3_vac224.sh            # 正式跑（55 轮）
#   bash tools/blank_gov/run_d3_vac224.sh smoke      # 冒烟（40 条 / 2 轮，先验证能跑通）
#
# 3060 8GB 注意：若 OOM，把 BATCH 改成 1 再跑。
set -u
cd "$(dirname "$0")/../.." || exit 1

PY=./venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3)

TAG=p157-vac224-rgb
LOG=artifacts/logs_D3_vac224.log

if [ "${1:-}" = "smoke" ]; then
  EXTRA="--max-train 40 --max-dev 20 --epochs 2 --total-epochs 55"
  LOG=artifacts/logs_D3_smoke.log
  echo "[D3] 冒烟模式：40 条 / 2 轮"
else
  EXTRA="--epochs 55 --total-epochs 55"
fi

echo "[D3] tag=$TAG  module=VAC  img=224  log=$LOG"
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
  $EXTRA
rc=$?
echo "[D3] 退出码 $rc；日志 $LOG"
exit $rc
