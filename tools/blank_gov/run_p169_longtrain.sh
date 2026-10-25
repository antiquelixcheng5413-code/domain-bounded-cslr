#!/usr/bin/env bash
# P169：VAC@160 + RGB，从 P150 best warm-start 长训练 + 余弦退火 + SWA
# 用法：bash tools/blank_gov/run_p169_longtrain.sh   （建议在后台跑，约 21h）
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

RESUME_CK="artifacts/checkpoints/p150-rgbfix-best.pt"
TAG="p169-longtrain-rgb"
EPOCHS=100
SWA_START=70
LOG="artifacts/logs_P169_vac160_longtrain.log"

# 🔴 ckpt 大小校验：<1MB 视为坏（CUDA sticky error 曾写出残 file）
if [ ! -s "$RESUME_CK" ] || [ "$(stat -c%s "$RESUME_CK" 2>/dev/null)" -lt 1048576 ]; then
  echo "❌ resume ckpt 不存在或过小：$RESUME_CK" >&2
  exit 1
fi

CMD="./venv/bin/python -u tools/blank_gov/p78_train_official_tfnet.py \
  --tag $TAG --module VAC --img-size 160 --batch 2 --mem-budget 5.2e9 \
  --amp-bf16 --lr 1e-4 --weight-decay 1e-4 --hidden 1024 --seed 42 \
  --official-pad --word-head 0.3 --word-boundary ctc_align --warmup-ep 3 \
  --fix-rgb-order --epochs $EPOCHS --total-epochs $EPOCHS \
  --sched cosine --eta-min 1e-6 --no-resume-sched --swa-start $SWA_START \
  --resume $RESUME_CK"

echo "启动 P169: $CMD"
echo "日志: $LOG"
# 🔴 用 tee -a 实时滚动；看日志滚动而非只看 GPU（铁律 #12）
$CMD 2>&1 | tee -a "$LOG"
echo "EXIT=$?"
