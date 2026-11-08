#!/bin/bash
# G1 / P223 自动链：6 轮训练 -> 重算基线 -> 全指标评估 -> R1-R8 判定 -> 阶段报告
# 判据见 docs/planning/G1-PREREG-2026-11-08.md（训练启动前写死）
set -u
cd /home/su127/FYP/domain-bounded-cslr || exit 1
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
PY=venv/bin/python
LOG=artifacts/logs_P223_g1_chain.log
: > "$LOG"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

COMMON="--module TFNet --img-size 160 --batch 2 --mem-budget 7e9 --amp-bf16
        --lr 2e-5 --weight-decay 1e-4 --hidden 1024 --seed 42
        --official-pad --fix-rgb-order --freeze-backbone --no-resume-sched
        --epochs 6 --total-epochs 6
        --resume artifacts/checkpoints/p184-init-tfnet160.pt"

log "===== G1 chain start ====="
log "===== TRAIN p223-g1 (6 epochs) ====="
$PY -u tools/blank_gov/p78_train_official_tfnet.py \
  $COMMON --tag p223-g1 \
  --csgc-weight 0.3 --csgc-momentum 0.99 --csgc-tau 0.07 --csgc-nneg 16 \
  --csgc-confusion artifacts/metrics/blank-gov/p222_confusion_map.json \
  > artifacts/logs_P223_g1_train.log 2>&1
rc=$?
log "TRAIN done rc=$rc"
grep -E "ep0[0-9][0-9] ctc" artifacts/logs_P223_g1_train.log | tail -8 >> "$LOG"
# 🔴 12:19 教训：训练崩溃(rc!=0)或未跑满 6 轮时，绝不能拿半成品 ckpt 出判定
NEP=$(grep -cE "ep006 ctc" artifacts/logs_P223_g1_train.log || true)
if [ "$rc" -ne 0 ] || [ "${NEP:-0}" -lt 1 ]; then
  log "TRAIN INCOMPLETE (rc=$rc, ep006 lines=${NEP:-0}) -> ABORT, no eval / no verdict"
  exit 1
fi

CK=artifacts/checkpoints/p223-g1-best.pt
if [ ! -f "$CK" ]; then
  log "NO best ckpt -> abort evaluation"
  exit 1
fi

log "===== EVAL p184-base (new evaluator, low_n for R8) ====="
$PY -u tools/blank_gov/night_eval_full.py \
  --ckpt artifacts/checkpoints/p184-tfnet-freeze-best.pt --tag p184-base \
  > artifacts/logs_P223_base_eval.log 2>&1
rc1=$?
log "base eval rc=$rc1"

log "===== EVAL p223-g1 ====="
$PY -u tools/blank_gov/night_eval_full.py --ckpt "$CK" --tag p223-g1 \
  > artifacts/logs_P223_g1_eval.log 2>&1
rc2=$?
log "g1 eval rc=$rc2"

if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ]; then
  log "EVAL failed -> abort judge"
  exit 1
fi

log "===== JUDGE R1-R8 ====="
$PY -u tools/blank_gov/night_judge.py --tag p223-g1 --base p184-base \
  --log artifacts/logs_P223_g1_train.log --kind train \
  --title "G1 / P223 真实混淆对 hard negatives（CSGC 负类替换）" \
  > artifacts/logs_P223_g1_judge.log 2>&1
grep -E '"verdict"|"delta_WER_pp"|"ci95"|R8' artifacts/logs_P223_g1_judge.log >> "$LOG"
VD=$(grep -m1 '"verdict"' artifacts/metrics/blank-gov/p223-g1-judge.json | tr -d ' ,"')
log "VERDICT $VD"

$PY -u tools/blank_gov/night_report.py --tag p223-g1 >> "$LOG" 2>&1
log "===== G1 chain done ====="
