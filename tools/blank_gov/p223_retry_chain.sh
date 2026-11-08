#!/bin/bash
# G1 / P223 断点续跑 + 驱动错误自动重试（2026-11-08 12:50）
# 11:18 首跑在 ep001 后撞 CUDA_ERROR_UNKNOWN；12:19 的 FAIL 判定因此无效（已隔离）。
# 本链从 p223-g1-last.pt 续跑 ep2..ep6，撞驱动错误则新进程重建 context 后重试。
# 所有 attempt 追加到同一训练日志，保证 judge 的 6 轮曲线完整。
set -u
cd /home/su127/FYP/domain-bounded-cslr || exit 1
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=venv/bin/python
LOG=artifacts/logs_P223_g1_retry.log
TRAIN=artifacts/logs_P223_g1_train.log
: > "$LOG"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

COMMON="--module TFNet --img-size 160 --batch 2 --mem-budget 7e9 --amp-bf16
        --lr 2e-5 --weight-decay 1e-4 --hidden 1024 --seed 42
        --official-pad --fix-rgb-order --freeze-backbone --no-resume-sched
        --epochs 6 --total-epochs 6 --tag p223-g1
        --csgc-weight 0.3 --csgc-momentum 0.99 --csgc-tau 0.07 --csgc-nneg 16
        --csgc-confusion artifacts/metrics/blank-gov/p222_confusion_map.json"

OK=0
for i in 1 2 3 4; do
  LAST=artifacts/checkpoints/p223-g1-last.pt
  if [ -f "$LAST" ]; then RESUME="$LAST"; else RESUME="artifacts/checkpoints/p184-init-tfnet160.pt"; fi
  log "===== attempt $i resume=$RESUME ====="
  $PY -u tools/blank_gov/p78_train_official_tfnet.py $COMMON --resume "$RESUME" \
    >> "$TRAIN" 2>&1
  rc=$?
  NEP=$(grep -cE "ep006 ctc" "$TRAIN" || true)
  log "attempt $i rc=$rc ep006_lines=${NEP:-0}"
  if [ "$rc" -eq 0 ] && [ "${NEP:-0}" -ge 1 ]; then
    OK=1
    log "TRAIN COMPLETE (6 epochs)"
    break
  fi
  log "retry in 30s"
  sleep 30
done

if [ "$OK" -ne 1 ]; then
  log "G1 FAILED after 4 attempts -> no evaluation / no verdict"
  exit 1
fi

CK=artifacts/checkpoints/p223-g1-best.pt
[ -f "$CK" ] || { log "no best ckpt -> abort"; exit 1; }

log "===== EVAL p184-base (low_n for R8) ====="
$PY -u tools/blank_gov/night_eval_full.py \
  --ckpt artifacts/checkpoints/p184-tfnet-freeze-best.pt --tag p184-base \
  > artifacts/logs_P223_base_eval.log 2>&1
r1=$?
log "base eval rc=$r1"

log "===== EVAL p223-g1 ====="
$PY -u tools/blank_gov/night_eval_full.py --ckpt "$CK" --tag p223-g1 \
  > artifacts/logs_P223_g1_eval.log 2>&1
r2=$?
log "g1 eval rc=$r2"
[ "$r1" -eq 0 ] && [ "$r2" -eq 0 ] || { log "EVAL failed -> abort"; exit 1; }

log "===== JUDGE R1-R8 ====="
$PY -u tools/blank_gov/night_judge.py --tag p223-g1 --base p184-base \
  --log "$TRAIN" --kind train \
  --title "G1 / P223 真实混淆对 hard negatives（CSGC 负类替换）" \
  > artifacts/logs_P223_g1_judge.log 2>&1
grep -E '"verdict"|R8|"delta_WER_pp"' artifacts/logs_P223_g1_judge.log >> "$LOG"
$PY -u tools/blank_gov/night_report.py --tag p223-g1 >> "$LOG" 2>&1
log "===== G1 retry chain done ====="
