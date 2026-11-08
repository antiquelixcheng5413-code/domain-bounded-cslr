#!/bin/bash
# H1 自动链：train 记忆库 -> dev 段 -> kNN 后验精修 λ 扫描 -> R1-R8 判定 -> 阶段报告
# 判据见 docs/planning/H1-PREREG-2026-11-08.md（启动前写死）
set -u
cd /home/su127/FYP/domain-bounded-cslr || exit 1
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=venv/bin/python
LOG=artifacts/logs_P224_h1_chain.log
: > "$LOG"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

dump() {
  local sp="$1"
  for i in 1 2 3; do
    log "===== dump $sp attempt $i ====="
    H1_RESUME=1 $PY -u tools/blank_gov/p224_h1_dump.py --split "$sp" \
      >> "artifacts/logs_P224_h1_${sp}.log" 2>&1
    rc=$?
    log "dump $sp rc=$rc"
    if [ "$rc" -eq 0 ] && [ -f "artifacts/cache/p224_${sp}_segments.pkl" ]; then
      return 0
    fi
    sleep 20
  done
  return 1
}

log "===== H1 chain start ====="
dump train || { log "train dump FAILED -> abort"; exit 1; }
dump dev   || { log "dev dump FAILED -> abort"; exit 1; }

log "===== kNN posterior refinement + lambda scan ====="
$PY -u tools/blank_gov/p226_h1_knn.py > artifacts/logs_P226_h1_knn.log 2>&1
rc=$?
log "knn rc=$rc"
grep -E "^lam=|chosen" artifacts/logs_P226_h1_knn.log >> "$LOG"
if [ "$rc" -ne 0 ] || [ ! -f artifacts/metrics/blank-gov/p226-h1-uwer.json ] \
   || [ ! -f artifacts/metrics/blank-gov/p226-h1-base-uwer.json ]; then
  log "knn FAILED -> abort"; exit 1
fi

log "===== JUDGE R1-R8 ====="
$PY -u tools/blank_gov/night_judge.py --tag p226-h1 --base p226-h1-base \
  --kind decode --title "H1 / P224-P226 检索增强 kNN 后验精修（RAC 分布插值）" \
  > artifacts/logs_P226_h1_judge.log 2>&1
jrc=$?
log "judge rc=$jrc"
grep -E '"verdict"|R8|"delta_WER_pp"' artifacts/logs_P226_h1_judge.log >> "$LOG"
if [ "$jrc" -ne 0 ] || [ ! -f artifacts/metrics/blank-gov/p226-h1-judge.json ]; then
  log "JUDGE FAILED -> abort (no verdict)"; exit 1
fi
$PY -u tools/blank_gov/night_report.py --tag p226-h1 >> "$LOG" 2>&1
rrc=$?
if [ "$rrc" -ne 0 ] || [ ! -f docs/experiments/stage-p226-h1.md ]; then
  log "REPORT FAILED -> abort"; exit 1
fi
log "===== H1 chain done ====="
