#!/usr/bin/env bash
# P228 / I1 + I0 自动链：I1 零成本 CPU 诊断 → I0 四档 GPU 扫描（断点续 + 最多 3 次重试）
cd /home/su127/FYP/domain-bounded-cslr || exit 1
LOG=artifacts/logs_P228_i0_chain.log
: > "$LOG"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

source venv/bin/activate || exit 1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export I0_RESUME=1

log "===== I1 + I0 chain start ====="
if [ -f artifacts/metrics/blank-gov/p228-i1-correlation.json ]; then
  log "I1 receipt exists -> skip (已在启动前单独跑过)"
else
  log "--- I1: CPU correlation diagnostic ---"
  venv/bin/python -u tools/blank_gov/p228_i1_corr.py \
      >> artifacts/logs_P228_i1.log 2>&1
  rc=$?
  log "I1 rc=$rc"
  if [ "$rc" -ne 0 ]; then
    log "I1 FAILED (解释性诊断失败不阻塞 I0，继续)"
  fi
fi

for i in 1 2 3; do
  log "--- I0 attempt $i ---"
  venv/bin/python -u tools/blank_gov/p228_i0_scan.py \
      >> artifacts/logs_P228_i0.log 2>&1
  rc=$?
  log "I0 attempt $i rc=$rc"
  if [ "$rc" -eq 0 ] && [ -f artifacts/metrics/blank-gov/p228-i0-verdict.txt ]; then
    log "DONE verdict=$(cat artifacts/metrics/blank-gov/p228-i0-verdict.txt)"
    exit 0
  fi
  log "retry after 30s (new process -> new CUDA context)"
  sleep 30
done
log "FAILED after 3 attempts; per-group receipts kept for manual resume"
exit 1
