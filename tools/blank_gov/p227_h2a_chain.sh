#!/usr/bin/env bash
# P227 / H2-a 自动重试链：撞 WSL 驱动错误时从每组收据续跑，最多 3 次。
cd /home/su127/FYP/domain-bounded-cslr || exit 1
LOG=artifacts/logs_P227_h2a_chain.log
: > "$LOG"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

source venv/bin/activate || exit 1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export H2A_RESUME=1

log "===== H2-a chain start (deadline-aware: 9 groups x 515 sentences) ====="
for i in 1 2 3; do
  log "attempt $i"
  venv/bin/python -u tools/blank_gov/p227_h2a_probe.py \
      >> artifacts/logs_P227_h2a.log 2>&1
  rc=$?
  log "attempt $i rc=$rc"
  if [ "$rc" -eq 0 ] && [ -f artifacts/metrics/blank-gov/p227-h2a-verdict.txt ]; then
    log "DONE verdict=$(cat artifacts/metrics/blank-gov/p227-h2a-verdict.txt)"
    exit 0
  fi
  log "retry after 30s (new process -> new CUDA context)"
  sleep 30
done
log "FAILED after 3 attempts; partial receipts kept for manual resume"
exit 1
