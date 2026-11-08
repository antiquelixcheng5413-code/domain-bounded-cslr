#!/bin/bash
# G0 断点续跑 + 驱动错误自动重试（2026-11-08 10:54）
# 08:30 首跑被用户叫停(1500/2487)；10:40 完整重跑在 2001/4973 撞 CUDA_ERROR_UNKNOWN。
# 现在从 p222_g0_partial.json 续跑；若再撞驱动错误，新进程重建 CUDA context 后重试。
set -u
cd /home/su127/FYP/domain-bounded-cslr || exit 1
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
LOG=artifacts/logs_P222_g0_resume.log
: > "$LOG"

for i in 1 2 3 4; do
  echo "[$(date '+%F %T')] === resume attempt $i ===" | tee -a "$LOG"
  G0_RESUME=1 python -u tools/blank_gov/p222_g0_train_confusion.py >> "$LOG" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] attempt $i rc=$rc" | tee -a "$LOG"
  if [ "$rc" -eq 0 ] && [ -f artifacts/metrics/blank-gov/p222-g0-verdict.txt ]; then
    echo "[$(date '+%F %T')] G0 COMPLETE verdict=$(cat artifacts/metrics/blank-gov/p222-g0-verdict.txt)" | tee -a "$LOG"
    exit 0
  fi
  # 崩时 partial 仍在上次成功存盘点，下一次自动从那里续
  echo "[$(date '+%F %T')] retry in 30s (partial still present)" | tee -a "$LOG"
  sleep 30
done
echo "[$(date '+%F %T')] G0 FAILED after 4 attempts" | tee -a "$LOG"
exit 1
