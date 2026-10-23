#!/usr/bin/env bash
# 链式：等 P167 跑完 -> 起 D3（VAC+224+RGB）
#
# 目的：P167 还剩十几个小时，中间 GPU 不能空转，也不能靠人守着切换。
# 这个脚本会：
#   1) 轮询等 p78_train 进程退出 + GPU 显存释放
#   2) 起 D3 冒烟（40 条 / 2 轮）——专门用来验证 VAC@224 在 8GB 上会不会 OOM
#      ⚠️ 这一步是必须的：D4(TFNet@224) 实测被迫 batch 1 才 6h/轮，
#         VAC@224 的显存峰值同样是未知数，不能靠外推
#   3) 冒烟通过 => 起 D3 正式跑（55 轮）
#      冒烟 OOM  => 自动把 BATCH 降到 1 再冒烟一次，通过则带 BATCH=1 跑正式
#
# 用法（前台持有会话跑，别 nohup，WSL 会回收）：
#   bash tools/blank_gov/run_d3_after_p167.sh
set -u
cd "$(dirname "$0")/../.." || exit 1

CHAIN_LOG=artifacts/logs_chain_d3_after_p167.log
mkdir -p artifacts

log() { echo "[chain $(date '+%F %T')] $*" | tee -a "$CHAIN_LOG"; }

log "=== 链式启动：等 P167 -> D3 ==="
# 记录 P167 的起点状态，便于事后核对等的是不是它
log "P167 已完成轮数: $(grep -cE 'ep[0-9]{3}' artifacts/logs_P167_wh0.log 2>/dev/null || echo '?')"
log "P167 最后一行: $(grep -E 'ep[0-9]{3}' artifacts/logs_P167_wh0.log 2>/dev/null | tail -1)"

# ---------- 1. 等 P167 退出 ----------
WAIT=0
while pgrep -f p78_train >/dev/null; do
  sleep 30
  WAIT=$((WAIT + 30))
  if [ $((WAIT % 600)) -eq 0 ]; then
    log "  仍在等 P167（已等 ${WAIT}s，当前 $(grep -cE 'ep[0-9]{3}' artifacts/logs_P167_wh0.log 2>/dev/null || echo '?') 轮）"
  fi
done
log "P167 进程已退出（共等 ${WAIT}s）"

# ---------- 2. 等 GPU 显存真正释放 ----------
for i in $(seq 1 20); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
  if [ "${USED:-99999}" -lt 500 ]; then
    log "GPU 已释放（${USED} MiB）"
    break
  fi
  log "  GPU 还有 ${USED} MiB，等释放… ($i/20)"
  sleep 15
done

# ---------- 3. D3 冒烟（验证 VAC@224 显存） ----------
log "=== 起 D3 冒烟（batch=${BATCH:-2}）==="
bash tools/blank_gov/run_d3_vac224.sh smoke > artifacts/logs_D3_smoke_run.log 2>&1
SMOKE_RC=$?
SMOKE_OOM=$(grep -icE 'out of memory' artifacts/logs_D3_smoke.log artifacts/logs_D3_smoke_run.log 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
log "冒烟退出码=$SMOKE_RC  OOM 计数=$SMOKE_OOM"

FINAL_BATCH="${BATCH:-2}"
if [ "$SMOKE_RC" -ne 0 ] || [ "$SMOKE_OOM" -gt 0 ]; then
  log "⚠️ batch=${FINAL_BATCH} 冒烟失败 ⇒ 降到 batch=1 重试"
  BATCH=1 bash tools/blank_gov/run_d3_vac224.sh smoke > artifacts/logs_D3_smoke_run.log 2>&1
  SMOKE_RC=$?
  SMOKE_OOM=$(grep -icE 'out of memory' artifacts/logs_D3_smoke.log artifacts/logs_D3_smoke_run.log 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
  log "batch=1 冒烟退出码=$SMOKE_RC  OOM 计数=$SMOKE_OOM"
  FINAL_BATCH=1
fi

if [ "$SMOKE_RC" -ne 0 ] || [ "$SMOKE_OOM" -gt 0 ]; then
  log "🔴 D3 两档 batch 都冒烟失败，不起正式跑。请人工看 artifacts/logs_D3_smoke_run.log"
  exit 1
fi
log "✅ 冒烟通过，正式跑 batch=$FINAL_BATCH"

# ---------- 4. 清掉冒烟产物（tag 已隔离，但保险起见确认正式 ckpt 不存在） ----------
ls -la artifacts/checkpoints/ | grep 'p157' | tee -a "$CHAIN_LOG"

# ---------- 5. D3 正式跑 ----------
log "=== 起 D3 正式跑（55 轮，batch=$FINAL_BATCH）==="
BATCH="$FINAL_BATCH" bash tools/blank_gov/run_d3_vac224.sh > artifacts/logs_D3_launch.log 2>&1
D3_RC=$?
log "D3 正式跑退出码=$D3_RC"

if [ "$D3_RC" -ne 0 ]; then
  log "⚠️ D3 中断 ⇒ 续跑命令： RESUME=1 BATCH=$FINAL_BATCH bash tools/blank_gov/run_d3_vac224.sh"
fi
exit $D3_RC
