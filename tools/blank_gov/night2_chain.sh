#!/usr/bin/env bash
# 夜间接力 #2：等 K1 链退出 -> S0 核对 -> L1/L2 CPU -> S3 seed43 训练 -> S4 汇总+commit
# 预注册：docs/planning/N2-PREREG-2026-11-08.md（20:55 写死）
# 严禁：push / test / 删文件 / 改判据 / 超时训练。允许自动 commit（不 push）。
cd /home/su127/FYP/domain-bounded-cslr || exit 1

LOG=artifacts/logs_NIGHT2_chain.log
STATE=artifacts/cache/night2_state.txt
: > "$LOG"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
state() { echo "$1 $(date '+%Y-%m-%d %H:%M:%S')" > "$STATE"; }

source venv/bin/activate || exit 1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=venv/bin/python
M=artifacts/metrics/blank-gov
DEADLINE=$(date -d '2026-11-09 09:00:00' +%s)
remain() { echo $(( DEADLINE - $(date +%s) )); }

K1PID=492
MANIFEST=artifacts/cache/p232_k1_manifest.json
S43LOG=artifacts/logs_P238_k1_s43_train.log

log "===== NIGHT2 chain start | deadline 2026-11-09 09:00 | remain=$(remain)s ====="
state "WAIT-K1"
log "预注册 N2-PREREG-2026-11-08.md (20:55) 已写死；严禁 push/test/删文件/改判据"

# ---------------------------------------------------------------- S0 wait K1
while kill -0 "$K1PID" 2>/dev/null; do
  r=$(remain)
  if [ "$r" -lt 18000 ]; then
    log "K1 仍未结束且 remain=${r}s <18000s -> 放弃等待，S0 标 INVALID，不启动 S3 训练"
    break
  fi
  sleep 60
done
log "K1 chain 已退出（或等待超时），进入 S0"
state "S0"

K1U=$M/p233-k1-uwer.json
K1J=$M/p233-k1-judge.json
K1_VALID=0
if [ -f "$K1U" ] && [ -f "$K1J" ] && grep -q 'ep006 ctc' artifacts/logs_P233_k1_train.log 2>/dev/null; then
  K1_VALID=$($PY -c "
import json
d=json.load(open('$K1U'))
print(1 if d.get('n')==515 and len(d.get('per_item',[]))==515 else 0)
" 2>/dev/null)
fi
K1V=$($PY -c "
import json
try:
    print(json.load(open('$K1J')).get('verdict', '?'))
except Exception:
    print('未生成')
" 2>/dev/null)
log "S0 K1_VALID=$K1_VALID  verdict=$K1V"
state "S0-DONE valid=$K1_VALID verdict=$K1V"

# ---------------------------------------------------------------- L1 (纯 CPU)
log "--- L1 spike 邻域后验融合扫描（零训练 CPU） ---"
state "L1"
$PY -u tools/blank_gov/p234_n2_spike_fusion.py > artifacts/logs_P234_n2_l1.log 2>&1
rc=$?
log "L1 rc=$rc verdict=$(cat $M/p234-n2-verdict.txt 2>/dev/null || echo 未生成)"
if [ "$rc" -ne 0 ]; then log "L1 失败（不阻塞后续阶段）"; fi
tail -8 artifacts/logs_P234_n2_l1.log >> "$LOG"
state "L1-DONE rc=$rc"

# ---------------------------------------------------------------- L2 (纯 CPU)
log "--- L2 K1 语料质量审计（零成本 CPU，探索性） ---"
state "L2"
$PY -u tools/blank_gov/p235_n2_k1_audit.py > artifacts/logs_P235_n2_l2.log 2>&1
rc=$?
log "L2 rc=$rc"
if [ "$rc" -ne 0 ]; then log "L2 失败（不阻塞后续阶段）"; fi
tail -6 artifacts/logs_P235_n2_l2.log >> "$LOG"
state "L2-DONE rc=$rc"

# ---------------------------------------------------------------- S3 seed43
NEED=18000     # 4.5h 训练 + 0.5h 收尾
r=$(remain)
if [ "$K1_VALID" -ne 1 ] || [ ! -f "$MANIFEST" ]; then
  log "S3 SKIP（K1 无效或 manifest 缺失），2 seed 复跑待人工补"
  state "S3-SKIP"
elif [ "$r" -lt "$NEED" ]; then
  log "S3 SKIP（remain=${r}s < ${NEED}s，时间盒不足），2 seed 复跑待人工补"
  state "S3-SKIP-TIME"
else
  log "--- S3 K1 seed43 复跑（6 轮，remain=${r}s） ---"
  state "S3-TRAIN"
  # 确保 K1 注入补丁存在（幂等）
  if ! $PY tools/blank_gov/p78_train_official_tfnet.py --help 2>&1 | grep -q -- "--k1-manifest"; then
    $PY tools/blank_gov/p233_patch_k1.py >> "$LOG" 2>&1
    $PY -m py_compile tools/blank_gov/p78_train_official_tfnet.py >> "$LOG" 2>&1
  fi
  COMMON="--module TFNet --img-size 160 --batch 2 --mem-budget 7e9 --amp-bf16 \
--lr 2e-5 --weight-decay 1e-4 --hidden 1024 --seed 43 --official-pad --fix-rgb-order \
--freeze-backbone --no-resume-sched"
  : > "$S43LOG"
  rc=1
  TL=$(( $(remain) - 3600 ))      # 留 1h 给评估/汇总，确保不越过 09:00
  if [ "$TL" -lt 3600 ]; then TL=3600; fi
  log "S3 训练时间上限 ${TL}s"
  for i in 1 2 3; do
    RES=artifacts/checkpoints/p184-init-tfnet160.pt
    if [ -f artifacts/checkpoints/p238-k1-s43-last.pt ]; then
      RES=artifacts/checkpoints/p238-k1-s43-last.pt
    fi
    log "S3 attempt $i resume=$RES remain=$(remain)s"
    timeout "$TL" $PY -u tools/blank_gov/p78_train_official_tfnet.py \
      $COMMON --tag p238-k1-s43 --k1-manifest "$MANIFEST" \
      --epochs 6 --total-epochs 6 --resume "$RES" >> "$S43LOG" 2>&1
    rc=$?
    log "S3 attempt $i rc=$rc"
    if [ "$rc" -eq 0 ] && grep -q 'ep006 ctc' "$S43LOG"; then break; fi
    if [ "$rc" -eq 124 ]; then log "S3 训练触发时间上限 -> 停止重试，保留 last.pt"; break; fi
    sleep 30
  done
  if [ "$rc" -ne 0 ] || ! grep -q 'ep006 ctc' "$S43LOG"; then
    log "S3 TRAIN FAILED（rc/轮数检查）-> ABORT 该阶段，不评估不判定"
    state "S3-TRAIN-FAILED"
  else
    log "S3 训练完成 6 轮，开始评估"
    $PY -u tools/blank_gov/night_eval_full.py \
      --ckpt artifacts/checkpoints/p238-k1-s43-best.pt --tag p238-k1-s43 \
      > artifacts/logs_P238_k1_s43_eval.log 2>&1
    erc=$?
    log "S3 eval rc=$erc"
    if [ "$erc" -eq 0 ]; then
      $PY -u tools/blank_gov/night_judge.py --tag p238-k1-s43 --base p184-base \
        --log "$S43LOG" --kind train \
        --title "N2-S3 / K1 语料合成 seed 43 复跑（2 seed 稳健性）" \
        > artifacts/logs_P238_k1_s43_judge.log 2>&1
      log "S3 judge rc=$?"
      $PY -u tools/blank_gov/night_report.py --tag p238-k1-s43 \
        --title "N2-S3 / K1 seed 43 复跑" >> "$LOG" 2>&1
    fi
    state "S3-DONE"
  fi
fi

# ---------------------------------------------------------------- S4 汇总
log "--- S4 汇总 + commit（不 push） ---"
state "S4"
$PY -u tools/blank_gov/night2_summary.py > artifacts/logs_NIGHT2_summary.log 2>&1
src=$?
log "summary rc=$src remain=$(remain)s"
tail -6 artifacts/logs_NIGHT2_summary.log >> "$LOG"

# 自动 commit（明确清单；文件不存在则跳过）
for f in \
  docs/planning/N2-PREREG-2026-11-08.md \
  docs/experiments/stage-P233-K1-corpus-synthesis.md \
  docs/experiments/stage-P234-N2-spike-fusion.md \
  docs/experiments/stage-P235-N2-K1-corpus-audit.md \
  docs/experiments/stage-p238-k1-s43.md \
  docs/experiments/OVERNIGHT-2-RUN-2026-11-09.md \
  tools/blank_gov/p234_n2_spike_fusion.py \
  tools/blank_gov/p235_n2_k1_audit.py \
  tools/blank_gov/night2_summary.py \
  tools/blank_gov/night2_chain.sh; do
  [ -f "$f" ] && git add "$f"
done
for f in $M/p233-k1-*.json $M/p234-n2-*.json $M/p235-n2-*.json \
         $M/p238-k1-s43-*.json; do
  [ -f "$f" ] && git add "$f"
done
cat > /tmp/night2_commit.txt <<'EOF'
N2 夜间接力 #2：K1 收尾 + L1 后验融合 + L2 语料审计 + S3 seed43

- 预注册 N2-PREREG-2026-11-08.md（20:55 写死，早于 K1 结果）：时间盒 09:00 硬停、
  队列 S0/L1/L2/S3/S4、护栏与主判据、严禁 push/test/删文件/改判据。
- L1 spike 邻域后验融合扫描（零训练纯 CPU）：k{1,2,4} x lambda{0.3,0.6} 共 6 档，
  护栏 R3/R5/R6 内取 WER 最低；主判据 Δ>=2.00pp 且配对 bootstrap CI 下界>0。
  与 I0「改输入帧率无用」互补，检验改后验邻域是否含新信息。
- L2 K1 语料质量与对齐边界审计：纯描述性诊断，不判 PASS/FAIL、不改 K1 判据。
- S3 K1 的 seed 43 复跑：条件触发（remain>=18000s 且 K1 有效），
  唯一改动 seed/tag，6 轮 PEFT + R1-R8 判定；项目规范「2 seed」要求，非调参重跑。
- 汇总 OVERNIGHT-2-RUN-2026-11-09.md 只转述收据原值，缺失项明确标注。
- 未跑 test、未 push；本次仅自动 commit。
EOF
git commit -F /tmp/night2_commit.txt >> "$LOG" 2>&1
crc=$?
HEADNOW=$(git log --oneline -1 | cut -c1-40)
AHEAD=$(git log --oneline origin/experiment/blank-governance-p0-p38..HEAD | wc -l)
log "commit rc=$crc head=$HEADNOW"
log "双端 ahead（推送仍等用户明确指令）：$AHEAD 个提交"

state "DONE"
log "===== NIGHT2 chain END $(date) remain=$(remain)s ====="
exit 0
