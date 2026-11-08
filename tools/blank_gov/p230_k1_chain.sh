#!/usr/bin/env bash
# P230 / K1 自动链：A 对齐片段库 -> B 生成句 -> C 合成样本 -> 冒烟 -> D 6轮训练 -> 判定 -> 报告
cd /home/su127/FYP/domain-bounded-cslr || exit 1
LOG=artifacts/logs_P230_k1_chain.log
: > "$LOG"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

source venv/bin/activate || exit 1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CLIPS=artifacts/cache/p230_k1_clips.json
SENTS=artifacts/cache/p231_k1_sentences.json
MANIFEST=artifacts/cache/p232_k1_manifest.json
PY=venv/bin/python
PIPE=tools/blank_gov/p230_k1_pipeline.py
TRAINLOG=artifacts/logs_P233_k1_train.log

log "===== K1 chain start (prereg 20:45) ====="

# ---- A: train 全量强制对齐（非训练、只读前向；分片断点 + 3 次重试） ----
if [ -f "$CLIPS" ]; then
  log "A skip (片段库已存在)"
else
  for i in 1 2 3; do
    log "A attempt $i"
    $PY -u "$PIPE" --stage align >> artifacts/logs_P230_k1_align.log 2>&1
    rc=$?
    log "A attempt $i rc=$rc"
    [ "$rc" -eq 0 ] && [ -f "$CLIPS" ] && break
    sleep 30
  done
  if [ ! -f "$CLIPS" ]; then log "A FAILED -> abort"; exit 1; fi
fi
log "A done"

# ---- B / C ----
if [ ! -f "$SENTS" ]; then
  $PY -u "$PIPE" --stage sentences >> artifacts/logs_P231_k1_sentences.log 2>&1
  [ $? -eq 0 ] || { log "B FAILED -> abort"; exit 1; }
fi
log "B done"
if [ ! -f "$MANIFEST" ]; then
  $PY -u "$PIPE" --stage synth >> artifacts/logs_P232_k1_synth.log 2>&1
  [ $? -eq 0 ] || { log "C FAILED -> abort"; exit 1; }
fi
log "C done"

# ---- 打补丁（幂等） ----
if ! $PY tools/blank_gov/p78_train_official_tfnet.py --help 2>&1 | grep -q -- "--k1-manifest"; then
  log "applying K1 patch"
  $PY tools/blank_gov/p233_patch_k1.py >> "$LOG" 2>&1 || { log "patch FAILED"; exit 1; }
  $PY -m py_compile tools/blank_gov/p78_train_official_tfnet.py >> "$LOG" 2>&1 || { log "patch syntax FAILED"; exit 1; }
else
  log "K1 patch already applied"
fi

COMMON="--module TFNet --img-size 160 --batch 2 --mem-budget 7e9 --amp-bf16 \
--lr 2e-5 --weight-decay 1e-4 --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
--freeze-backbone --no-resume-sched"

# ---- 冒烟（只验证能跑，不判方向） ----
log "smoke (32/16, 1 epoch)"
timeout 1200 $PY -u tools/blank_gov/p78_train_official_tfnet.py \
  $COMMON --tag p233-k1-smoke --workers 0 \
  --k1-manifest "$MANIFEST" \
  --epochs 1 --total-epochs 6 --max-train 32 --max-dev 16 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt \
  > artifacts/logs_P233_k1_smoke.log 2>&1
rc=$?
log "smoke rc=$rc"
if [ "$rc" -ne 0 ]; then log "SMOKE FAILED -> abort"; exit 1; fi

# ---- D: 正式 6 轮（断点续 + 3 次重试） ----
: > "$TRAINLOG"
for i in 1 2 3; do
  log "D attempt $i"
  $PY -u tools/blank_gov/p78_train_official_tfnet.py \
    $COMMON --tag p233-k1 \
    --k1-manifest "$MANIFEST" \
    --epochs 6 --total-epochs 6 \
    --resume artifacts/checkpoints/p184-init-tfnet160.pt \
    >> "$TRAINLOG" 2>&1
  rc=$?
  log "D attempt $i rc=$rc"
  if [ "$rc" -eq 0 ] && grep -q "ep006 ctc" "$TRAINLOG"; then
    break
  fi
  log "D retry after 30s"
  sleep 30
done
if [ "$rc" -ne 0 ] || ! grep -q "ep006 ctc" "$TRAINLOG"; then
  log "TRAIN FAILED (rc/轮数检查) -> ABORT，不评估不判定"
  exit 1
fi
log "D done (6 epochs)"

# ---- 评估 + 判定 + 报告 ----
log "eval"
$PY -u tools/blank_gov/night_eval_full.py \
  --ckpt artifacts/checkpoints/p233-k1-best.pt --tag p233-k1 \
  > artifacts/logs_P233_k1_eval.log 2>&1
[ $? -eq 0 ] || { log "EVAL FAILED -> abort"; exit 1; }

log "judge"
$PY -u tools/blank_gov/night_judge.py --tag p233-k1 --base p184-base \
  --log "$TRAINLOG" --kind train \
  --title "K1 / P233 语料合成（CTC 对齐片段库 + 新 gloss 句拼接）" \
  > artifacts/logs_P233_k1_judge.log 2>&1
[ $? -eq 0 ] || { log "JUDGE FAILED -> abort"; exit 1; }

log "report"
$PY -u "$PIPE" --stage report >> "$LOG" 2>&1
log "===== K1 chain END verdict=$(cat artifacts/metrics/blank-gov/p233-k1-judge.json | grep -m1 '"verdict"' | tr -d ' ,\"' | cut -d: -f2) ====="
exit 0
