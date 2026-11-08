#!/bin/bash
# 夜间自动实验链 2026-11-07 23:35 -> 2026-11-08 09:00
# 规则：一次只跑一个训练；每阶段写 state；9 点硬停；
#       所有判据启动前写死（night_judge.py），不事后调整；
#       未跑 test、未 commit、未 push。
set -u
cd /home/su127/FYP/domain-bounded-cslr || exit 1
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
PY=venv/bin/python
LOG=artifacts/logs_NIGHT_chain.log
DEADLINE=$(date -d "2026-11-08 09:00:00" +%s)
MIN_TRAIN=12600          # 3.5h，训练启动所需最短剩余时间
mkdir -p artifacts/cache artifacts/metrics/blank-gov docs/experiments
: > "$LOG"

log()  { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
remain() { echo $(( DEADLINE - $(date +%s) )); }
remain_gt() { [ "$(remain)" -gt "$1" ]; }
state() { echo "$1" > artifacts/cache/night_state.txt; log "STATE -> $1"; }

COMMON="--module TFNet --img-size 160 --batch 2 --mem-budget 7e9 --amp-bf16
        --lr 2e-5 --weight-decay 1e-4 --hidden 1024 --seed 42
        --official-pad --fix-rgb-order --freeze-backbone --no-resume-sched
        --epochs 6 --total-epochs 6
        --resume artifacts/checkpoints/p184-init-tfnet160.pt"

smoke() {  # $1=tag $2..=extra args
  local tag="$1"; shift
  log "SMOKE $tag"
  timeout 1500 $PY -u tools/blank_gov/p78_train_official_tfnet.py \
    $COMMON --tag "$tag" --workers 0 --epochs 1 --total-epochs 6 \
    --max-train 32 --max-dev 16 "$@" \
    > "artifacts/logs_${tag}.log" 2>&1
  local rc=$?
  if grep -qE "Traceback|CUDA out of memory|ERROR" "artifacts/logs_${tag}.log"; then
    log "SMOKE $tag FAILED (rc=$rc)"; tail -12 "artifacts/logs_${tag}.log" >> "$LOG"; return 1
  fi
  log "SMOKE $tag OK (rc=$rc)"; return 0
}

train_eval_judge() {  # $1=tag $2=ckpt $3=logfile $4=title $5..=extra args
  local tag="$1" ckpt="$2" tlog="$3" title="$4"; shift 4
  state "TRAIN $tag"
  if ! remain_gt "$MIN_TRAIN"; then
    log "SKIP $tag: remaining $(remain)s < ${MIN_TRAIN}s"; return 1
  fi
  local tl=$(( $(remain) - 1800 ))    # 留 30 min 给评估/报告/E3/收尾
  log "TRAIN $tag start (timeout ${tl}s)"
  timeout "$tl" $PY -u tools/blank_gov/p78_train_official_tfnet.py \
    $COMMON --tag "$tag" "$@" > "$tlog" 2>&1
  local rc=$?
  log "TRAIN $tag done rc=$rc"
  grep -E "ep0[0-9][0-9] ctc" "$tlog" | tail -8 >> "$LOG"
  if [ ! -f "$ckpt" ]; then
    log "SKIP eval $tag: no best ckpt ($ckpt)"; return 1
  fi
  state "EVAL $tag"
  $PY -u tools/blank_gov/night_eval_full.py --ckpt "$ckpt" --tag "$tag" \
    > "artifacts/logs_${tag}_eval.log" 2>&1
  if [ $? -ne 0 ]; then log "EVAL $tag FAILED"; tail -12 "artifacts/logs_${tag}_eval.log" >> "$LOG"; return 1; fi
  $PY -u tools/blank_gov/night_judge.py --tag "$tag" --base p184-base \
    --log "$tlog" --title "$title" --kind train \
    > "artifacts/logs_${tag}_judge.log" 2>&1
  grep -E '"verdict"|"delta_WER_pp"|"ci95"' "artifacts/logs_${tag}_judge.log" >> "$LOG"
  $PY -u tools/blank_gov/night_report.py --tag "$tag" >> "$LOG" 2>&1
  state "DONE $tag"
  return 0
}

finalize() {
  state "FINALIZE"
  $PY -u tools/blank_gov/night_report.py --summary \
    --tags "p212-e1-night,p215-f0,p213-f1,p214-f2,p216-e3" >> "$LOG" 2>&1
  local desk="/mnt/c/Users/su127/Desktop/中文手语识别"
  cat >> "$desk/.workbuddy/memory/2026-11-08.md" <<'LOGEOF'

## 夜间自动链（2026-11-07 23:35 → 2026-11-08 09:00，自动生成）
- 用户指令：明早 9 点前自动往下跑、自行决策、跑完继续找方向接着运行。
- 链：`tools/blank_gov/night_chain.sh`；统一判据 `night_judge.py`（预注册 R1–R7，
  主看 uWER/低频桶 + 高频桶/micro/插入数护栏）；统一评估 `night_eval_full.py`
  （WER_official + uWER + 低频/高频桶 + ins/sub/del + micro + 逐句编辑数）。
- 队列：E1 收尾 → F0 插入率感知解码扫描(零训练) → F1 训练期 logit adjustment
  → F2 视觉-文本语义原型对齐 → E3 手形聚类前提探针 → 总汇总。
- 依据：F1【有】Menon ICLR21 / Balanced Softmax + 本地 E0 前提 corr=0.8756；
  F2【有】FGXM PRL25 + 【间接】CVT-SLR CVPR23 / ACMC25；E3【有】EMNLP23 / HandsOn FG25。
- 每个训练：冒烟 → 正式 6 轮（沿用 P184 配方单变量）→ 全指标评估 → 判定 → 阶段报告
  （WSL + 桌面镜像）。9 点硬停，每阶段写 state 文件。
- 状态：未跑 test、未 commit、未 push。
LOGEOF
  log "FINALIZE done; state=$(cat artifacts/cache/night_state.txt)"
}

log "===== NIGHT CHAIN START $(date) ====="
log "deadline=$(date -d @$DEADLINE)  remaining=$(remain)s"
state "INIT"

# ---------------- S0: 等 E1 链结束并补全统一口径 ----------------
state "WAIT-E1"
while pgrep -f "p212_chain_e1.sh" >/dev/null 2>&1; do
  if ! remain_gt 600; then log "deadline while waiting E1"; break; fi
  sleep 60
done
log "E1 chain no longer running"
sleep 20

if remain_gt 1800; then
  state "S0-BASE-EVAL"
  $PY -u tools/blank_gov/night_eval_full.py \
    --ckpt artifacts/checkpoints/p184-tfnet-freeze-best.pt --tag p184-base \
    > artifacts/logs_p184-base_eval.log 2>&1 \
    && log "p184-base full metrics OK" \
    || log "p184-base eval FAILED"
  E1CK=artifacts/checkpoints/p212-e1-csgc-best.pt
  if [ -f "$E1CK" ]; then
    $PY -u tools/blank_gov/night_eval_full.py --ckpt "$E1CK" --tag p212-e1-night \
      > artifacts/logs_p212-e1-night_eval.log 2>&1
    $PY -u tools/blank_gov/night_judge.py --tag p212-e1-night --base p184-base \
      --log artifacts/logs_P212_e1_train.log --kind train \
      --title "E1 / P212 CSGC 跨句 gloss 原型对比（统一口径复判）" \
      > artifacts/logs_p212-e1-night_judge.log 2>&1
    $PY -u tools/blank_gov/night_report.py --tag p212-e1-night >> "$LOG" 2>&1
    log "E1 night verdict: $(grep -m1 '"verdict"' artifacts/logs_p212-e1-night_judge.log)"
  else
    log "E1 best ckpt missing, skip E1 re-judge"
  fi
fi

# ---------------- S1: F0 零训练解码扫描 ----------------
if remain_gt 1500; then
  state "F0-DECODE"
  $PY -u tools/blank_gov/p215_f0_decode_scan.py > artifacts/logs_P215_f0.log 2>&1
  log "F0 scan rc=$?  tail: $(tail -2 artifacts/logs_P215_f0.log | tr '\n' ' ')"
  if [ -f artifacts/metrics/blank-gov/p215-f0-uwer.json ]; then
    $PY -u tools/blank_gov/night_judge.py --tag p215-f0 --base p184-base \
      --kind decode --title "F0 / P215 插入率感知解码扫描（零训练）" \
      > artifacts/logs_P215_f0_judge.log 2>&1
    $PY -u tools/blank_gov/night_report.py --tag p215-f0 >> "$LOG" 2>&1
    log "F0 verdict: $(grep -m1 '"verdict"' artifacts/logs_P215_f0_judge.log)"
  fi
  state "DONE-F0"
fi

# ---------------- S2: F1 训练期 logit adjustment ----------------
if remain_gt "$MIN_TRAIN"; then
  state "F1-PATCH"
  cp tools/blank_gov/p78_train_official_tfnet.py /tmp/p78_pre_f1.py
  $PY tools/blank_gov/p213_patch_la.py >> "$LOG" 2>&1
  $PY -m py_compile tools/blank_gov/p78_train_official_tfnet.py \
    && $PY tools/blank_gov/p78_train_official_tfnet.py --help 2>&1 | grep -q -- "--la-tau" \
    && log "F1 patch OK" || { log "F1 patch FAILED, skip F1"; }
  if grep -q "F1 patch OK" "$LOG" && [ -f /tmp/p78_pre_f1.py ]; then
    if smoke "p213-f1-smoke" --la-tau 0.3 --la-pi-power 1.0; then
      train_eval_judge "p213-f1" \
        artifacts/checkpoints/p213-f1-best.pt \
        artifacts/logs_P213_f1_train.log \
        "F1 / P213 训练期 logit adjustment（LA-CTC）" \
        --la-tau 0.3 --la-pi-power 1.0
    else
      log "F1 smoke failed, skip training"
    fi
  fi
fi

# ---------------- S3: F2 视觉-文本语义原型对齐 ----------------
if remain_gt "$MIN_TRAIN"; then
  state "F2-TEXT-EMB"
  timeout 1800 $PY -u tools/blank_gov/p214_prep_text_emb.py \
    > artifacts/logs_P214_text_emb.log 2>&1
  log "text emb rc=$?  $(tail -2 artifacts/logs_P214_text_emb.log | tr '\n' ' ')"
  if [ -f artifacts/cache/p214_vta_target.npy ]; then
    state "F2-PATCH"
    cp tools/blank_gov/p78_train_official_tfnet.py /tmp/p78_pre_f2.py
    $PY tools/blank_gov/p214_patch_vta.py >> "$LOG" 2>&1
    $PY -m py_compile tools/blank_gov/p78_train_official_tfnet.py \
      && $PY tools/blank_gov/p78_train_official_tfnet.py --help 2>&1 | grep -q -- "--vta-weight" \
      && log "F2 patch OK" || { log "F2 patch FAILED, skip F2"; }
    if grep -q "F2 patch OK" "$LOG"; then
      if smoke "p214-f2-smoke" --vta-weight 0.05; then
        train_eval_judge "p214-f2" \
          artifacts/checkpoints/p214-f2-best.pt \
          artifacts/logs_P214_f2_train.log \
          "F2 / P214 视觉-文本语义原型对齐（VTA）" \
          --vta-weight 0.05
      else
        log "F2 smoke failed, skip training"
      fi
    fi
  else
    log "text emb missing, skip F2"
  fi
fi

# ---------------- S4: E3 手形聚类前提探针（纯 CPU） ----------------
if remain_gt 600; then
  state "E3-PROBE"
  timeout 900 $PY -u tools/blank_gov/p216_e3_handshape_probe.py \
    > artifacts/logs_P216_e3.log 2>&1
  log "E3 probe rc=$?  $(tail -3 artifacts/logs_P216_e3.log | tr '\n' ' ')"
  state "DONE-E3"
fi

# ---------------- S5: 备用探索（若还有 >=40 min，跑训练后 ckpt 的后处理组合扫描） ----------------
if remain_gt 2400 && [ -f tools/blank_gov/p217_posthoc_scan.py ]; then
  state "S5-POSTHOC"
  timeout 1800 $PY -u tools/blank_gov/p217_posthoc_scan.py \
    > artifacts/logs_P217_posthoc.log 2>&1
  log "S5 rc=$?  $(tail -3 artifacts/logs_P217_posthoc.log | tr '\n' ' ')"
  state "DONE-S5"
else
  log "S5 skipped (remaining $(remain)s)"
fi

finalize
log "===== NIGHT CHAIN END $(date)  remaining=$(remain)s ====="
