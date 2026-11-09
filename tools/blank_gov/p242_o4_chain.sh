#!/usr/bin/env bash
# P242 / O4 自动链：冒烟 -> 6 轮 PEFT -> 评估 -> R1-R8 判定 -> 机制报告 -> commit（不 push）
# 预注册 docs/planning/O4-PREREG-2026-11-09.md（08:24 写死）
cd /home/su127/FYP/domain-bounded-cslr || exit 1
LOG=artifacts/logs_P242_o4_chain.log
: > "$LOG"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

source venv/bin/activate || exit 1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=venv/bin/python
M=artifacts/metrics/blank-gov
TRAINLOG=artifacts/logs_P242_o4_train.log

COMMON="--module TFNet --img-size 160 --batch 2 --mem-budget 7e9 --amp-bf16 \
--lr 2e-5 --weight-decay 1e-4 --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
--freeze-backbone --no-resume-sched"
O4="--o4-weight 0.3 --o4-tau 0.07 --o4-clips artifacts/cache/p230_k1_clips.json"

log "===== O4 chain start | prereg O4-PREREG-2026-11-09.md (08:24) ====="
log "严禁：push / test / 改判据 / 调 lambda-tau / 删文件 / 第二个训练"

# ---------------------------------------------------------------- 冒烟
log "--- smoke (32/16, 1 epoch) ---"
timeout 1200 $PY -u tools/blank_gov/p78_train_official_tfnet.py \
  $COMMON $O4 --tag p242-o4-smoke --workers 0 \
  --epochs 1 --total-epochs 6 --max-train 32 --max-dev 16 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt \
  > artifacts/logs_P242_o4_smoke.log 2>&1
rc=$?
log "smoke rc=$rc"
grep -E "^O4:|ep001 ctc|Traceback|Error|nan_steps" artifacts/logs_P242_o4_smoke.log >> "$LOG"
if [ "$rc" -ne 0 ]; then log "SMOKE FAILED -> abort"; exit 1; fi
if ! grep -q "^O4:" artifacts/logs_P242_o4_smoke.log; then
  log "O4 bank 未生效 -> abort"; exit 1
fi

# ---------------------------------------------------------------- 正式训练
: > "$TRAINLOG"
rc=1
for i in 1 2 3; do
  RES=artifacts/checkpoints/p184-init-tfnet160.pt
  [ -f artifacts/checkpoints/p242-o4-last.pt ] && RES=artifacts/checkpoints/p242-o4-last.pt
  log "train attempt $i resume=$RES"
  timeout 19800 $PY -u tools/blank_gov/p78_train_official_tfnet.py \
    $COMMON $O4 --tag p242-o4 --epochs 6 --total-epochs 6 --resume "$RES" \
    >> "$TRAINLOG" 2>&1
  rc=$?
  log "train attempt $i rc=$rc"
  if [ "$rc" -eq 0 ] && grep -q "ep006 ctc" "$TRAINLOG"; then break; fi
  if [ "$rc" -eq 124 ]; then log "timeout -> stop retry"; break; fi
  sleep 30
done
if [ "$rc" -ne 0 ] || ! grep -q "ep006 ctc" "$TRAINLOG"; then
  log "TRAIN FAILED (rc/轮数检查) -> ABORT，不评估不判定"
  exit 1
fi
log "train done (6 epochs)"
grep -E "ep0[0-9][0-9] ctc" "$TRAINLOG" | tail -6 >> "$LOG"

# ---------------------------------------------------------------- 评估
log "--- eval: wait 150s for WSL GPU recovery, batch=1, max 3 tries ---"
sleep 150
erc=1
for j in 1 2 3; do
  $PY -u tools/blank_gov/night_eval_full.py \
    --ckpt artifacts/checkpoints/p242-o4-best.pt --tag p242-o4 --batch 1 \
    > artifacts/logs_P242_o4_eval.log 2>&1
  erc=$?
  log "eval attempt $j rc=$erc"
  [ "$erc" -eq 0 ] && break
  sleep 150
done
if [ "$erc" -ne 0 ]; then log "EVAL FAILED -> abort"; exit 1; fi

$PY -u tools/blank_gov/n2_align_peritem.py --cand p242-o4 --base p184-base >> "$LOG" 2>&1
log "align rc=$?"

$PY -u tools/blank_gov/night_judge.py --tag p242-o4 --base p184-base \
  --log "$TRAINLOG" --kind train \
  --title "O4 / P242 训练期 Viterbi 区间级监督（静态片段库 + 帧级 cosine CE）" \
  > artifacts/logs_P242_o4_judge.log 2>&1
jrc=$?
log "judge rc=$jrc"
grep -E '"verdict"|R[1-8]_|delta_WER_pp|ci95' artifacts/logs_P242_o4_judge.log >> "$LOG"

$PY -u tools/blank_gov/night_report.py --tag p242-o4 \
  --title "O4 训练期区间级监督" >> "$LOG" 2>&1
log "report rc=$?"

# ---------------------------------------------------------------- 机制报告
$PY -u - <<'PYEOF' > artifacts/logs_P242_o4_mechanism.log 2>&1
import json
from pathlib import Path
R = Path("/home/su127/FYP/domain-bounded-cslr")
D = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
M = R / "artifacts/metrics/blank-gov"

def jload(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

b = jload(M / "p184-base-uwer.json")
c = jload(M / "p242-o4-uwer.json")
j = jload(M / "p242-o4-judge.json")
if b is None or c is None:
    print("机制报告跳过：收据缺失")
    raise SystemExit(0)
nb0 = b.get("nonblank_frame_ratio_pct")
nb1 = c.get("nonblank_frame_ratio_pct")
verdict = j.get("verdict", "?") if j else "?"
out = {
    "tag": "p242-o4", "verdict": verdict,
    "nonblank_frame_ratio_base_pct": nb0,
    "nonblank_frame_ratio_cand_pct": nb1,
    "delta_pp": (round(nb1 - nb0, 3) if isinstance(nb0, (int, float))
                 and isinstance(nb1, (int, float)) else None),
    "mechanism_verified": bool(isinstance(nb0, (int, float))
                               and isinstance(nb1, (int, float)) and nb1 > nb0),
    "note": "机制报告义务（非 PASS 门槛）：若 R1-R8 全过但占比未上升，仍判 PASS 但标注机制未验证",
}
(M / "p242-o4-mechanism.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
txt = """
## 附：O4 机制验证（预注册 O4-PREREG §2，非 PASS 门槛）

- CTC 非 blank 帧占比：基线 **%s%%** → O4 **%s%%**（Δ %s pp）。
- 机制是否验证（占比上升）：**%s**。
- verdict = **%s**。按预注册：该指标是**报告义务**，不是新增门槛；
  若 R1–R8 全过但未上升，仍判 PASS 并标注「机制未验证，需复核」。
- 区间只来自 train 的 K1 静态 Viterbi 片段库；λ=0.3、τ=0.07 写死未调；
  未跑 test、未改任何判据。
""" % (nb0, nb1, out["delta_pp"], out["mechanism_verified"], verdict)
for base in (R, D):
    p = base / "docs/experiments/stage-p242-o4.md"
    if p.exists():
        with open(p, "a", encoding="utf-8") as f:
            f.write(txt)
print(json.dumps(out, ensure_ascii=False, indent=2))
PYEOF
log "mechanism rc=$?"
tail -8 artifacts/logs_P242_o4_mechanism.log >> "$LOG"

# ---------------------------------------------------------------- commit（不 push）
git add docs/planning/O4-PREREG-2026-11-09.md \
        docs/experiments/stage-p242-o4.md \
        tools/blank_gov/o4_interval_module.py tools/blank_gov/p242_patch_o4.py \
        tools/blank_gov/p242_o4_chain.sh tools/blank_gov/p78_train_official_tfnet.py \
        $M/p242-o4-*.json $M/p242-o4-mechanism.json 2>/dev/null
cat > /tmp/o4_commit.txt <<'EOF'
O4 训练期 Viterbi 区间级监督（P242）：预注册 + 实现 + 6 轮结果与判定

- 预注册 O4-PREREG-2026-11-09.md（08:24 写死）：唯一单变量 = 训练期
  在 classifier11 输入特征上加静态 Viterbi 区间的帧级 cosine-CE；
  λ=0.3 / τ=0.07 / 片段库 p230_k1_clips.json（K1 A 阶段 CTC 强制对齐，只含 train）/
  比例映射避开 MAX_FRAMES 下采样 / 目标原型取分类头权重并 detach、不新增可训练参数。
  不动 CTC 输出 / blank / 类先验，以避开 F1 与 P185 的失败模式。
- 判据统一 R1-R8；附加机制报告义务（非门槛）：必须报 CTC 非 blank 帧占比。
  PASS 仅代表值得 2 seed 复跑、不进 test；FAIL 不调参重跑。
- 与已判死项区别：P185 多尺度 CTC 头 / F1 类先验 / E1 动量原型 InfoNCE /
  F2 文本嵌入原型 / D2 自洽 KL / D3 TFD 边界，机制均不同。
  反证据已写入：P185「额外监督破坏对齐」、F1 明确负向、EMNLP26 L2 边界平滑负收益、
  同模型切分的边界自我强化偏差。
- 链：冒烟 -> 6 轮 PEFT -> 等 150s + batch1 评估(3 次重试) -> sid 对齐 ->
  R1-R8 判定 -> 机制报告 -> commit（不 push）。训练须 rc==0 且 ep006 存在。
未跑 test；本次仅 commit，未 push。
EOF
git commit -F /tmp/o4_commit.txt >> "$LOG" 2>&1
crc=$?
log "commit rc=$crc head=$(git log --oneline -1 | cut -c1-40)"
log "===== O4 chain END $(date) ====="
exit 0
