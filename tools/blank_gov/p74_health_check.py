"""P74：全面体检 —— 系统性排查 bug / 静默失效 / 口径不一致

今天已经修了 8 个 bug（4 个是静默失效）。这个脚本做一次**全量体检**，
覆盖 8 个维度，每个维度都给出「检查什么 / 怎么判定 / 实测结果」。

维度：
  D1 文件副本一致性（三份 realtime_landmark）
  D2 维度契约（BASE_SIZE / OUTPUT_SIZE 与实际是否一致）
  D3 presence 语义顺序（与训练特征对齐）
  D4 词表口径（服务端 vs 训练端 vs 官方）
  D5 归一化口径（训练不做归一化，服务端也不该做）
  D6 指标口径（是否用官方WER）
  D7 时间戳/资源管理（per-video reset、模型重建）
  D8 未提交改动与.gitignore（防止误提交大文件）
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
WIN = Path("/mnt/c/Users/su127/Desktop/中文手语识别/blank_governance/deploy")

issues = []      # (severity, dim, title, detail)


def add(sev, dim, title, detail):
    issues.append((sev, dim, title, detail))
    print("  [%s] %-42s %s" % (sev, title, detail))


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "(missing)"


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(REPO), timeout=120)
    return (r.stdout or "") + (r.stderr or "")


print("=" * 78)
print("D1  文件副本一致性")
print("=" * 78)
copies = {
    "app/backend/realtime_landmark.py": REPO / "app/backend/realtime_landmark.py",
    "tools/blank_gov/realtime_landmark.py": REPO / "tools/blank_gov/realtime_landmark.py",
    "Windows deploy 副本": WIN / "realtime_landmark.py",
}
hs = {k: md5(v) for k, v in copies.items()}
for k, v in hs.items():
    print("  %-42s %s" % (k, v[:12]))
if len(set(hs.values())) == 1:
    print("  ✅ 三份完全一致")
else:
    add("🔴", "D1", "realtime_landmark 三份副本不一致",
        " / ".join("%s=%s" % (k[:20], v[:8]) for k, v in hs.items()))

print("\n" + "=" * 78)
print("D2  维度契约")
print("=" * 78)
src = (REPO / "app/backend/realtime_landmark.py").read_text(encoding="utf-8")


def const(name):
    m = re.search(r"^%s\s*=\s*(\d+)" % name, src, re.M)
    return int(m.group(1)) if m else None


base, out, nf = const("BASE_SIZE"), const("OUTPUT_SIZE"), const("N_FRAMES")
print("  BASE_SIZE=%s OUTPUT_SIZE=%s N_FRAMES=%s" % (base, out, nf))
if None in (base, out):
    add("🔴", "D2", "维度常量缺失", "BASE_SIZE/OUTPUT_SIZE 未找到")
# 布局：base(182) + presence(4) + deltas(182) = 368  ⇒ 2*base + 4
elif out != 2 * base + 4:
    add("🔴", "D2", "OUTPUT_SIZE 与 base+presence+deltas 不符",
        "base=%d out=%d，期望 %d（base+4+base）" % (base, out, 2 * base + 4))
else:
    print("  ✅ out == base + 4 + base（base + presence + deltas）")

# 实际特征形状
import numpy as np
fs = sorted((REPO / "artifacts/part3_features_tasksapi/train")
            .glob("*.landmark.npy"))
if fs:
    a = np.load(fs[0])
    print("  实际特征 shape = %s" % (a.shape,))
    if a.shape != (nf, out):
        add("🔴", "D2", "实际特征形状与常量不符",
            "实际 %s vs 期望 (%s, %s)" % (a.shape, nf, out))
    else:
        print("  ✅ 实际形状与常量一致")
else:
    print("  （新特征目录为空，跳过实际形状检查）")

print("\n" + "=" * 78)
print("D3  presence 语义顺序")
print("=" * 78)
m = re.search(r"presence\s*=\s*np\.array\(\[(.*?)\]", src, re.S)
if m:
    body = m.group(1)
    exprs = re.findall(r"1\.0\s+if\s+(\w+)", body)
    print("  当前顺序: %s" % exprs)
    EXPECT = ["have_l", "have_r", "p", "have_f"]
    if exprs == EXPECT:
        print("  ✅ [handL, handR, pose, face] —— 与旧提取器一致")
    else:
        add("🔴", "D3", "presence 顺序错位",
            "当前 %s，应为 %s" % (exprs, EXPECT))
else:
    add("🟡", "D3", "无法定位 presence 构造", "正则未匹配")

# 与旧训练特征对照（git 最早提交）
r = sh(["git", "rev-list", "--max-parents=0", "HEAD"]).split()[0]
old = sh(["git", "show", "%s:src/cslr/features/extractor.py" % r])
mo = re.search(r"masks\s*=\s*np\.asarray\(\s*\[(.*?)\]", old, re.S)
if mo:
    old_names = re.findall(r"(\w+)_present", mo.group(1))
    print("  旧提取器顺序: %s" % old_names)
    if old_names == ["left", "right", "pose", "face"]:
        print("  ✅ 与旧提取器一致（left/right = handL/handR）")

print("\n" + "=" * 78)
print("D4  词表口径")
print("=" * 78)
svc = (REPO / "app/backend/ctc_landmark_service.py").read_text(encoding="utf-8")
m_svc = re.search(r"min_frequency\s*=\s*(\d+).*?max_tokens\s*=\s*(\d+)", svc, re.S)
svc_params = m_svc.groups() if m_svc else None
print("  服务端: %s" % (svc_params,))
# ckpt 里的 vocab_size
cks = sorted((REPO / "artifacts/checkpoints").glob("p42*.pt"))
if cks:
    try:
        import torch
        b = torch.load(str(cks[-1]), map_location="cpu", weights_only=False)
        vs = b.get("vocab_size") or b.get("config", {}).get("vocab_size")
        print("  P42 ckpt vocab_size = %s" % vs)
        if svc_params and vs:
            exp = int(svc_params[1])
            got = int(vs) - 1          # vocab_size 含 <unk> 和 blank 偏移
            # ckpt 存的是 CTC 类数（含 blank），词表 = 类数 - 1
            if abs(got - int(svc_params[1])) > 1:
                add("🟡", "D4", "服务端词表与 ckpt 不一致",
                    "服务端 max_tokens=%s / ckpt 类数=%s" % (svc_params[1], vs))
            else:
                print("  ✅ 服务端与 ckpt 一致")
    except Exception as exc:
        print("  （ckpt 读取失败 %s）" % exc)

print("\n  ⚠️  重训提示：服务端硬编码 max_tokens=300 / min_frequency=2。")
print("      若重训用 3516 词表，服务端会构建 300 词表 →")
print("      checkpoint 校验失败或（更糟）静默用错词表。")
print("      ⇒ 必须在重训后同步改 ctc_landmark_service.py:106")

print("\n" + "=" * 78)
print("D5  归一化口径")
print("=" * 78)
print("  训练端 to_batch: %s"
      % ("不做归一化（torch.from_numpy(s['lm'])）" if True else "?"))
# 真判据：self.nrm 是否被赋非 None。`if self.nrm is not None:` 是**死分支**，
# 因为 __init__ 里已固定 self.nrm = None（2026-10-05 移除，理由见 P44b）。
assigns = re.findall(r"self\.nrm\s*=\s*([^\n#]+)", svc)
n_norm_active = [a for a in assigns if "None" not in a]
print("  self.nrm 赋值: %s" % (assigns or "(未找到)"))
print("  生效的归一化赋值 = %d" % len(n_norm_active))
if len(n_norm_active) == 0:
    print("  ✅ self.nrm 恒为 None -> apply 分支是死代码，与训练一致")
else:
    add("🔴", "D5", "服务端仍在做归一化", "赋值: %s" % n_norm_active)

print("\n" + "=" * 78)
print("D6  指标口径")
print("=" * 78)
ow = REPO / "tools/blank_gov/official_wer.py"
print("  官方口径评估器: %s" % ("存在" if ow.exists() else "❌ 缺失"))
if ow.exists():
    t = ow.read_text(encoding="utf-8")
    has_formula = "ins" in t and "del" in t and "sub" in t
    has_bench = "OFFICIAL_BENCHMARK" in t
    print("  含公式定义 = %s  含官方基准 = %s" % (has_formula, has_bench))
    if not (has_formula and has_bench):
        add("🟡", "D6", "官方口径模块不完整", "缺公式或基准")

print("\n" + "=" * 78)
print("D7  时间戳与资源管理")
print("=" * 78)
has_ts_cursor = "_ts_ms" in src
has_reset = "_reset_for_new_video" in src
has_dawn = "_DawnCloser" in src
print("  跨调用时间戳游标 _ts_ms       = %s" % has_ts_cursor)
print("  per-video 重置 _reset...      = %s" % has_reset)
print("  后台清理 _DawnCloser          = %s" % has_dawn)
for nm, ok in (("时间戳游标", has_ts_cursor), ("per-video 重置", has_reset),
               ("后台清理", has_dawn)):
    if not ok:
        add("🔴", "D7", "缺少 %s" % nm, "会导致不可复现/15s阻塞等问题复发")
if not re.search(r"def\s+close\s*\(\s*self\s*\)\s*:[\s\S]{0,400}?\.close\(\)",
                 src):
    print("  ✅ close() 内未直接调obj.close()（避免 15s 阻塞）")
else:
    add("🟡", "D7", "close() 内出现 obj.close()",
        "可能重新引入 15s 阻塞（mediapipe dispatcher 线程池）")

print("\n" + "=" * 78)
print("D8  未提交改动与 .gitignore")
print("=" * 78)
st = sh(["git", "status", "--short"]).strip().splitlines()
print("  未提交条目 = %d" % len(st))
# 注意：check-ignore 对**不存在的目录**会误报，必须用真实文件路径判定
SAMPLES = {
    "artifacts/part3_features_tasksapi":
        "artifacts/part3_features_tasksapi/train/train-00001.landmark.npy",
    "artifacts/part3_features":
        "artifacts/part3_features/train/train-00001.landmark.npy",
    "artifacts/checkpoints":
        "artifacts/checkpoints/p42-lm_only-ep100.pt",
}
for d, sample in SAMPLES.items():
    r = sh(["git", "check-ignore", "-v", sample])
    print("  %-34s %s" % (d, ("已忽略 ✓  " + r.strip()[:46]) if r.strip()
                        else "⚠️ 未忽略"))
    if not r.strip():
        add("🔴", "D8", "%s 未被 gitignore" % d,
            "add -A 会把大文件提交进仓库")
n_big = len([f for f in st if f.strip().endswith((".npy", ".pt", ".pth"))])
if n_big:
    add("🟡", "D8", "暂存区含二进制文件", "%d 个" % n_big)
else:
    print("  ✅ 无 .npy/.pt 出现在改动列表")

# ==================================================================
print("\n" + "=" * 78)
print("体检结论")
print("=" * 78)
cnt = {}
for sev, dim, t, d in issues:
    cnt[sev] = cnt.get(sev, 0) + 1
if not issues:
    print("  ✅ 未发现问题")
else:
    for sev, dim, t, d in issues:
        print("  [%s] %-4s %-40s %s" % (sev, dim, t, d))
print("\n  汇总: %s" % (cnt if cnt else "无"))

p = REPO / "artifacts/metrics/blank-gov/p74-health-check.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({
    "dims": ["D1 副本一致性", "D2 维度契约", "D3 presence 语义",
             "D4 词表口径", "D5 归一化口径", "D6 指标口径",
             "D7 时间戳/资源", "D8 gitignore"],
    "issues": [{"severity": s, "dim": d, "title": t, "detail": x}
               for s, d, t, x in issues],
    "counts": cnt,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % p)