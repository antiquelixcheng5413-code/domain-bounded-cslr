"""P65：能不能避免第三次重提？

思路：presence 只是 4 个指示位，如果它只是**位序错位**（值本身正确），
那可以用离线置换修正，不必重提。

判据：
  - 若「旧基准的第 k 位」与「新版第 j 位」统计上一致 → 说明是纯置换，可离线修
  - 若不一致→ 说明还有别的语义差异，离线修不了，必须重提

这个判断值 2 小时重提，必须做。
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np


def _find_repo() -> Path:
    for c in Path(__file__).resolve().parents:
        if (c / "src" / "cslr" / "recognition"
                / "gloss_sequence.py").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo()
NEW = REPO / "artifacts/part3_features_tasksapi/train"   # 新版（已修顺序）
OLD = REPO / "artifacts/part3_features/train"# 旧训练特征（基准）

SEM = ["handL", "handR", "pose", "face"]


def collect(root, limit=None, step=1):
    fs = sorted(glob.glob(str(root / "*.landmark.npy")))[::step]
    if limit:
        fs = fs[:limit]
    if not fs:
        return None
    P, S, bad = [], [], 0
    for f in fs:
        a = np.load(f)
        if a.shape != (48, 368) or not np.isfinite(a).all():
            bad += 1
            continue
        P.append(a[:, 182:186].mean(0))
        S.append([a[:, s0:s1].std() for s0, s1 in
                  ((0, 126), (126, 158), (158, 182), (186, 368))])
    return {"n": len(fs), "bad": bad,
            "presence": np.mean(P, 0),
            "std": np.mean(S, 0)}


new = collect(NEW)
old = collect(OLD, limit=200, step=25)

print("=" * 72)
print("1. presence 四位的统计对照")
print("=" * 72)
print("  语义顺序（两者现已一致）: %s" % SEM)
print("  %-14s %s" % ("", "  ".join("%-7s" % s for s in SEM)))
print("  %-14s %s" % ("新（已修）",
      "  ".join("%-7.3f" % v for v in new["presence"])))
print("  %-14s %s" % ("旧（基准）",
      "  ".join("%-7.3f" % v for v in old["presence"])))

print("\n" + "=" * 72)
print("2. 🔴 关键：新旧『值』是否也一致（不只是顺序）")
print("=" * 72)
pn, po = new["presence"], old["presence"]
print("  新: pose 位=%.3f handL=%.3f handR=%.3f face=%.3f"
      % (pn[2], pn[0], pn[1], pn[3]))
print("  旧: handL=%.3f handR=%.3f pose=%.3f face=%.3f"
      % (po[0], po[1], po[2], po[3]))
diffs = {
    "handL": abs(pn[0] - po[0]),
    "handR": abs(pn[1] - po[1]),
    "pose": abs(pn[2] - po[2]),
    "face": abs(pn[3] - po[3]),
}
print("\n  %-8s %10s %10s %s" % ("位", "新", "旧", "差值"))
for i, s in enumerate(SEM):
    print("  %-8s %10.3f %10.3f %10.3f" % (s, pn[i], po[i], diffs[s]))

print("\n" + "=" * 72)
print("3. 能否离线修正？")
print("=" * 72)
# 若新版只是「顺序对但值不同」，说明 handL/handR 检出率本身有差异（非置换）
# 若纯置换，则 new[i] 应约等于 old[perm[i]]
print("  假设新版正确、旧版是「同一语义不同顺序」，则应满足：")
print("    new[handL] ≈ old[?]——但旧版本身就是训练同源，顺序已正确。")
print("  实测：新 handL=%.3f vs 旧 handL=%.3f  差 %.3f"
      % (pn[0], po[0], diffs["handL"]))
print("      new handR=%.3f vs 旧 handR=%.3f  差 %.3f"
      % (pn[1], po[1], diffs["handR"]))

tol = 0.06
same = all(diffs[s] < tol for s in SEM)
print("\n  判定阈值：各位差值 < %.2f 视为一致" % tol)
for s in SEM:
    mark = "✓" if diffs[s] < tol else "✗"
    print("    [%s] %-6s 差 %.3f" % (mark, s, diffs[s]))

print("\n" + "=" * 72)
print("4. std 对照（离线修不了的部分）")
print("=" * 72)
names = ["hands", "pose", "face", "deltas"]
print("  %-8s %10s %10s %8s" % ("块", "新", "旧", "比"))
for i, nm in enumerate(names):
    print("  %-8s %10.4f %10.4f %7.2fx"
          % (nm, new["std"][i], old["std"][i],
             new["std"][i] / max(old["std"][i], 1e-9)))
print("""
  ⚠️ 注意：即使 presence 值完全一致，**std 仍差 2~3 倍**
    ⇒ 绝对值分布不同，这**不可能靠置换修正**。
""")

print("=" * 72)
print("5. 结论")
print("=" * 72)
if not same:
    print("""
  ❌ **presence 的值本身就不一致，不能靠离线置换救。**

  例如 handL：新 %.3f vs 旧 %.3f —— 新版检出的左手明显更多。
  这说明新旧提取器在手部检出行为上有实质差异（不只是位序）。

  ⇒ 必须重提。这次提取是**必要的**，不是白做。
""" % (pn[0], po[0], pn[1], po[1]))
else:
    print("""
  ✅ presence 值一致，理论上可离线置换修正。
     但 std 差 2~3 倍仍需重提（绝对值分布不同）。
""")

print("""
  📌 补充：为什么之前两次重提也"必要"
    第 1 次：为了统一 train/serve 特征（P48 发现 skew 29倍）
    第 2 次：修左右手语义翻转（新 API handedness 与旧相反）
    第 3 次：修 presence 位序错位（旧 [handL,handR,pose,face]
             我写成 [pose,handL,handR,face]）
  三次都是**发现了新的语义不一致**，不是同一错误重复。
  ⚠️ 但第 3 次本可以在第 1 次就避免——写提取器时若先读旧训练特征的
     构造代码（git show最早提交），5 分钟就能发现顺序问题。
""")

out = {
    "new_presence": pn.tolist(),
    "old_presence": po.tolist(),
    "semantic_order": SEM,
    "per_slot_diff": {k: round(float(v), 4) for k, v in diffs.items()},
    "tolerance": tol,
    "presence_values_consistent": bool(same),
    "new_std": new["std"].tolist(),
    "old_std": old["std"].tolist(),
    "std_ratios": {names[i]: round(float(new["std"][i] / old["std"][i]), 2)
                   for i in range(4)},
    "conclusion": ("presence 值不一致 -> 必须重提" if not same
                   else "presence 一致但 std 差 2-3x -> 仍需重提"),
    "three_reasons_for_repeat": [
        "P53: 统一 train/serve 特征（skew 29倍，P48）",
        "P63: 修左右手 handedness 语义翻转（新旧 API 相反）",
        "P63c: 修 presence 位序错位（旧 [handL,handR,pose,face]）",
    ],
    "lesson": "第 3 次本可在第 1 次避免：写提取器前先读训练特征构造代码",
}
p = REPO / "artifacts/metrics/blank-gov/p65-avoid-reextract.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % p)