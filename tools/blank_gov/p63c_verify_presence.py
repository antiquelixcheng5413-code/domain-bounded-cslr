"""P63c：presence 顺序修复的实测验证（不看代码猜）

对照基准：旧特征（训练同源，presence 顺序已知正确）
判据：新特征的 presence 均值应与旧基准在同一量级，且
      **第3 位（pose）应接近 1.0**（旧特征实测恒为 1.000）
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
NEW = REPO / "artifacts/part3_features_tasksapi/train"
OLD = REPO / "artifacts/part3_features/train"

SEM = ["handL", "handR", "pose", "face"]


def stats(root, step=1, limit=None):
    fs = sorted(glob.glob(str(root / "*.landmark.npy")))
    if limit:
        fs = fs[:limit]
    fs = fs[::step]
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
    F = np.concatenate([np.load(f)[:, 182:186] for f in fs[:60]], 0)
    return {
        "n": len(fs), "bad": bad,
        "presence_mean": np.mean(P, 0).round(4).tolist(),
        "hands_sum": float(np.mean(P, 0)[0] + np.mean(P, 0)[1]),
        "std_blocks": np.mean(S, 0).round(4).tolist(),
        "both_hands_frame": float(((F[:, 0] > .5) & (F[:, 1] > .5)).mean()),
        "no_hand_frame": float(((F[:, 0] < .5) & (F[:, 1] < .5)).mean()),
    }


print("=" * 72)
print("presence 顺序修复验证（语义顺序 = [handL, handR, pose, face]）")
print("=" * 72)

new = stats(NEW, step=1, limit=40)
old = stats(OLD, step=40)

print("\n%-16s %-34s" % ("", "  ".join("%-7s" % s for s in SEM)))
print("%-16s %s" % ("新（修复后）",
      "  ".join("%-7.3f" % v for v in new["presence_mean"])))
print("%-16s %s" % ("旧（基准）",
      "  ".join("%-7.3f" % v for v in old["presence_mean"])))
print("\n%-16s %-34s" % ("", "  ".join("%-7s" % s for s in SEM)))
print("%-16s %s" % ("新 双手和", "%.3f" % new["hands_sum"]))
print("%-16s %s" % ("旧 双手和", "%.3f" % old["hands_sum"]))

print("\n各块 std 对照：")
names = ["hands", "pose", "face", "deltas"]
for i, nm in enumerate(names):
    ratio = new["std_blocks"][i] / max(old["std_blocks"][i], 1e-9)
    print("  %-8s 新 %.4f  旧 %.4f  比 %.2fx"
          % (nm, new["std_blocks"][i], old["std_blocks"][i], ratio))

print("\n逐帧手部检出：")
print("  双手都检出 新 %.3f  旧 %.3f" % (new["both_hands_frame"],
                                       old["both_hands_frame"]))
print("  双手都无    新 %.3f  旧 %.3f" % (new["no_hand_frame"],
                                       old["no_hand_frame"]))

#判定
checks = [
    ("pose 位（第3位）应接近 1.0", new["presence_mean"][2] > 0.9),
    ("pose 位不再恒为第1位", new["presence_mean"][0] < 0.95),
    ("双手和 与旧基准同量级（0.8~2.2）",
     0.8 <= new["hands_sum"] <= 2.2),
    ("双手都检出帧 > 0.35", new["both_hands_frame"] > 0.35),
    ("无手帧 < 0.10（旧基准 0.000）", new["no_hand_frame"] < 0.10),
    ("hands std 比值 < 3.5x",
     new["std_blocks"][0] / old["std_blocks"][0] < 3.5),
    ("无 shape/NaN 异常", new["bad"] == 0),
]
print("\n" + "=" * 72)
print("判定")
print("=" * 72)
allok = True
for nm, ok in checks:
    print("  [%s] %s" % ("✓" if ok else "✗", nm))
    allok = allok and ok
print("\n=> %s" % ("**presence 修复已验证**" if allok
                  else "**仍有偏差，需继续排查**"))

out = {
    "semantic_order": SEM,
    "new": new, "old": old,
    "checks": [{"name": n, "pass": o} for n, o in checks],
    "all_pass": allok,
}
p = REPO / "artifacts/metrics/blank-gov/p63c-presence-verify.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("收据-> %s" % p)