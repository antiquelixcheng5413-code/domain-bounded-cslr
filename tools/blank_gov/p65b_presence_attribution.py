"""P65b：新旧presence 值差异的真实原因 —— 决定要不要继续重提"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
OLD = REPO / "artifacts/part3_features/train"
NEW = REPO / "artifacts/part3_features_tasksapi/train"

SEM = ["handL", "handR", "pose", "face"]


def frame_presence(root, limit=200, step=1):
    fs = sorted(glob.glob(str(root / "*.landmark.npy")))[::step]
    if limit:
        fs = fs[:limit]
    F = np.concatenate([np.load(f)[:, 182:186] for f in fs], 0)
    return fs, F


def report(tag, root, **kw):
    fs, F = frame_presence(root, **kw)
    print("\n【%s】%d 个文件，%d 帧" % (tag, len(fs), len(F)))
    print("  逐帧presence: handL=%.3f handR=%.3f pose=%.3f face=%.3f"
          % (F[:, 0].mean(), F[:, 1].mean(), F[:, 2].mean(), F[:, 3].mean()))
    l, r = F[:, 0] > .5, F[:, 1] > .5
    print("  双手都检出 = %.3f" % (l & r).mean())
    print("  只检出左手 = %.3f" % (l & ~r).mean())
    print("  只检出右手 = %.3f" % (~l & r).mean())
    print("  都没检出   = %.3f" % (~l & ~r).mean())
    return F


print("=" * 72)
print("presence 值差异的归因")
print("=" * 72)
Fo = report("旧特征（训练同源）", OLD, limit=200)
Fn = report("新特征（已修顺序）", NEW, limit=100)

print("\n" + "=" * 72)
print("1. handL/handR：低检出是「单手帧」还是「全检不出」")
print("=" * 72)
for tag, F in (("旧", Fo), ("新", Fn)):
    l, r = F[:, 0] > .5, F[:, 1] > .5
    single = (l ^ r).mean()
    neither = (~l & ~r).mean()
    print("  %s: 单手帧 %.3f   全无帧 %.3f   ->单手帧占检出的 %.1f%%"
          % (tag, single, neither, 100 * single / max(1 - neither, 1e-9)))

print("\n" + "=" * 72)
print("2. 🔴 face 位：新版 0.720 vs 旧版 0.998 —— 差 0.278")
print("=" * 72)
print("  旧 face 基本恒检出（holistic 一次性输出478 点脸）")
print("  新 face=0.720 -> Tasks API 的 FaceLandmarker 有 28% 帧检不出")
print("  ⇒ 这是**新版独有的问题**，与 presence 顺序无关。")
print("     face 段 24 维在检出时是全零，等于 28% 的帧丢失 6.5% 的维度。")

print("\n" + "=" * 72)
print("3. 判定：能否离线修正")
print("=" * 72)
print("  presence 值不一致（handL 差 0.348/ handR 差 0.398 / face 差 0.278）")
print("  → 不是位序问题，是**检出行为实质不同**")
print("  → 离线置换修不了")
print("\n  但！这里出现一个新判断：")
print("    新版 handL/handR 检出 0.94/0.94，旧版 0.59/0.54")
print("    —— **新版检出率反而更高**。这不是 bug，可能是 MediaPipe")
print("       Tasks API 的 HandLandmarker(num_hands=2) 比 holistic 更激进。")
print("    face 检出新版更差（0.72 vs 0.998）才是真正需要调的参数。")

print("\n" + "=" * 72)
print("4. 决策建议")
print("=" * 72)
print("""
  【选项 A】继续当前提取（已跑 %d/4973）
     理由：presence 值差异不可离线修；检出率差异是 API 固有行为，
           属于「特征来源统一」的正常代价。
     风险：face 检出 0.72 偏低，可能影响效果。

  【选项 B】先停，把 FaceLandmarker 的检出参数调好再重提
     理由：face 从 0.998 掉到 0.72 是**退化**，不该带着它跑 2 小时。
     可试：num_faces / min_detection_confidence / 输出尺寸放大。

  【选项 C】继续跑，同时并行测 face 参数
     代价：抢 CPU（提取正在占满）。

  我的建议：**B**。理由是 face 段占 24 维（6.5%），
  28% 的帧全零不是小问题；而调参数只需 5 分钟+ 抽 30 条验证。
""" % len(glob.glob(str(NEW / "*.landmark.npy"))))

out = {
    "old_frame_presence": Fo.mean(0).tolist(),
    "new_frame_presence": Fn.mean(0).tolist(),
    "old_both_hands": float(((Fo[:, 0] > .5) & (Fo[:, 1] > .5)).mean()),
    "new_both_hands": float(((Fn[:, 0] > .5) & (Fn[:, 1] > .5)).mean()),
    "old_no_hands": float(((Fo[:, 0] <= .5) & (Fo[:, 1] <= .5)).mean()),
    "new_no_hands": float(((Fn[:, 0] <= .5) & (Fn[:, 1] <= .5)).mean()),
    "finding_1": "新旧 presence 值实质不同（非位序），离线置换修不了",
    "finding_2": "新版 handL/handR 检出 0.94/0.94 > 旧版 0.59/0.54 —— 新版更高，非 bug",
    "finding_3": "🔴 新版 face 检出 0.720 << 旧版 0.998 —— **这是退化，需先调参数**",
    "recommendation": "选项 B：先停，调 FaceLandmarker 参数，抽 30 条验证后再重提",
}
p = REPO / "artifacts/metrics/blank-gov/p65b-presence-attribution.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % p)