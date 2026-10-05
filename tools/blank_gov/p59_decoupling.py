"""P59：核实「扩词表」与「特征提取」是否解耦 —— 决定正在跑的提取要不要停"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
NEW = REPO / "artifacts/part3_features_tasksapi"

files = sorted(glob.glob(str(NEW / "train" / "*.landmark.npy")))
print("=" * 72)
print("1. 特征文件里有什么")
print("=" * 72)
a = np.load(files[0])
print("  路径示例: %s" % Path(files[0]).name)
print("  shape = %s  dtype = %s" % (a.shape, a.dtype))
print("  全部数值范围: [%.3f, %.3f]" % (float(a.min()), float(a.max())))
print("  有 NaN 吗: %s" % (not np.isfinite(a).all()))

# 维度语义（与 memory 里 P38 一致）
blocks = [("hands 双手", 0, 126), ("pose 上身", 126, 158),
          ("face 面部", 158, 182), ("presence 检出", 182, 186),
          ("deltas 差分", 186, 368)]
print("\n  维度布局（纯 landmark 坐标，与词表无关）：")
for nm, s0, s1 in blocks:
    print("    [%3d:%3d] %-16s dim=%3d" % (s0, s1, nm, s1 - s0))

print("\n" + "=" * 72)
print("2. 关键判断：扩词表需要重新提特征吗")
print("=" * 72)
print("""
  特征提取做的是：视频 -> MediaPipe -> 48 帧 × 368 维坐标
  词表做的是    ：gloss 字符串 -> 词表索引 -> CTC target

  两者在数据流上不相交：
      视频 --提取--> (48,368) 特征 --查词表--> CTC target
      词表改动只影响右边那一路（target），
      左边（特征）一个字节都不用动。
  ⇒ 扩词表**不需要**重新提取特征，正在跑的提取继续跑完即可。
""")

print("  提取接口签名（只看视频路径，不接收词表）：")
svc = REPO / "app/backend/realtime_landmark.py"
for i, ln in enumerate(svc.read_text(encoding="utf-8").splitlines()):
    if "def extract_to_48x368" in ln:
        print("    %s:%d  %s" % (svc.name, i + 1, ln.strip()))
    if i > 0 and "REPO" in ln and "Path(__file__)" in ln:
        break

print("\n  已提取可复用数量：")
print("    train %d 条（断点续跑机制在，不会重跑已完成的）" % len(files))

print("\n" + "=" * 72)
print("3. 那么真正要改的只有一处")
print("=" * 72)
p40 = REPO / "tools/blank_gov/p40_rgb_main.py"
src = p40.read_text(encoding="utf-8")
for i, ln in enumerate(src.splitlines()):
    if "max_tokens" in ln or "min_frequency" in ln:
        print("  %s:%d  %s" % (p40.name, i + 1, ln.strip()))
print("""
  这两行是 argparse 默认值，改成 --max-tokens 3515 --min-frequency 1 即可。
  **不需要改任何特征文件。**
""")

out = {
    "decoupled": True,
    "feature_shape": list(a.shape),
    "feature_dtype": str(a.dtype),
    "has_nan": bool(not np.isfinite(a).all()),
    "block_layout": {nm: [s0, s1] for nm, s0, s1 in blocks},
    "already_extracted_train": len(files),
    "verdict": ("扩词表不需要重新提特征；正在跑的提取应继续跑完，"
                "之后只需改 --max-tokens 3515 --min-frequency 1"),
}
p = REPO / "artifacts/metrics/blank-gov/p59-decoupling.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % p)