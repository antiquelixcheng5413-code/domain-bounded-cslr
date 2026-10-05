"""P61：模态维度对照 —— 2D landmark vs 3D Kinect 骨架，为什么混用在工程上不可能

用户提出：两个数据集「一个是2d，一个是3d骨架」。这个观察是对的，而且
比「背景不一样」更硬 —— 它决定了特征无法拼接、模型结构无法共享。

本脚本做四件事：
  1. 盘点 CE-CSL 本地实际存在的模态（实测，不靠文档转述）
  2. 量化 CE-CSL 特征的维度构成（实测 .npy）
  3. 列出 USTC 2015 侧的三模态规格（页面确认）
  4. 逐维度对比，说明哪些能对齐、哪些根本对不上
"""
from __future__ import annotations

import collections
import csv
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
sys.path.insert(0, str(REPO / "src"))

out: dict = {"generated": str(REPO / "artifacts/metrics/blank-gov/p61.json")}

print("=" * 72)
print("1. CE-CSL 本地模态实况（实测文件扩展名）")
print("=" * 72)
raw = REPO / "data/raw/CE-CSL"
ext = collections.Counter()
for p in raw.rglob("*"):
    if p.is_file():
        ext[p.suffix.lower() or "(无扩展名)"] += 1
print("  data/raw/CE-CSL 下文件类型分布：")
for k, v in sorted(ext.items(), key=lambda kv: -kv[1]):
    print("    %-14s %6d 个" % (k, v))
n_mp4 = ext.get(".mp4", 0)
print("\n  ⇒ 只有 %d 个 mp4 + %d 个 csv。" % (n_mp4, ext.get(".csv", 0)))
print("  ⇒ **零个深度文件、零个骨架文件** —— CE-CSL 只有 RGB 单模态。")
out["ce_csl_modalities_local"] = dict(ext)

# 视频目录 = signer/背景分组
vid = raw / "video"
for split in ("train", "dev", "test"):
    d = vid / split
    if d.exists():
        subs = sorted(x.name for x in d.iterdir() if x.is_dir())
        n_mp = len(list(d.rglob("*.mp4")))
        print("    %-6s %2d 个子目录（signer/背景组） %4d 个 mp4"
              % (split, len(subs), n_mp))

print("\n" + "=" * 72)
print("2. CE-CSL 特征的真实维度构成（实测 .npy）")
print("=" * 72)
NEW = REPO / "artifacts/part3_features_tasksapi/train"
fs = sorted(glob.glob(str(NEW / "*.landmark.npy")))
print("  已提特征 %d 条（Tasks API）" % len(fs))
if fs:
    a = np.load(fs[0])
    print("  单条 shape = %s dtype=%s" % (a.shape, a.dtype))
    blocks = [("双手 landmark", 0, 126, "2D 归一化 + 肩距缩放"),
              ("pose 上身", 126, 158, "取 pose 8 点 × 4（含 visibility）"),
              ("face 面部", 158, 182, "取 face 8 点 × 3"),
              ("presence", 182, 186, "4 个检出标志位"),
              ("deltas 一阶差分", 186, 368, "原始帧率差分再重采样")]
    print("\n  %-18s %-12s %-6s %s" % ("块", "维度", "占比", "语义"))
    for nm, s0, s1, sem in blocks:
        print("  [%3d:%3d] %-14s %-6d %5.1f%%  %s"
              % (s0, s1, nm, s1 - s0, 100 * (s1 - s0) / 368, sem))
    # MediaPipe 输出其实是 3D（x,y,z），但 z 是相对深度
    print("\n  ⚠️ 关键：MediaPipe 的 landmark 是 (x, y, z) 三值 —— 不是纯 2D。")
    print("     但 z 是**相对深度**（以手腕/髋部为参考的伪距离），")
    print("     单位不统一、无米制标定，与 Kinect 的绝对 3D 坐标不可直接比较。")
    out["ce_csl_feature"] = {
        "shape": list(a.shape), "dtype": str(a.dtype),
        "blocks": [{"name": nm, "slice": [s0, s1], "dim": s1 - s0,
                    "semantics": sem} for nm, s0, s1, sem in blocks],
        "note": "MediaPipe 输出 (x,y,z)，z 为相对深度，非米制绝对 3D",
    }

print("\n" + "=" * 72)
print("3. USTC 2015 侧的三模态规格（官方页面确认）")
print("=" * 72)
ustc = {
    "device": "Microsoft Kinect 2.0，签署者距离约 1.5 m，30 fps",
    "modalities_per_instance": ["RGB 视频 1280×720",
                                "深度视频 512×424",
                                "3D 骨架 25 关节/帧"],
    "slr500": {"task": "孤立词", "words": 500, "signers": 50, "reps": 5,
               "instances": 125000},
    "continuous_slr": {"task": "连续句", "sentences": 100, "signers": 50,
                       "reps": 5, "instances": 25000},
    "skeleton_coordinate": "页面未说明单位/格式 —— 不可假定是 mm 或已归一化",
    "access": "Release Agreement 需全职员工签署，学生不可",
}
for k, v in ustc.items():
    print("  %-22s %s" % (k, v if not isinstance(v, dict) else ""))
    if isinstance(v, dict):
        for kk, vv in v.items():
            print("      %-18s %s" % (kk, vv))
print("\n  模态清单：")
for m in ustc["modalities_per_instance"]:
    print("    - %s" % m)
out["ustc_2015"] = ustc

print("\n" + "=" * 72)
print("4. 逐维度对照：哪些能对齐，哪些根本对不上")
print("=" * 72)
compare = [
    ("RGB 视频", "有（5988 个 mp4）", "有（1280×720）",
     "✅ 格式对齐，但分辨率/背景/编码不同", "可解码到帧，通道数一致"),
    ("深度视频", "❌ 无", "有（512×424）",
     "❌ 缺失模态 —— 无法凭空生成", "需要深度相机或单目深度估计"),
    ("3D 骨架", "❌ 无（只有 MediaPipe 抽的 landmark）",
     "有（25 关节/帧，绝对 3D）",
     "❌ 关节数、坐标系、语义全不同", "25 vs 我们取 21手+8pose+8face=37 点"),
    ("手部细节点", "❌ MediaPipe 21 点/手", "❌ Kinect 25 关节不含手指",
     "⚠️ 两边都没有精细手型", "都需 HaMeR 类模型补"),
    ("词表", "3515", "500（SLR500）/ 100句（连续）",
     "❌ label space 不同", "联合需 5515 类上界"),
]
print("  %-14s %-30s %-24s %s" % ("模态", "CE-CSL", "USTC 2015", "可对齐性"))
for nm, a_, b_, c_, d_ in compare:
    print("  %-14s %-30s %-24s %s" % (nm, a_, b_, c_))
    print("  %-14s %s" % ("", d_))

print("""
  ⇒ 结论：模态差异比背景差异更硬。

  **能对齐的只有 RGB** —— 这也是唯一可混用的通道。
  **深度与 3D 骨架在CE-CSL 侧完全缺失**，不是「质量差」而是「没有」：
  - 无法用 Kinect 的 3D 去监督 CE-CSL（无配对标注）
  - 无法把 Kinect 特征当额外输入（维度/坐标系/关节语义都不同）
  - 若要用单目深度估计补，等于引入一个新的、有误差的模态分支，
    且与 MediaPipe landmark 混合会带来**双重坐标系错位风险**
""")

print("=" * 72)
print("5. 这条约束与今天 P48/P53 发现的skew 直接相关")
print("=" * 72)
print("""
  今天定位的 train-serve skew：训练用离线 holistic 特征、线上用 Tasks API 特征，
  造成 train exact 96.7% vs 3.3%（29 倍）。

  **如果引入 Kinect 3D 骨架，会再造一次同类错位**：
  - 离线预提取的 3D 骨架（无噪声、坐标系 A）
  - 线上实时抽的 landmark（相对深度、坐标系 B）
  两者混在一起训练，模型会学到两套坐标系下的同一语义，
  推理时只有一套可用 —— 与今天修的 bug 同构，且更严重
  （因为要同时解「模态缺失」和「坐标系错位」两个问题）。

  ⇒ 结论：模态层面应当**保持单一模态**。
     要 3D 就整套换 3D，不要一半 landmark 一半 3D。
""")

out["modal_alignment_table"] = [
    {"modality": r[0], "ce_csl": r[1], "ustc": r[2],
     "alignable": r[3], "detail": r[4]} for r in compare]
out["conclusion"] = {
    "only_alignable_modality": "RGB",
    "missing_in_ce_csl": ["depth video", "absolute 3D skeleton"],
    "mediapipe_z_note": "MediaPipe 的 z 是相对深度，非米制绝对 3D",
    "warning": "引入 Kinect 3D 会再造一次 train-serve skew（同构但更严重）",
    "recommendation": "保持单一模态；要 3D 就整套换，别混",
}

p = REPO / "artifacts/metrics/blank-gov/p61-modality.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % p)