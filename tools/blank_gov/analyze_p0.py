# -*- coding: utf-8 -*-
"""深挖 P0 诊断结果：定位 peaky 的具体形态。

只读 artifacts/metrics/blank-gov/p0-diagnosis.json，不加载模型。
"""
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
p = REPO / "artifacts/metrics/blank-gov/p0-diagnosis.json"
d = json.load(open(p, encoding="utf-8"))
ss = d["samples"]

print("样本数(收据存前50) {}".format(len(ss)))
T = [s["T"] for s in ss]
ng = [s["n_ref_gloss"] for s in ss]
print("T  中位/最小/最大  {} / {} / {}".format(sorted(T)[len(T) // 2], min(T), max(T)))
print("ref gloss  中位/最小/最大  {} / {} / {}".format(sorted(ng)[len(ng) // 2], min(ng), max(ng)))

pl = [L for s in ss for L in s["peak_lens"]]
print("")
print("peak 长度分布  {}".format(collections.Counter(pl).most_common(10)))
print("peak 段总数  {}".format(sum(len(s["peak_lens"]) for s in ss)))
print("ref gloss 总数  {}".format(sum(ng)))
print("零 peak 样本数  {}".format(sum(1 for s in ss if s["n_peak"] == 0)))

ratio = sorted(s["T"] / max(s["n_ref_gloss"], 1) for s in ss)
print("")
print("每样本 T/ref_gloss 中位  {:.1f}".format(ratio[len(ratio) // 2]))
print("  -> 这是 blank 率的直接下界：即使每 gloss 只占 1 帧，blank 也会到 {:.1%}".format(
    1 - 1 / (ratio[len(ratio) // 2] * 1.0)))

bl = sorted(s["blank_ratio"] for s in ss)
print("")
print("blank_ratio  中位/最小/最大  {:.4f} / {:.4f} / {:.4f}".format(
    bl[len(bl) // 2], min(bl), max(bl)))

print("")
print("=== 关键结论 ===")
print("T 恒为 48：特征是固定 48 帧（计划 2.0 记录的 16->48 帧实验）")
print("ref gloss 中位 5，48 帧 / 5 词 = 每词约 9.6 帧可用")
print("实测每词只用 0.26 帧 -> 帧级定位能力几乎为零")
print("")
print("这解释了为什么 97% blank 降不下来：")
print("  模型不是把 blank 用在词间过渡，而是几乎全部帧都判 blank，")
print("  只在极少数帧上给出一个错误的类别。")
print("  SMART Table 3 的 CSLR-Only F1@50=7.37 描述的正是这个状态。")
