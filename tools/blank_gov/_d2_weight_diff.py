#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D2 机制诊断：一致性正则是否真的改变了模型权重？

P184 与 D2 同 init、同 seed、同配方、同 ep4 best，唯一变量是 cons 项。
若权重差异极小 ⇒ cons 项数值太小，等价于「没施加」；
若差异明显 ⇒ 正则生效但无增益（机制证伪更彻底）。
"""
import torch

P = "/home/su127/FYP/domain-bounded-cslr/artifacts/checkpoints/"
a = torch.load(P + "p184-tfnet-freeze-best.pt", map_location="cpu",
               weights_only=False)
b = torch.load(P + "p208-d2-cons-best.pt", map_location="cpu",
               weights_only=False)
print("p184 epoch:", a.get("epoch"), "wer:", a.get("wer_official"))
print("d2   epoch:", b.get("epoch"), "wer:", b.get("wer_official"))
sa, sb = a["model_state"], b["model_state"]
print("keys:", len(sa), len(sb), "same names:", set(sa) == set(sb))

rows = []
for k in sa:
    if k not in sb:
        continue
    x, y = sa[k], sb[k]
    if not (torch.is_tensor(x) and torch.is_tensor(y)) or x.shape != y.shape:
        continue
    if not x.is_floating_point():
        continue
    d = (x.float() - y.float())
    rows.append((k, float(d.abs().max()), float(d.norm() / (x.float().norm() + 1e-12)),
                 float(x.float().norm())))

rows.sort(key=lambda r: -r[2])
print("\ntop-10 relative diff:")
for k, mx, rel, xn in rows[:10]:
    print("  %-52s max|d| %.3e  rel %.3e" % (k[:52], mx, rel))
allrel = [r[2] for r in rows]
allmax = [r[1] for r in rows]
import statistics
print("\nmedian rel diff: %.3e" % statistics.median(allrel))
print("max rel diff    : %.3e" % max(allrel))
print("median max|d|   : %.3e" % statistics.median(allmax))
print("exact-identical tensors: %d / %d" %
      (sum(1 for r in rows if r[1] == 0.0), len(rows)))
