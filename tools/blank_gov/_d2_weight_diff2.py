#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D2 机制诊断（细化）：按模块分组看权重是否真的被一致性项改变。"""
from collections import defaultdict
import torch

P = "/home/su127/FYP/domain-bounded-cslr/artifacts/checkpoints/"
a = torch.load(P + "p184-tfnet-freeze-best.pt", map_location="cpu",
               weights_only=False)["model_state"]
b = torch.load(P + "p208-d2-cons-best.pt", map_location="cpu",
               weights_only=False)["model_state"]

groups = defaultdict(lambda: {"n": 0, "same": 0, "maxrel": 0.0})
for k, x in a.items():
    if k not in b or not torch.is_tensor(x) or not x.is_floating_point():
        continue
    y = b[k]
    if x.shape != y.shape:
        continue
    d = (x.float() - y.float()).abs().max().item()
    g = k.split(".")[0] if "." in k else k
    groups[g]["n"] += 1
    groups[g]["same"] += (d == 0.0)
    groups[g]["maxrel"] = max(groups[g]["maxrel"], d)

print("%-22s %6s %6s %8s  %s" % ("module", "n", "same", "same%", "max|diff|"))
for g, v in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
    print("%-22s %6d %6d %7.1f%%  %.3e" %
          (g, v["n"], v["same"], 100.0 * v["same"] / v["n"], v["maxrel"]))

print("\n所有发生变化的张量：")
chg = []
for k, x in a.items():
    if k not in b or not torch.is_tensor(x) or not x.is_floating_point():
        continue
    if x.shape != b[k].shape:
        continue
    d = (x.float() - b[k].float()).abs().max().item()
    if d > 0:
        chg.append((k, d))
print("changed:", len(chg))
for k, d in sorted(chg, key=lambda t: -t[1])[:20]:
    print("   %-56s %.3e" % (k[:56], d))
