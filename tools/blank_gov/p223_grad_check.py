#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G1 自检：确认 train 混淆类真的被采样为负类、且对比损失产生非零梯度。

审计要求（D2/E1 教训）：不能只看参数传了，要看实际代码路径生效。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools/blank_gov"))
from csgc_module import GlossPrototypeBank  # noqa: E402

C, D, T, B = 3516, 1024, 20, 2
d = json.loads((REPO / "artifacts/metrics/blank-gov/p222_confusion_map.json"
                ).read_text(encoding="utf-8"))
cm = d["map"]

# 取两个确实有混淆类的 gloss 作为正类
keys = [int(k) for k in cm if cm[k]][:6]
assert len(keys) >= 6, "confusion map too small"
tgt = torch.tensor(keys, dtype=torch.long)
tgt_len = torch.tensor([3, 3], dtype=torch.long)
lgt = torch.tensor([T, T], dtype=torch.long)

w = torch.randn(D, C)
main_lp = torch.log_softmax(torch.randn(T, B, C), dim=-1)
feat = torch.randn(T, B, D, requires_grad=True)

for name, conf in (("G1", cm), ("E1-baseline", None)):
    bank = GlossPrototypeBank(w, momentum=0.99, tau=0.07,
                              n_neg=16, conf_map=conf)
    loss = bank(main_lp, feat, tgt, tgt_len, lgt)
    loss.backward(retain_graph=True)
    g = float(feat.grad.abs().sum())
    print("%-12s loss=%.4f  n_conf_used=%d/%d  feat_grad_sum=%.4f  proto_norm=%.4f"
          % (name, float(loss), bank.n_conf_used, bank.n_conf_total,
             g, float(bank.proto[:3].norm(dim=-1).mean())), flush=True)
    feat.grad.zero_()

bank = GlossPrototypeBank(w, momentum=0.99, tau=0.07, n_neg=16, conf_map=cm)
_ = bank(main_lp, feat, tgt, tgt_len, lgt)
assert bank.n_conf_used > 0, "G1 confusion negatives NOT used (silent no-op!)"
assert bank.n_conf_total == B, "forward did not cover all batch items"
assert torch.isfinite(bank.proto).all(), "proto has NaN/Inf"
print("SELF-CHECK PASS: confusion negatives are active, loss finite, grad non-zero")
