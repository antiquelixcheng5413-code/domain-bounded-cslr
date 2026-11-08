#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P214/F2 补丁：给 p78 加视觉-文本语义原型对齐（--vta-weight，默认 0 关闭）。

对 NormLinear 的列归一化类原型 w_c 加 cosine 一致性正则：
  L_vta = lambda * sum_c mask_c (1 - cos(w_c, T_c)) / sum_c mask_c
T_c 是离线由 Qwen2.5-1.5B gloss 嵌入 PCA 到 1024 后固定的语义目标，
mask 对低频类加权。不引入可训练参数，推理零开销。
"""
from __future__ import annotations

from pathlib import Path

P = Path("/home/su127/FYP/domain-bounded-cslr/tools/blank_gov/p78_train_official_tfnet.py")
s = P.read_text(encoding="utf-8")
n = 0

if "--vta-weight" in s:
    print("VTA patch already applied, skip")
    raise SystemExit(0)

anchor = '    ap.add_argument("--csgc-weight", type=float, default=0.0,\n'
args = '''    ap.add_argument("--vta-weight", type=float, default=0.0,
                    help="P214/F2：视觉-文本语义原型对齐权重（0=关闭）。"
                         "训练期-only，推理零开销；用 Qwen gloss 嵌入牵引低频类原型")
    ap.add_argument("--vta-emb", default="artifacts/cache/p214_vta_target.npy",
                    help="P214/F2：语义目标 [C,1024]")
    ap.add_argument("--vta-mask", default="artifacts/cache/p214_vta_mask.npy",
                    help="P214/F2：类别权重 [C]")
'''
assert anchor in s, "arg anchor missing"
s = s.replace(anchor, args + anchor, 1)
n += 1

anchor2 = "    # P212/E1：跨句 gloss 原型对比（训练期-only）\n"
init = '''    # P214/F2：视觉-文本语义原型对齐（训练期-only，投影固定无新参数）
    VTA_TARGET = None
    VTA_MASK = None
    if a.vta_weight > 0:
        # 链保证 cwd = 仓库根；默认参数为相对路径
        _t = np.load(a.vta_emb)
        _m = np.load(a.vta_mask)
        VTA_TARGET = torch.from_numpy(_t.astype("float32")).to(dev)
        VTA_MASK = torch.from_numpy(_m.astype("float32")).to(dev)
        print("VTA: target %s mask %s  low-freq weight=1.0 classes=%d"
              % (tuple(VTA_TARGET.shape), tuple(VTA_MASK.shape),
                 int((VTA_MASK >= 0.99).sum())), flush=True)
'''
assert anchor2 in s, "init anchor missing"
s = s.replace(anchor2, init + anchor2, 1)
n += 1

anchor3 = "            opt.zero_grad()\n            loss.backward()\n"
losscode = '''            # P214/F2：类原型 -> 文本语义目标的 cosine 一致性（低频类加权）
            if VTA_TARGET is not None:
                _wn = F.normalize(model.classifier11.weight.float(), dim=0)
                _cos = (_wn * VTA_TARGET.t()).sum(0)
                _mm = VTA_MASK
                loss = loss + a.vta_weight * (
                    ((1.0 - _cos) * _mm).sum() / _mm.sum().clamp_min(1.0))
'''
assert anchor3 in s, "loss anchor missing"
s = s.replace(anchor3, losscode + anchor3, 1)
n += 1

P.write_text(s, encoding="utf-8")
print("VTA patch applied: %d edits" % n)
