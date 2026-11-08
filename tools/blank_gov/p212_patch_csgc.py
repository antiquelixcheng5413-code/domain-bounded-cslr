#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给 p78 加 E1/CSGC 开关（默认关闭，不改变任何已有行为）。"""
from pathlib import Path

P = Path("/home/su127/FYP/domain-bounded-cslr/tools/blank_gov/"
         "p78_train_official_tfnet.py")

ARG_ANCHOR = '''    ap.add_argument("--adapter", default="none", choices=["none", "conformer"],'''
ARG_NEW = '''    ap.add_argument("--csgc-weight", type=float, default=0.0,
                    help="P212/E1：跨句 gloss 原型对比损失权重（0=关闭）。"
                         "训练期-only，推理零开销；原型用 NormLinear 权重初始化")
    ap.add_argument("--csgc-momentum", type=float, default=0.99,
                    help="P212/E1：gloss 原型动量")
    ap.add_argument("--csgc-tau", type=float, default=0.07,
                    help="P212/E1：InfoNCE 温度")
    ap.add_argument("--csgc-nneg", type=int, default=16,
                    help="P212/E1：hard negative 数量")
    ap.add_argument("--adapter", default="none", choices=["none", "conformer"],'''

INIT_ANCHOR = '''    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)'''
INIT_NEW = '''    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    # P212/E1：跨句 gloss 原型对比（训练期-only）
    global CSGC_BANK, CSGC_CAP
    CSGC_BANK, CSGC_CAP = None, None
    if a.csgc_weight > 0:
        from csgc_module import GlossPrototypeBank, FeatureCapture
        CSGC_BANK = GlossPrototypeBank(
            model.classifier11.weight, momentum=a.csgc_momentum,
            tau=a.csgc_tau, n_neg=a.csgc_nneg).to(dev)
        CSGC_CAP = FeatureCapture().attach(model)
        print("CSGC: proto %s, momentum %.3f, tau %.3f, nneg %d"
              % (tuple(CSGC_BANK.proto.shape), a.csgc_momentum,
                 a.csgc_tau, a.csgc_nneg), flush=True)'''

LOSS_ANCHOR = '''            opt.zero_grad()
            loss.backward()'''
LOSS_NEW = '''            # P212/E1：跨句 gloss 原型对比损失（训练期-only）
            if CSGC_BANK is not None and CSGC_CAP.feat is not None:
                _cl = CSGC_BANK(main_lp, CSGC_CAP.feat, tgt, tgt_len, lgt)
                loss = loss + a.csgc_weight * _cl
            opt.zero_grad()
            loss.backward()'''

# forward 前重置 hook，保证只取本 step 的第一次 classifier11 调用
FWD_ANCHOR = '''            with torch.amp.autocast("cuda", enabled=a.amp_bf16,
                                    dtype=torch.bfloat16):
                out = model(vid, dl, True)'''
FWD_NEW = '''            if CSGC_CAP is not None:
                CSGC_CAP.reset()
            with torch.amp.autocast("cuda", enabled=a.amp_bf16,
                                    dtype=torch.bfloat16):
                out = model(vid, dl, True)'''

s = P.read_text(encoding="utf-8")
for anchor in (ARG_ANCHOR, INIT_ANCHOR, FWD_ANCHOR, LOSS_ANCHOR):
    assert s.count(anchor) == 1, "anchor count != 1: %r" % anchor[:50]
assert "--csgc-weight" not in s, "already patched"
s = s.replace(ARG_ANCHOR, ARG_NEW, 1)
s = s.replace(INIT_ANCHOR, INIT_NEW, 1)
s = s.replace(FWD_ANCHOR, FWD_NEW, 1)
s = s.replace(LOSS_ANCHOR, LOSS_NEW, 1)
P.write_text(s, encoding="utf-8")
print("patched OK; csgc-weight count =", s.count("--csgc-weight"))
