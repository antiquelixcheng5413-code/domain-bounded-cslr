#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给 p78 加 D2 一致性正则开关（默认关闭，不改变任何已有行为）。

严格两处替换，替换前校验锚点唯一，替换后打印 diff 位置供审计。
"""
from pathlib import Path

P = Path("/home/su127/FYP/domain-bounded-cslr/tools/blank_gov/"
         "p78_train_official_tfnet.py")

ARG_ANCHOR = '''    ap.add_argument("--adapter", default="none", choices=["none", "conformer"],'''

ARG_NEW = '''    ap.add_argument("--cons-weight", type=float, default=0.0,
                    help="P208/D2：多视图一致性正则权重（0=关闭，不改变默认行为）。"
                         "两个视图各做一次前向，主 CTC 仍只用第一视图；"
                         "对称 KL 按非 blank 帧加权，避免 CTC 峰值分布里 blank 主导")
    ap.add_argument("--cons-noise", type=float, default=0.05,
                    help="P208/D2：第二视图的高斯扰动幅度（输入已归一化到 [-1,1]）")
    ap.add_argument("--adapter", default="none", choices=["none", "conformer"],'''

LOSS_ANCHOR = '''                loss = loss + a.aux_ww * (_bce + _wc)
            opt.zero_grad()'''

LOSS_NEW = '''                loss = loss + a.aux_ww * (_bce + _wc)
            # P208 / D2：多视图一致性正则（R-Drop / CR-CTC 风格）
            # 第二视图 = 输入加轻微高斯扰动；不改帧数 ⇒ CTC 对齐与 lgt 完全不变。
            # 对称 KL 用 detach 的对方分布作 target（标准 R-Drop 做法，双向都有梯度）；
            # 按非 blank 帧加权，避免 CTC 峰值分布中 ~87% blank 帧主导 KL。
            if a.cons_weight > 0:
                _vid2 = (vid + a.cons_noise * torch.randn_like(vid)).clamp(-1.0, 1.0)
                with torch.amp.autocast("cuda", enabled=a.amp_bf16,
                                        dtype=torch.bfloat16):
                    _out2 = model(_vid2, dl, True)
                _out2 = [(o.float() if (torch.is_tensor(o) and o.is_floating_point())
                          else o) for o in _out2]
                _lp2 = ls(_out2[0])
                if torch.isfinite(_lp2).all():
                    _p1 = main_lp.detach().exp()
                    _p2 = _lp2.detach().exp()
                    _T = min(main_lp.shape[0], _lp2.shape[0])
                    _wnb = 1.0 - 0.5 * (_p1[:_T, :, 0] + _p2[:_T, :, 0])
                    _msk = (torch.arange(_T, device=main_lp.device)[:, None]
                            < lgt.to(main_lp.device).long()[None, :])
                    _kl1 = F.kl_div(_lp2[:_T], _p1[:_T], reduction="none").sum(-1)
                    _kl2 = F.kl_div(main_lp[:_T], _p2[:_T], reduction="none").sum(-1)
                    _kl = 0.5 * (_kl1 + _kl2)
                    _w = (_wnb * _msk.float()).detach()
                    _cons = (_kl * _w).sum() / _w.sum().clamp_min(1.0)
                    loss = loss + a.cons_weight * _cons
            opt.zero_grad()'''

s = P.read_text(encoding="utf-8")
assert s.count(ARG_ANCHOR) == 1, "arg anchor count != 1"
assert s.count(LOSS_ANCHOR) == 1, "loss anchor count != 1"
assert "--cons-weight" not in s, "already patched"
s = s.replace(ARG_ANCHOR, ARG_NEW, 1)
s = s.replace(LOSS_ANCHOR, LOSS_NEW, 1)
P.write_text(s, encoding="utf-8")
print("patched OK; cons switch count =", s.count("--cons-weight"))
