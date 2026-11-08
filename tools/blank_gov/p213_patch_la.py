#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P213/F1 补丁：给 p78 加训练期 logit adjustment（--la-tau，默认 0 关闭）。

la(lp) = log_softmax(lp - tau * log pi_c)，blank 不动；推理时完全不调整。
只改训练 loss 用的 log-prob，main_lp / out[*] 原值保持不变，
因此 CSGC / 一致性 / word_head 等既有开关的行为完全不受影响。
"""
from __future__ import annotations

import re
from pathlib import Path

P = Path("/home/su127/FYP/domain-bounded-cslr/tools/blank_gov/p78_train_official_tfnet.py")
s = P.read_text(encoding="utf-8")
n = 0

if "--la-tau" in s:
    print("LA patch already applied, skip")
    raise SystemExit(0)

# 1) 参数
anchor = '    ap.add_argument("--csgc-weight", type=float, default=0.0,\n'
new_args = '''    ap.add_argument("--la-tau", type=float, default=0.0,
                    help="P213/F1：训练期 logit adjustment 强度（0=关闭，不改变默认行为）。"
                         "la(lp)=log_softmax(lp - tau*log pi_c)，blank 不动，推理零开销")
    ap.add_argument("--la-pi-power", type=float, default=1.0,
                    help="P213/F1：类先验指数（1.0=标准 LA，<1 更温和）")
'''
assert anchor in s, "arg anchor missing"
s = s.replace(anchor, new_args + anchor, 1)
n += 1

# 2) 初始化（ctc_loss 定义之后）
anchor2 = '    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)\n'
init = '''    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    # P213/F1：训练期 logit adjustment（Menon ICLR21 / Balanced Softmax）
    #   pi_c 走训练实际路径 DPM.PreWords -> word2idx，与 D4 频次表同口径。
    #   la_tau=0 时 la_lp 原样返回，严格等价默认行为。
    LA_LOGPI = None
    if a.la_tau > 0:
        _tr = read_csv_split("train.csv")
        _cnt = np.zeros(wordSetNum + 1, dtype=np.float64)
        for _sid, (_p, _seq) in _tr.items():
            for _t in DPM.PreWords(str(_seq).split("/")):
                _g = word2idx.get(_t)
                if _g is not None and 0 <= _g < len(_cnt):
                    _cnt[_g] += 1.0
        _cnt[1:] = np.maximum(_cnt[1:], 1.0)
        _pi = _cnt[1:] / _cnt[1:].sum()
        _logpi = np.zeros(wordSetNum + 1, dtype=np.float32)
        _logpi[1:] = (a.la_pi_power * np.log(_pi)).astype(np.float32)
        LA_LOGPI = torch.from_numpy(_logpi).to(dev).view(1, 1, -1)
        print("LA-CTC: tau=%.3f pi_power=%.2f  logpi[1:] min=%.3f max=%.3f"
              % (a.la_tau, a.la_pi_power, float(_logpi[1:].min()),
                 float(_logpi[1:].max())), flush=True)

    def la_lp(lp):
        """输入已是 log-prob [T,B,C]；只在训练 loss 上使用。"""
        if LA_LOGPI is None:
            return lp
        z = lp.float() - a.la_tau * LA_LOGPI.to(lp.device, torch.float32)
        return torch.log_softmax(z, dim=-1).to(lp.dtype)
'''
assert anchor2 in s, "ctc_loss anchor missing"
s = s.replace(anchor2, init, 1)
n += 1

# 3) 主 CTC per-sample
old3 = "            _ctc_per = ctc_loss(main_lp, tgt, lgt, tgt_len.cpu())\n"
new3 = "            _ctc_per = ctc_loss(la_lp(main_lp), tgt, lgt, tgt_len.cpu())\n"
assert old3 in s, "main ctc anchor missing"
s = s.replace(old3, new3, 1)
n += 1

# 4) 辅助 CTC 头（TFNet 4 个 + 非 TFNet 1 个），保证先验校正不被辅助头抵消
pat = r"ctc_loss\(ls\((out\[\d\])\), tgt, lgt, tgt_len\.cpu\(\)\)"
s, k = re.subn(pat, r"ctc_loss(la_lp(ls(\1)), tgt, lgt, tgt_len.cpu())", s)
assert k >= 3, "expected >=3 auxiliary CTC calls, got %d" % k
n += k

P.write_text(s, encoding="utf-8")
print("LA patch applied: %d edits (%d aux CTC calls)" % (n, k))
