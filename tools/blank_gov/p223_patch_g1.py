#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P223/G1 补丁：给 p78 加 --csgc-confusion（默认空，严格保持 E1 原行为）。

G1 与 E1 唯一差别：CSGC 的负类来源从「后验最高的非目标类」换成
P222 从 train 集挖出的真实混淆类（帧级 top-2 竞争对，无 dev 泄漏）。
"""
from __future__ import annotations

from pathlib import Path

P = Path("/home/su127/FYP/domain-bounded-cslr/tools/blank_gov/p78_train_official_tfnet.py")
s = P.read_text(encoding="utf-8")

if "--csgc-confusion" in s:
    print("G1 patch already applied, skip")
    raise SystemExit(0)

anchor = '    ap.add_argument("--csgc-weight", type=float, default=0.0,\n'
arg = '''    ap.add_argument("--csgc-confusion", default="",
                    help="P223/G1：train 混淆对表 json（confusion map）。"
                         "空=严格保持 E1 原行为；给定时 CSGC 负类优先取真实混淆类")
'''
assert anchor in s, "arg anchor missing"
s = s.replace(anchor, arg + anchor, 1)

old = '''        CSGC_BANK = GlossPrototypeBank(
            model.classifier11.weight, momentum=a.csgc_momentum,
            tau=a.csgc_tau, n_neg=a.csgc_nneg).to(dev)'''
new = '''        _cm = {}
        if a.csgc_confusion:
            import json as _json
            _d = _json.loads(Path(a.csgc_confusion).read_text(encoding="utf-8"))
            _cm = _d.get("map", _d)
            print("G1 confusion map: %d gloss / %d pairs <- %s"
                  % (len(_cm), sum(len(v) for v in _cm.values()),
                     a.csgc_confusion), flush=True)
        CSGC_BANK = GlossPrototypeBank(
            model.classifier11.weight, momentum=a.csgc_momentum,
            tau=a.csgc_tau, n_neg=a.csgc_nneg, conf_map=_cm).to(dev)'''
assert old in s, "bank creation anchor missing"
s = s.replace(old, new, 1)

P.write_text(s, encoding="utf-8")
print("G1 patch applied: 2 edits")
