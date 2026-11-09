#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给 p78 加默认关闭的 --o4-* 开关（P242/O4 区间级监督）。

4 处精确替换；每处 assert 唯一，失败即报错不静默。
默认 --o4-weight 0 ⇒ 严格保持已有行为。
"""
from pathlib import Path

P = Path("/home/su127/FYP/domain-bounded-cslr/tools/blank_gov/p78_train_official_tfnet.py")
s = P.read_text(encoding="utf-8")
n = 0

# ---- 1) 参数区 ----
old1 = '''    ap.add_argument("--k1-manifest", default="",
                    help="P233/K1：合成训练样本 manifest json；空=严格保持原行为")'''
new1 = old1 + '''
    ap.add_argument("--o4-weight", type=float, default=0.0,
                    help="P242/O4：静态 Viterbi 区间级帧级表征监督权重（0=关闭）")
    ap.add_argument("--o4-tau", type=float, default=0.07,
                    help="P242/O4：cosine CE 温度")
    ap.add_argument("--o4-clips", default="artifacts/cache/p230_k1_clips.json",
                    help="P242/O4：K1 A 阶段 CTC 强制对齐片段库（只含 train）")'''
assert s.count(old1) == 1, "anchor1 count=%d" % s.count(old1)
s = s.replace(old1, new1, 1); n += 1

# ---- 2) main() 内、训练循环前的初始化 ----
old2 = '''    hist = list(prev_hist)
    t0 = time.time()'''
new2 = '''    # ---- P242/O4：训练期 Viterbi 区间级监督（默认关闭） ----
    O4_BANK = None
    O4_CAP = None
    o4_interval_loss = None
    if a.o4_weight > 0:
        from o4_interval_module import IntervalBank
        from o4_interval_module import interval_loss as o4_interval_loss
        from csgc_module import FeatureCapture as O4FeatureCapture
        O4_BANK = IntervalBank(a.o4_clips, word2idx, str(RGB_ROOT))
        O4_CAP = O4FeatureCapture().attach(model)
        print("O4: %s | lambda=%.3f tau=%.3f" %
              (O4_BANK.summary(), a.o4_weight, a.o4_tau), flush=True)

''' + old2
assert s.count(old2) == 1, "anchor2 count=%d" % s.count(old2)
s = s.replace(old2, new2, 1); n += 1

# ---- 3) forward 前 reset hook ----
old3 = '''            if CSGC_CAP is not None:
                CSGC_CAP.reset()'''
new3 = old3 + '''
            if O4_CAP is not None:
                O4_CAP.reset()'''
assert s.count(old3) == 1, "anchor3 count=%d" % s.count(old3)
s = s.replace(old3, new3, 1); n += 1

# ---- 4) loss 组装末尾（VTA 之后、zero_grad 之前） ----
old4 = '''                loss = loss + a.vta_weight * (
                    ((1.0 - _cos) * _mm).sum() / _mm.sum().clamp_min(1.0))
            opt.zero_grad()'''
new4 = '''                loss = loss + a.vta_weight * (
                    ((1.0 - _cos) * _mm).sum() / _mm.sum().clamp_min(1.0))
            # P242/O4：静态 Viterbi 区间级帧级表征监督（训练期-only，推理零开销）
            # 只在 classifier11 输入特征上加 cosine-CE；不动 CTC 输出 / blank / 类先验；
            # 目标原型取当前分类头权重并 detach ⇒ 不新增可训练参数。
            if O4_BANK is not None and O4_CAP is not None and O4_CAP.feat is not None:
                _W = F.normalize(model.classifier11.weight.detach().float().t(), dim=1)
                _il, _nfr = o4_interval_loss(O4_CAP.feat, lgt, _sids,
                                             O4_BANK, _W, a.o4_tau)
                loss = loss + a.o4_weight * _il
            opt.zero_grad()'''
assert s.count(old4) == 1, "anchor4 count=%d" % s.count(old4)
s = s.replace(old4, new4, 1); n += 1

P.write_text(s, encoding="utf-8")
print("O4 patch applied: %d edits -> %s" % (n, P))
