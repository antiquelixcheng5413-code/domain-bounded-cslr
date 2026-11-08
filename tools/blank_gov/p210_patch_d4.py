#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给 p78 加 D4 类平衡样本加权开关（默认关闭，不改变任何已有行为）。

机制：CTCLoss(reduction="none") 给出 per-sample loss，按句内 gloss 的
train 频次稀有度加权，再归一化到均值 1（不改变整体 lr 尺度）。
零新增参数、推理零开销。

注意（诚实边界）：这不是 FGXM 的视觉-语言一致性 loss，也不是 PGFLoss 的
pose-guided focal；它是**低频实词重加权**的最便宜近似，依据等级【间接】。
"""
from pathlib import Path

P = Path("/home/su127/FYP/domain-bounded-cslr/tools/blank_gov/"
         "p78_train_official_tfnet.py")

# ── 1) 参数
ARG_ANCHOR = '''    ap.add_argument("--adapter", default="none", choices=["none", "conformer"],'''
ARG_NEW = '''    ap.add_argument("--cb-alpha", type=float, default=0.0,
                    help="P210/D4：类平衡样本加权强度（0=关闭，不改变默认行为）。"
                         "按句内 gloss 的 train 频次稀有度给 CTC loss 加权，"
                         "权重归一化到均值 1；针对低频实词识别差")
    ap.add_argument("--adapter", default="none", choices=["none", "conformer"],'''

# ── 2) 训练循环：per-sample CTC 加权
OLD = '''            main_lp = ls(out[0])
            loss_ctc = ctc_loss(main_lp, tgt, lgt, tgt_len.cpu()).mean()'''
NEW = '''            main_lp = ls(out[0])
            _ctc_per = ctc_loss(main_lp, tgt, lgt, tgt_len.cpu())
            if a.cb_alpha > 0 and CB_FREQ is not None:
                _ws = []
                _flat = tgt.view(-1).tolist()
                _off = 0
                for _b in range(_ctc_per.shape[0]):
                    _n = int(tgt_len[_b])
                    _ids_b = _flat[_off:_off + _n]
                    _off += _n
                    _inv = [1.0 / ((CB_FREQ[g] + 1.0) ** 0.5)
                            for g in _ids_b if 0 < g < len(CB_FREQ)]
                    _ws.append(float(np.mean(_inv)) if _inv else 0.0)
                _w = np.asarray(_ws, dtype=np.float64)
                _w = _w / max(_w.mean(), 1e-8)
                _w = torch.as_tensor(1.0 + a.cb_alpha * (_w - 1.0),
                                     dtype=_ctc_per.dtype,
                                     device=_ctc_per.device)
                loss_ctc = (_ctc_per * _w).mean()
            else:
                loss_ctc = _ctc_per.mean()'''

# ── 3) 词频全局表：在训练循环开始前构建
INIT_ANCHOR = '''    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)'''
INIT_NEW = '''    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    # P210/D4：train 标注的 gloss 频次表（按 word2idx 索引），供类平衡加权用
    CB_FREQ = None
    if a.cb_alpha > 0:
        try:
            # 🔴 走训练实际路径：DPM.PreWords(gloss.split("/")) → word2idx，
            #    与 RGBSeqDataset.__init__ 的 ids 构造逐字一致。
            #    不能用原始字符串（最2/很1/帮助（我）等变体词条会被归一化，
            #    按原始串查表会漏掉 434 类、1703 次出现）。
            _tr = read_csv_split("train.csv")
            _cnt = np.zeros(wordSetNum + 1, dtype=np.float64)
            _n_miss = 0
            for _sid, (_p, _seq) in _tr.items():
                for _t in DPM.PreWords(str(_seq).split("/")):
                    _gi = word2idx.get(_t)
                    if _gi is not None and 0 <= _gi < len(_cnt):
                        _cnt[_gi] += 1.0
                    else:
                        _n_miss += 1
            CB_FREQ = _cnt
            print("CB_FREQ: non-zero %d / %d, median %.1f, min %.1f, "
                  "miss(含表头/UNK)=%d"
                  % (int((_cnt > 0).sum()), len(_cnt),
                     float(np.median(_cnt[_cnt > 0])) if (_cnt > 0).any() else 0,
                     float(_cnt[_cnt > 0].min()) if (_cnt > 0).any() else 0,
                     _n_miss), flush=True)
        except Exception as _e:
            print("CB_FREQ build FAILED: %r (fallback: no weighting)" % _e,
                  flush=True)
            CB_FREQ = None'''

s = P.read_text(encoding="utf-8")
assert s.count(ARG_ANCHOR) == 1, "arg anchor"
assert s.count(OLD) == 1, "loss anchor"
assert s.count(INIT_ANCHOR) == 1, "init anchor"
assert "--cb-alpha" not in s, "already patched"
s = s.replace(ARG_ANCHOR, ARG_NEW, 1)
s = s.replace(INIT_ANCHOR, INIT_NEW, 1)
s = s.replace(OLD, NEW, 1)
P.write_text(s, encoding="utf-8")
print("patched OK; cb-alpha count =", s.count("--cb-alpha"))
