#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P242 / O4：静态 CTC Viterbi 区间级帧级表征监督（训练期-only）。

区间只来自 train（K1 A 阶段的 p230_k1_clips.json），用**比例映射**
t = round(frac * T_conv_b) 避开 MAX_FRAMES 下采样造成的帧索引错位。
只在 classifier11 输入特征上加帧级 cosine-CE，不动 CTC 输出 / blank / 类先验，
不新增任何可训练参数。预注册：docs/planning/O4-PREREG-2026-11-09.md。
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


class IntervalBank:
    """按 sid 反查 K1 片段库，返回 [(frac_start, frac_end, gloss_id), ...]。"""

    def __init__(self, clips_path, word2idx, rgb_root):
        p = Path(clips_path)
        data = json.loads(p.read_text(encoding="utf-8"))
        clips = data["clips"]
        root = Path(rgb_root) / "train"
        n_jpg, by_sid, n_pairs, n_missing = {}, defaultdict(list), 0, 0

        for g, rows in clips.items():
            gi = word2idx.get(g)
            if gi is None or gi <= 0:
                n_missing += 1
                continue
            for item in rows:
                src, a, b = str(item[0]), int(item[1]), int(item[2])
                if src not in n_jpg:
                    d = root / src
                    n_jpg[src] = (len([f for f in os.listdir(d) if f.endswith(".jpg")])
                                  if d.exists() else 0)
                n = n_jpg[src]
                if n <= 0:
                    continue
                sid = src.split("/")[-1]
                by_sid[sid].append((a / n, min(b / n, 1.0), int(gi)))
                n_pairs += 1

        self.by_sid = dict(by_sid)
        self.n_sids = len(self.by_sid)
        self.n_intervals = n_pairs
        self.n_missing_gloss = n_missing
        self.n_jpg_dirs = len(n_jpg)
        self.mean_iv_per_sid = (n_pairs / max(self.n_sids, 1))

    def summary(self):
        return ("sids=%d intervals=%d mean_iv/sid=%.2f dirs=%d missing_gloss=%d"
                % (self.n_sids, self.n_intervals, self.mean_iv_per_sid,
                   self.n_jpg_dirs, self.n_missing_gloss))

    def get(self, sid):
        return self.by_sid.get(str(sid), ())


def interval_loss(feat, lgt, sids, bank, W, tau=0.07):
    """区间内每帧的 cosine 分类 CE。

    feat : [T,B,D]  classifier11 输入特征（bf16/fp32 均可）
    lgt  : [B]      每样本编码器时间步
    sids : list[str]
    W    : [C,D]    固定分类头权重（已 detach）
    返回 (loss, n_frames_used)；无区间时返回零损失（保持图连通）。
    """
    T, B, D = feat.shape
    feats, targets = [], []
    lens = lgt.view(-1).tolist()
    n_frames = 0
    for b in range(B):
        ivs = bank.get(sids[b])
        if not ivs:
            continue
        Tb = min(int(lens[b]), T)
        if Tb <= 1:
            continue
        for fa, fb, g in ivs:
            t0 = min(Tb - 1, max(0, int(round(fa * Tb))))
            t1 = min(Tb, max(t0 + 1, int(round(fb * Tb))))
            feats.append(feat[t0:t1, b, :].float())
            targets.append(torch.full((t1 - t0,), g, dtype=torch.long,
                                      device=feat.device))
            n_frames += t1 - t0
    if not feats:
        return feat.sum() * 0.0, 0
    X = F.normalize(torch.cat(feats, dim=0), dim=1)
    Wn = F.normalize(W.float(), dim=1)
    logits = (X @ Wn.t()) / max(tau, 1e-6)
    Y = torch.cat(targets, dim=0)
    return F.cross_entropy(logits, Y), n_frames
