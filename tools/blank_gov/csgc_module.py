#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P212 / E1：Cross-Sentence Gloss Consistency（CSGC, AAAI 2024）最小实现。

依据：Rao et al., AAAI 2024, DOI 10.1609/aaai.v38i5.28265 ——
"a prototype is maintained for each gloss category ... gloss contrastive loss
 to pull samples to their prototype and push away from others"。

本实现要点（针对本项目约束做了简化，诚实标注）：
  - 原型表 [C, D] 用 **NormLinear 的列归一化权重** 初始化：
    cosine classifier 的权重列本身就是类原型，且覆盖全部 3515 类
    （含 48% 只出现 1 次的长尾类），避免低频类原型从未被更新的欠定问题。
  - 帧→gloss 实例用 CTC 后验在帧维 softmax 做**软聚合**，
    不依赖帧级标注或强制对齐（本项目只有句子级 gloss）。
  - 正类 = 该句真实 gloss；负类 = 后验最高的非目标类（hard negatives，
    直击 P191 的同类手势细粒度混淆）。
  - 对比损失 InfoNCE + 动量更新原型；**训练期-only，推理零开销、零新增参数进模型**。
  - 未实现原论文的 ASFS 辅助相似度分类器（需改推理，超预算）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlossPrototypeBank(nn.Module):
    def __init__(self, classifier_weight, momentum=0.99, tau=0.07,
                 n_neg=16, conf_map=None):
        """classifier_weight: [D, C] 的 NormLinear 权重（已训练）。"""
        super().__init__()
        with torch.no_grad():
            proto = F.normalize(classifier_weight.detach().float(), dim=0).t()
        self.register_buffer("proto", proto.contiguous())   # [C, D]
        self.momentum = momentum
        self.tau = tau
        self.n_neg = n_neg
        # P223/G1：train 集挖出的真实混淆类表 {gloss_id(str): [confusable_ids]}
        # None 时严格保持 E1 原行为（负类 = 后验最高的非目标类）
        self.conf_map = conf_map or {}
        self.n_conf_used = 0
        self.n_conf_total = 0

    @torch.no_grad()
    def _momentum_update(self, cls_ids, z):
        for i, g in enumerate(cls_ids):
            if g in self._pos_set:
                p = self.proto[g]
                self.proto[g] = F.normalize(
                    self.momentum * p + (1.0 - self.momentum) * z[i].detach(),
                    dim=-1)

    def forward(self, main_lp, feat, tgt, tgt_len, lgt):
        """main_lp [T,B,C] log-prob；feat [T,B,D] 主 temporal 帧特征。

        tgt 是 1-D 拼接，tgt_len 给出每句长度。返回标量对比损失。
        """
        T, B, C = main_lp.shape
        dev = main_lp.device
        losses = []
        off = 0
        for b in range(B):
            n_b = int(tgt_len[b])
            ids = [int(x) for x in tgt[off:off + n_b].tolist()]
            off += n_b
            pos = [g for g in ids if 0 < g < C]
            if not pos:
                continue
            Tb = min(T, int(lgt[b]))
            if Tb <= 0:
                continue
            lp = main_lp[:Tb, b, :]          # [Tb, C]
            f = feat[:Tb, b, :]              # [Tb, D]
            pos_set = set(pos)
            with torch.no_grad():
                # P223/G1：优先取 train 集真实混淆类（P222 帧级 top-2 竞争对），
                # 不足 n_neg 时才用 E1 原有的「后验最高非目标类」补齐。
                conf = []
                if self.conf_map:
                    for g in pos:
                        conf.extend(self.conf_map.get(str(g),
                                                     self.conf_map.get(g, [])))
                    conf = [int(x) for x in conf
                            if 0 < int(x) < C and int(x) not in pos_set]
                    conf = list(dict.fromkeys(conf))[:self.n_neg]
                self.n_conf_used += len(conf)
                self.n_conf_total += 1
                need = self.n_neg - len(conf)
                if need > 0:
                    score = lp[:, 1:].max(dim=0).values   # [C-1]
                    k = min(need * 2, score.numel())
                    cand = (score.topk(k).indices + 1).tolist()
                    extra = [g for g in cand
                             if g not in pos_set and g not in conf][:need]
                else:
                    extra = []
                neg = conf + extra
            cls = list(dict.fromkeys(pos + neg))
            if len(cls) < 2:
                continue
            cls_t = torch.as_tensor(cls, device=dev, dtype=torch.long)
            w = torch.softmax(lp[:, cls_t], dim=0)              # [Tb, |cls|]
            z = (w.unsqueeze(-1) * f.unsqueeze(1)).sum(0)       # [|cls|, D]
            z = F.normalize(z, dim=-1)
            p = F.normalize(self.proto[cls_t], dim=-1)           # [|cls|, D]
            sim = (z @ p.t()) / self.tau                          # [|cls|, |cls|]
            lab = torch.arange(len(cls), device=dev)
            losses.append(F.cross_entropy(sim, lab))
            self._pos_set = set(pos)
            self._momentum_update(cls, z)
        if not losses:
            return main_lp.new_zeros(())
        return torch.stack(losses).mean()


class FeatureCapture:
    """挂 classifier11 的 forward hook，只取**第一次**调用（主 temporal 输出）。

    TFNet 里 classifier11 与 classifier22 是同一对象，一次 forward 会触发两次；
    第 1 次输入是 outputs['predictions']（喂给 logProbs1 主输出），
    第 2 次是 conv1d 的 x。必须只取第 1 次，否则特征与 main_lp 不对齐。
    """

    def __init__(self):
        self.feat = None
        self.handle = None

    def reset(self):
        self.feat = None

    def hook(self, _module, inputs, _output):
        if self.feat is None and inputs and torch.is_tensor(inputs[0]):
            self.feat = inputs[0]

    def attach(self, model):
        self.handle = model.classifier11.register_forward_hook(self.hook)
        return self

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
