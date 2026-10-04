# -*- coding: utf-8 -*-
"""临时：独立实现 CTC 前向，与 nn.CTCLoss 对拍，定位仓库 ctc_loss 的问题。"""
import sys

import torch
import torch.nn as nn

torch.manual_seed(0)
B, T, C = 2, 10, 5
logits = torch.randn(B, T, C)
lp_batch_time = torch.log_softmax(logits, dim=-1)      # [B,T,C] 仓库的布局
lp = lp_batch_time.transpose(0, 1)                    # [T,B,C] torch 要求

tgt = torch.tensor([[1, 2, 0], [3, 0, 0]])
tl = torch.tensor([2, 1])
il = torch.tensor([10, 10])


def forward_logp(lp_tbc, il, tl, tgt, blank=0):
    """标准 CTC 前向，对 2D padded targets 按 target_lengths 截断。"""
    T_, B_, C_ = lp_tbc.shape
    out = []
    for b in range(B_):
        L = int(tl[b])
        target = tgt[b, :L].tolist()
        S = len(target)
        neg = torch.tensor(-1e30)
        # alpha[t, s]
        alpha = torch.full((int(il[b]) + 1, 2 * S + 1), -1e30)
        alpha[0, 0] = 0.0
        for t in range(1, int(il[b]) + 1):
            for s in range(2 * S + 1):
                emit = (s % 2 == 0) and s > 0
                stay = 0
                if s % 2 == 1:
                    # 刚读入一个 label，只能从 blank 转移
                    if s >= 1:
                        stay = alpha[t - 1, s - 1] + lp_tbc[t - 1, b, blank]
                else:
                    stay = alpha[t - 1, s] + lp_tbc[t - 1, b, blank]
                if emit:
                    lab = target[s // 2 - 1]
                    cands = [stay, alpha[t - 1, s]]
                    if s >= 2 and target[s // 2 - 1] != target[s // 2 - 2]:
                        cands.append(alpha[t - 1, s - 2] + lp_tbc[t - 1, b, lab])
                    v = cands[0]
                    for c in cands[1:]:
                        v = torch.logsumexp(torch.stack([v, c]), 0)
                    alpha[t, s] = v
                else:
                    alpha[t, s] = stay
        # 末端可以停在任意 s（末尾是 blank 或 label 都可）
        finals = [alpha[int(il[b]), s] for s in range(2 * S + 1)
                  if s % 2 == 0 or (s % 2 == 1 and s == 2 * S - 1)]
        v = finals[0]
        for c in finals[1:]:
            v = torch.logsumexp(torch.stack([v, c]), 0)
        out.append(float(v))
    return out


ref = forward_logp(lp, il, tl, tgt)
got = nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)(lp, tgt, il, tl)
print("独立前向 logP :", [round(x, 4) for x in ref], "  -> loss", [round(-x, 4) for x in ref])
print("nn.CTCLoss    :", [round(float(x), 4) for x in got.tolist()])
print()
print("仓库 ctc_loss 传的是 [B,T,C] 且做了 transpose -> 与本测试的 lp 相同")
print("结论：一致" if all(abs(a - float(b)) < 1e-3 for a, b in zip(ref, got)) else "结论：不一致 —— 仓库实现有 BUG")

# 关键测试：把 target_lengths 全部设为 S（padded 也算进 target）
print()
print("=" * 60)
print("测试 padded 值(=blank=0)被误当作真实 target 时的行为")
tgt2 = torch.tensor([[1, 2, 0], [3, 0, 0]])
tl_wrong = torch.tensor([3, 3])   # 错误：把 padding 也算进目标
got_wrong = nn.CTCLoss(blank=0, reduction="none")(lp, tgt2, il, tl_wrong)
print("target_lengths 正确(2,1):", [round(float(x), 4) for x in got.tolist()])
print("target_lengths 错误(3,3):", [round(float(x), 4) for x in got_wrong.tolist()])
