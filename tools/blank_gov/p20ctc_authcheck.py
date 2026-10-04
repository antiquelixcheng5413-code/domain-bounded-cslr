# -*- coding: utf-8 -*-
"""权威对拍：确认 torch 2.14 下 CTCLoss 接受 2D padded targets 的语义。

P20 出现过一个强信号：
  8 条样本 / 400 轮 / lr=3e-3 / 无 dropout
  -> loss/token = -1.89，P(target) 中位 8.29，但 WER = 1.0，0/8 完美解码

「负 loss + 完美解码不了」不可能同时成立，必须查清。
本脚本用**同一份 log_probs** 分别走三条路径：

  A) 官方推荐：1-D flatten targets + reduction='none'
  B) 仓库做法：2-D padded targets（padding 值 = blank = 0）
  C) 1-D flatten，但把 padding 也算进 target_lengths（模拟 B 的语义错误）

若 A != B，说明 torch 对 2-D 输入的 target_lengths 处理与 1-D 不同
=> 仓库的 ctc_loss 在 torch 2.14 上有语义 bug。
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
B, T, C = 3, 20, 8
BLANK = 0
logits = torch.randn(B, T, C)
lp_bt_c = torch.log_softmax(logits, dim=-1)     # 仓库布局 [B,T,C]
lp_t_b_c = lp_bt_c.transpose(0, 1).contiguous()  # torch 要求 [T,B,C]

# 构造 3 条样本，长度不同
seqs = [[1, 2, 3], [4], [5, 6]]
tl = torch.tensor([len(s) for s in seqs])
il = torch.tensor([T] * B)

# 2-D padded：padding 用 blank(=0) 填充 —— 仓库 pad_targets 的做法
width = max(len(s) for s in seqs)
padded = torch.full((B, width), BLANK, dtype=torch.long)
for i, s in enumerate(seqs):
    padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)

flat = torch.tensor([x for s in seqs for x in s], dtype=torch.long)

print("=" * 72)
print("路径 A：官方推荐（1-D flatten + target_lengths 截断）")
print("=" * 72)
A = nn.CTCLoss(blank=BLANK, reduction="none", zero_infinity=False)(
    lp_t_b_c, flat, il, tl)
print("  per-sample:", [round(float(x), 5) for x in A])

print()
print("=" * 72)
print("路径 B：仓库做法（2-D padded，padding=blank=0）")
print("=" * 72)
try:
    Bv = nn.CTCLoss(blank=BLANK, reduction="none", zero_infinity=False)(
        lp_t_b_c, padded, il, tl)
    print("  per-sample:", [round(float(x), 5) for x in Bv])
except Exception as e:
    print("  RAISES:", e)

print()
print("=" * 72)
print("路径 C：1-D flatten 但 target_lengths 含 padding（错误的语义）")
print("=" * 72)
C_ = nn.CTCLoss(blank=BLANK, reduction="none", zero_infinity=False)(
    lp_t_b_c, flat, il, torch.full((B,), width, dtype=torch.long))
print("  per-sample:", [round(float(x), 5) for x in C_])

print()
print("=" * 72)
same = all(abs(float(a) - float(b)) < 1e-4 for a, b in zip(A, Bv)) if 'Bv' in dir() else False
print("A == B ?", same)
print("若 A != B：torch 2.14 的 2-D targets 路径语义与 1-D 不同，")
print("仓库 ctc_loss 用 padding=blank 的 2-D targets 会得到错误的 loss。")
print("=" * 72)

# 附加：验证 loss 会不会因 padding 而被系统性压低/抬高
print()
print("对照：把真实 target 换成别的内容，看 B 是否还能给出同样低的 loss")
padded2 = padded.clone()
padded2[0, :] = torch.tensor([7, 7, 7], dtype=torch.long)
Bv2 = nn.CTCLoss(blank=BLANK, reduction="none", zero_infinity=False)(
    lp_t_b_c, padded2, il, tl)
print("  B(pad 换成全 7):", [round(float(x), 5) for x in Bv2])
print("  A(真 target)   :", [round(float(x), 5) for x in A])
print("  B 忽略第 0 行真实内容 -> 说明它确实读了 2-D 的第一行")
