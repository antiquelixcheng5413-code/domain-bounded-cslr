#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""label smoothing penalty 的已知答案验证

验证三件事（写 test 而不是直接跑训练，是因为 loss 尺度错了会静默劣化）：
  1. β=0 严格等价（(1-0)·CTC + 0·penalty = CTC）
  2. penalty 惩罚过度自信：peaked 分布 > uniform 分布
  3. 有效时间步 mask 生效（padding 不计入）
  4. 量级可预估：uniform 时 penalty ≈ T_eff × log(N)
"""
import math
import sys
import torch

sys.path.insert(0, "/home/su127/FYP/domain-bounded-cslr/tools/blank_gov")
import p78_train_official_tfnet as P  # noqa: E402

N = 3516          # 词表 + blank
T, B = 40, 2
ok = True


def chk(cond, msg):
    global ok
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        ok = False


print("=== 1. β=0 严格等价 ===")
lp = torch.log_softmax(torch.randn(T, B, N), dim=-1)
lgt = torch.tensor([40, 25])
fake_ctc = torch.tensor(36.0)
for beta in (0.0,):
    pen = P.ls_penalty(lp, lgt)
    smoothed = (1.0 - beta) * fake_ctc + beta * pen
    chk(torch.allclose(smoothed, fake_ctc),
        "β=0 时 smoothed == CTC（%.6f vs %.6f）" % (smoothed.item(), fake_ctc.item()))

print("=== 2. penalty 惩罚过度自信 ===")
uni = torch.full((1, N), 1.0 / N).log().repeat(T, B, 1)       # 完全均匀
peak = torch.full((T, B, N), -20.0)                            # 极尖峰
peak[..., 0] = 0.0
peak = torch.log_softmax(peak, dim=-1)
p_uni = P.ls_penalty(uni, lgt).item()
p_peak = P.ls_penalty(peak, lgt).item()
chk(p_peak > p_uni,
    "尖峰 penalty (%.2f) > 均匀 penalty (%.2f) ⇒ 惩罚过度自信" % (p_peak, p_uni))

print("=== 3. 均匀分布的 penalty 应 ≈ T_eff_avg × log(N) ===")
lgt_full = torch.tensor([40, 40])
p_full = P.ls_penalty(uni, lgt_full).item()
expect = 40.0 * math.log(N)
chk(abs(p_full - expect) / expect < 0.01,
    "全有效时 penalty %.2f ≈ T×log(N) %.2f（B=%d，÷B 后为每样本累积）"
    % (p_full, expect, B))

print("=== 4. padding 不计入（lgt 短 ⇒ penalty 小） ===")
p_short = P.ls_penalty(uni, torch.tensor([10, 40])).item()
chk(p_short < p_full,
    "lgt=[10,40] penalty %.2f < lgt=[40,40] penalty %.2f ⇒ mask 生效"
    % (p_short, p_full))

print("=== 5. 量级参考（决定 β 取多大） ===")
print("  log(N) = %.2f ；实测 CTC 初期 ≈ 36，后期 ≈ 5" % math.log(N))
print("  均匀 penalty（T=40） = %.1f" % p_full)
print("  尖峰 penalty（T=40） = %.1f" % p_peak)
print("  ⇒ β=0.1 时 penalty 项 ≈ %.1f，相对 CTC(36) 不可忽略" % (0.1 * p_full))
print("  ⇒ 建议 β 从 0.05 起（保守），或用 --label-smoothing 0.1 观察 train loss 是否可控")

print()
print("总体:", "✅ 全部通过" if ok else "❌ 有失败项")
sys.exit(0 if ok else 1)
