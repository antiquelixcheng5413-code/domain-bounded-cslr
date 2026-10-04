# -*- coding: utf-8 -*-
"""终极定位：decode_batch 拿到的 log_probs 到底是什么。

P20dbg2 已证明：
  手工对 `arr[r, :t]` 在 axis=1 做 argmax -> 解码几乎全对
  `decode_batch(arr, ol, 1)`                     -> 解码全错
而 `greedy_decode` 源码逻辑正确。

唯一剩下的可能：**传给 decode_batch 的数组形状/内容与预期不同**。
本脚本打印 arr 的真实 shape，并直接调用 greedy_decode 看它拿到什么。
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/su127/FYP/domain-bounded-cslr/src")
from cslr.recognition.decode import greedy_decode, classes_to_token_ids

# 构造一个已知的、与 P20dbg2 相同情形的小例子
# 模拟：48 帧，真实序列 [5, 32, 26, 1]（target_ids）
T, C = 48, 302
rng = np.random.default_rng(0)
arr = np.full((T, C), -12.0)
# 让 blank 占绝大多数
arr[:, 0] = -1e-4
# 在第 5,10,20,30 帧放上目标 label，置信度极高
for step, cls in zip([5, 10, 20, 30], [5, 32, 26, 1]):
    arr[step, :] = -12.0
    arr[step, cls] = -1e-5
arr[:, 0] = -1e-4

print("arr.shape =", arr.shape, " 期望 [B=1, T=48, C=302] 或 [T=48, C=302]")

# 情况1：decode_batch 传入 arr[0, :48]（[T,C]）—— 正确用法
ok = greedy_decode(arr[0, :48])
print()
print("用法1  greedy_decode(arr[0,:48])  -> classes =", ok.classes)
print("       token_ids =", classes_to_token_ids(ok.classes))

# 情况2：误传整个 arr（[1,48,302]）—— 若发生，argmax(axis=1) 会在 C 轴上取
try:
    bad = greedy_decode(arr)
    print()
    print("用法2  greedy_decode(arr) 形状越界检查 ->", bad.classes)
except Exception as e:
    print()
    print("用法2  greedy_decode(arr) -> RAISES:", e)

# 关键：如果 batch 里 B>1，decode_batch 逐 row 取 log_probs[row, :length]
# 而 length 来自 ol。这里 arr 只有 1 行，所以看不出问题。
# 现在构造 B=2 的情况，验证 ol 与 T 的关系
print()
print("=" * 70)
print("验证 decode_batch 逐行切片在 B>1 时是否错位")
print("=" * 70)
B = 2
arr2 = rng.normal(size=(B, T, C)) * 0.1
arr2[:, :, 0] = 5.0          # blank 最高
arr2[0, 5, 7] = 9.0          # 第一条第 5 帧是 class 7
arr2[1, 9, 3] = 9.0          # 第二条第 9 帧是 class 3
lens = [48, 48]
for row in range(B):
    steps = arr2[row, :lens[row]]
    d = greedy_decode(steps)
    print("  row {}: shape={} -> classes={}".format(row, steps.shape, d.classes))
print()
print("若上面 classes 出现意外大长度/怪值，说明切片维度错了")
