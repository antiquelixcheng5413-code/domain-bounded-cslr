#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P160：回答「为什么 R/G/B 顺序会影响性能？2+1 不是等于 1+2 吗」

理论（第一层卷积，逐点写）
    y  = w_R*R + w_G*G + w_B*B
    输入换成 BGR 顺序后：
    y' = w_R*B + w_G*G + w_B*R
    Δ  = y - y' = (w_R - w_B) * (R - B)              ★ 关键公式

⇒ Δ = 0 只有两种情况：
    ① w_R == w_B ：该滤波器是「灰度型」，三通道权重相同 ⇒ 通道顺序无所谓
    ② R   == B   ：该像素是灰的
「加法可交换」没错，但交换加数 ≠ 交换被不同系数乘的输入：
    (2*a + 1*b) ≠ (2*b + 1*a)     当 a ≠ b

本脚本实测：
  1. 预训练 conv1 的红/蓝切片差多少（灰度型 vs 色拮抗型各多少）
  2. 我们的帧到底有多少颜色（是不是接近灰度 —— 这决定上面公式的量级）
  3. 把输入 R/B 互换后，conv1 特征 / resnet18 全网络特征变了多少
"""
import os

import cv2
import numpy as np
import torch

torch.set_num_threads(2)

REPO = "/home/su127/FYP/domain-bounded-cslr"
FR = REPO + "/artifacts/official_rgb/dev"


def cos01(a, b):
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


# ---------- 1. 预训练 conv1：R vs B 切片 ----------
from torchvision.models import resnet18, ResNet18_Weights  # noqa: E402

net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
net.eval()
W = net.conv1.weight.detach().numpy()          # [64,3,7,7]
wR, wB = W[:, 0], W[:, 2]
c_RB = cos01(wR, wB)
num = np.linalg.norm((wR - wB).reshape(64, -1), axis=1)
den = np.linalg.norm(W.reshape(64, -1), axis=1) + 1e-12

print("=" * 64)
print("1) ImageNet 预训练 resnet18 conv1（64 个 7x7 滤波器）")
print("=" * 64)
print("  cos(w_R, w_B): 平均 %+.3f  中位 %+.3f  最小 %+.3f"
      % (c_RB.mean(), np.median(c_RB), c_RB.min()))
print("  ‖w_R - w_B‖/‖w‖ 平均 %.3f" % (num / den).mean())
print("  灰度型 (cos>0.8): %2d/64  ← 换通道无所谓" % (c_RB > 0.8).sum())
print("  中间型          : %2d/64" % (((c_RB >= 0.2) & (c_RB <= 0.8)).sum()))
print("  色拮抗 (cos<0.2): %2d/64  ← 换通道输出就变" % (c_RB < 0.2).sum())

# ---------- 2. 我们的帧含多少颜色 ----------
imgs = []
for root, _d, files in os.walk(FR):
    jpgs = sorted([f for f in files if f.lower().endswith(".jpg")])
    if jpgs:
        imgs += [os.path.join(root, f) for f in jpgs[:2]]
    if len(imgs) >= 40:
        break
imgs = imgs[:40]
if not imgs:
    print("\n!! 没找到帧: %s" % FR)
    raise SystemExit(0)

diffs, lum_all, chan = [], [], []
for p in imgs:
    im = cv2.imread(p).astype(float)          # cv2 约定 [B,G,R]
    b, g, r = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    d = r - b
    L = 0.299 * r + 0.587 * g + 0.114 * b
    diffs.append(d.ravel())
    lum_all.append(L.ravel())
    chan.append(np.stack([r.mean(), g.mean(), b.mean()]))
D = np.concatenate(diffs)
LUM = np.concatenate(lum_all)
C = np.array(chan).mean(0)

print()
print("=" * 64)
print("2) dev 帧（%d 张）的颜色含量" % len(imgs))
print("=" * 64)
print("  通道均值  R %.1f  G %.1f  B %.1f" % (C[0], C[1], C[2]))
print("  (R-B)  均值 %+.1f   平均|R-B| %.1f" % (D.mean(), np.abs(D).mean()))
print("  |R-B| 分位数:  50%% %.0f   90%% %.0f   99%% %.0f   max %.0f"
      % tuple(np.percentile(np.abs(D), [50, 90, 99, 100])))
fg = np.abs(D)[LUM > np.percentile(LUM, 75)]      # 亮处≈前景/手
print("  亮区(|R-B|, 前25%%亮度): 平均 %.1f   90%%分位 %.0f"
      % (fg.mean(), np.percentile(fg, 90)))
print("  ⇒ 对照：纯灰画面 |R-B|=0；肤色典型 |R-B|≈60~80")

# ---------- 3. 互换 R/B 后特征变多少 ----------
conv = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    conv.weight.copy_(torch.from_numpy(W))

head = torch.nn.Sequential(*list(net.children())[:8])   # conv1..layer3(avg前)
head.eval()

rel_conv, rel_net, coss = [], [], []
for p in imgs[:12]:
    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)      # → [R,G,B]
    im = cv2.resize(im, (224, 224))
    t = torch.from_numpy(im).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
    ts = t.flip(1)                                            # ★ 交换通道维
    with torch.no_grad():
        y = torch.relu(conv(t))
        ys = torch.relu(conv(ts))
        f = head(t).flatten()
        fs = head(ts).flatten()
    rel_conv.append(((y - ys).norm() / (y.norm() + 1e-12)).item())
    rel_net.append(((f - fs).norm() / (f.norm() + 1e-12)).item())
    coss.append(float(torch.nn.functional.cosine_similarity(f, fs, dim=0)))

print()
print("=" * 64)
print("3) 输入互换 R/B 后，预训练主干的特征变化（12 张）")
print("=" * 64)
print("  conv1(ReLU后)  ‖Δy‖/‖y‖ : 平均 %5.1f%%  最大 %5.1f%%"
      % (100 * np.mean(rel_conv), 100 * np.max(rel_conv)))
print("  resnet18 主干特征 ‖Δf‖/‖f‖: 平均 %5.1f%%  最大 %5.1f%%"
      % (100 * np.mean(rel_net), 100 * np.max(rel_net)))
print("  主干特征 cos(f, f_swap)   : 平均 %+.4f  （1.0=完全没差别）" % np.mean(coss))
