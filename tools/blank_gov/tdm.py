# -*- coding: utf-8 -*-
"""TDM 多尺度时序差分模块（SignShift Sec 3.3，式 1-6）。

文献依据（已逐条核实原文，见下）：
  SignShift (arXiv, MM'26) Sec 3.3 "Temporal Difference Module (TDM)"

  式(1) 多尺度差分，偏移集 D = {1, 2, 4, ..., d_max}
       X_Δ^d(t) = x_t - x_{t-d}   (t > d)，否则 0
  式(2) 共享 1x1 Conv 投影后跨尺度平均
       X_Δ = (1/|D|) Σ_{d∈D} Proj(X_Δ^d)，Proj 为**共享** 1x1 conv
  式(3) 可学门控  G = σ(Conv1D(X_Δ))
  式(4) 门控残差  X̂_Δ = G ⊙ X_Δ + Proj(X)
       —— 残差保留原始语义，门控突出有显著时序变化的帧
  式(5) 手/脸流各自独立做同样的多尺度差分
  式(6) 三流融合，可学标量门 α_h、α_f
       X̂ = 1/3 [ Proj(X̂_Δ) + α_h·Proj_h(X̂^h_Δ) + α_f·Proj_f(X̂^f_Δ) ]

  超参：d_max = 16（Sec 4.1 "a stack of dilated 1D convolutions with
        maximum dilation d_max = 16"）
  消融：Table 2 —— TDM 单独贡献 F1@50 +2.48 ~ +5.79

迁移到本项目的关键判断（必须写明）：
  论文的三流是 global video / hand / face，hand 用 HaMeR、face 用 ResNet-18，
  **都需要 RGB**。本项目只有 MediaPipe landmark，**没有 RGB 帧**。
  但 landmark 的 368 维本身就按部位分块（见 dataset.py FEATURE_BLOCKS）：
      hands  [0:126]    双手 21x3
      pose   [126:158]  上半身 8x4
      face   [158:182]  面部 8x3
      masks  [182:186]  4 个存在性 mask
      hand_deltas  [186:312]
      body_deltas  [312:368]
  所以**部位流可以由 landmark 的分块直接给出**，不需要 HaMeR/ResNet-18。
  这是本实现与论文的最大差异，必须在收据里标注。

  三流对应：
      global 流 <- 全 368 维
      hand 流    <- hands(126) + hand_deltas(126) = 252 维
      face 流    <- face(24)
  注意：论文 face 流用 BlazeFace 裁剪 + ResNet-18，本项目只有 8 个点。
  **face 流维度很小（24 维），质量必然不如论文**，这是已知的迁移损失。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# SignShift Sec 4.1：d_max = 16
DEFAULT_DMAX = 16
# 论文式(1) 的偏移集 {1, 2, 4, ..., d_max}
DEFAULT_OFFSETS = (1, 2, 4, 8, 16)


def multi_scale_difference(x: torch.Tensor, offsets=DEFAULT_OFFSETS) -> list[torch.Tensor]:
    """式(1)：多尺度帧间差分。

    x: (B, T, D) -> 长度 len(offsets) 的 [(B, T, D), ...]
    t <= d 的位置置 0（论文式(1) 的分段定义）。
    """
    out = []
    for d in offsets:
        diff = torch.zeros_like(x)
        if x.shape[1] > d:
            diff[:, d:, :] = x[:, d:, :] - x[:, :-d, :]
        out.append(diff)
    return out


class TDM(nn.Module):
    """Temporal Difference Module（式 2-4，单流）。

    流程：多尺度差分 -> 共享 1x1 Conv -> 跨尺度平均 -> 门控残差。
    论文明确 Proj 是**共享**的（式 2 括号："Proj(·) denotes a shared 1×1
    convolution"），所以这里用一个 Conv1d 作用于所有尺度。
    """

    def __init__(self, in_dim: int, hidden: int = 128, offsets=DEFAULT_OFFSETS):
        super().__init__()
        self.offsets = tuple(offsets)
        # 式(2) 共享投影
        self.proj = nn.Conv1d(in_dim, hidden, kernel_size=1)
        # 式(3) 门控
        self.gate = nn.Conv1d(hidden, hidden, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, hidden)"""
        diffs = multi_scale_difference(x, self.offsets)      # 式(1)
        # 式(2)：共享 1x1 conv + 跨尺度平均
        proj = [self.proj(d.transpose(1, 2)) for d in diffs]
        x_delta = torch.stack(proj, dim=0).mean(dim=0)      # (B, hidden, T)
        # 式(3) 门控
        g = torch.sigmoid(self.gate(x_delta))
        # 式(4) 门控残差：残差项用原序列投影，保留原始语义
        residual = self.proj(x.transpose(1, 2))
        out = g * x_delta + residual
        return out.transpose(1, 2)                          # (B, T, hidden)


class MultiStreamTDM(nn.Module):
    """式(5)(6)：global / hand / face 三流 TDM + 可学习标量门融合。

    严格按论文式(6)：
        X̂ = 1/3 [ Proj(X̂_Δ) + α_h·Proj_h(X̂^h_Δ) + α_f·Proj_f(X̂^f_Δ) ]

    其中 `X̂^h_Δ` 是**hand 维度**上的差分增强表示（式 5 的 TDM(X^h)），
    经 `Proj_h`（1x1 conv，hand_dim -> hidden）投到共享隐空间。
    α_h、α_f 是**可训练标量**（论文原文："α_h and α_f are trainable scalar gates
    that control the relative contribution of the hand and face streams"）。
    论文没有给初始化值，此处用 1.0（等权起步）。

    实现要点：`TDM` 内部已含 `Proj`（式 2）与残差（式 4），输出为 hidden 维。
    式(6) 里的 `Proj_h` / `Proj_f` 是**额外**的 1x1 conv，把 hand/face 的
    TDM 输出再对齐到共享隐空间 —— 因为各流通用的 hidden 宽度可能与
    原始维度不同。切勿把 hidden 维输出直接喂进按 hand_dim 建的 conv。
    """

    def __init__(self, global_dim: int, hand_dim: int, face_dim: int,
                 hidden: int = 128, offsets=DEFAULT_OFFSETS,
                 use_hand: bool = True, use_face: bool = True):
        super().__init__()
        self.use_hand = use_hand
        self.use_face = use_face
        # global 流：TDM 内含 Proj(式2)+门控残差(式3,4)，已是 hidden 维
        self.tdm_g = TDM(global_dim, hidden, offsets)
        if use_hand:
            self.tdm_h = TDM(hand_dim, hidden, offsets)
            # 式(6) 的 Proj_h：hidden -> hidden（对齐共享隐空间）
            self.proj_h = nn.Conv1d(hidden, hidden, kernel_size=1)
            self.alpha_h = nn.Parameter(torch.ones(1))
        if use_face:
            self.tdm_f = TDM(face_dim, hidden, offsets)
            self.proj_f = nn.Conv1d(hidden, hidden, kernel_size=1)
            self.alpha_f = nn.Parameter(torch.ones(1))
        self.out_dim = hidden

    def forward(self, x_g: torch.Tensor, x_h=None, x_f=None) -> torch.Tensor:
        """式(6)。缺流时按实际路数归一，保持论文的等权语义。"""
        terms = [self.tdm_g(x_g)]
        n = 1
        if self.use_hand and x_h is not None:
            tdm_h = self.tdm_h(x_h)                          # (B,T,hidden)
            terms.append(self.alpha_h *
                         self.proj_h(tdm_h.transpose(1, 2)).transpose(1, 2))
            n += 1
        if self.use_face and x_f is not None:
            tdm_f = self.tdm_f(x_f)
            terms.append(self.alpha_f *
                         self.proj_f(tdm_f.transpose(1, 2)).transpose(1, 2))
            n += 1
        return sum(terms) / float(n)


# ----------------------------------------------------------------------
# 本项目 landmark 368 维的部位切分（对齐 dataset.py FEATURE_BLOCKS）
# ----------------------------------------------------------------------
HANDS_SLICE = slice(0, 126)
POSE_SLICE = slice(126, 158)
FACE_SLICE = slice(158, 182)
HAND_DELTAS_SLICE = slice(186, 312)
BODY_DELTAS_SLICE = slice(312, 368)

HAND_DIM = 126 + 126      # hands + hand_deltas = 252
FACE_DIM = 24


def split_streams(x: torch.Tensor):
    """把序列切成三流。

    两种输入布局：

    1. **原始 landmark (B, T, 368)** —— 用于直接对特征做 TDM：
         global = 全 368 维
         hand   = hands(126) + hand_deltas(126) = 252 维
         face   = face(24)
       论文三流对应 global video / hand mesh / face embedding。

    2. **编码后特征 (B, T, C)**（C 任意，如 BiLSTM 的 512）——
       此时**没有部位语义**，按通道等分三段，保证三流维度之和 = C。
       论文的 hand/face 流靠 HaMeR/ResNet-18 从 RGB 提取，
       landmark 编码后已丢失显式部位划分，此处是退化近似，
       必须在报告中标注为迁移损失。

    通过 last dim 是否等于 368 自动选择。
    """
    if x.shape[-1] == 368:
        x_g = x
        x_h = torch.cat([x[:, :, HANDS_SLICE], x[:, :, HAND_DELTAS_SLICE]], dim=-1)
        x_f = x[:, :, FACE_SLICE]
        return x_g, x_h, x_f
    # 编码后特征：按通道等分三段
    c = x.shape[-1]
    c1 = c // 2
    c2 = c // 4
    return x[:, :, :c1], x[:, :, c1:c1 + c2], x[:, :, c1 + c2:]
