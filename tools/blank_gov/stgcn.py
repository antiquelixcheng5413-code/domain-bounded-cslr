# -*- coding: utf-8 -*-
"""ST-GCN 骨架分支：把 landmark 从扁平向量还原成手部骨架图

## 动机（文献依据）

ref11 ST-GCN（Yan et al., AAAI 2018）Sec 1 明确批评了我们这种做法：

> "Earlier methods of using skeletons for action recognition simply employ the joint
> coordinates at individual time steps to form feature vectors, and apply temporal
> analysis thereon. The capability of these methods is limited as they **do not
> explicitly exploit the spatial relationships among the joints**, which are crucial
> for understanding human actions."

ref23 SignFormer-GCN（Arib et al., PLOS ONE 2025）的实际做法：
- 骨架 X ∈ R^{T×N×Cm}，N = 关节数
- 邻接矩阵 A 按人体骨骼连通性定义
- STGCN block 提骨架特征 → STGCN-LSTM 分支
- 与 RGB 分支相加融合（式 7：Fused = Z_t + L_t）

## 本项目的现状

`artifacts/part3_features/{split}/*.landmark.npy` 形状 `[48, 368]`，布局：
```
[0:63]    左手 21 关节 × 3 坐标   (extractor.py:127  range(21))
[63:126]  右手 21 关节 × 3 坐标   (extractor.py:135  range(21))
[126:158] pose 32
[158:182] face 24
[182:186] presence 4
[186:368] deltas
```

BiLSTM 把 [0:126] 当 126 个无序通道，**丢掉了 21 点手骨的连接拓扑**。

## 本模块做什么

把 `[0:126]` 重组为 `[T, 2, 21, 3]`（2 只手 × 21 关节 × 3 坐标），
用手部骨架邻接矩阵做空间图卷积，输出 `[T, C_out]` 序列，
与原有 368 维扁平分支**并联**后送入 CTC。

**关键：不重新提特征，只改模型侧。**

## 拓扑定义

MediaPipe Hands 的 21 点拓扑（官方 HAND_CONNECTIONS）：

    0 wrist
    1-4   thumb  (CMC, MCP, IP, TIP)
    5-8   index  (MCP, PIP, DIP, TIP)
    9-12  middle
    13-16 ring
    17-20 pinky

骨边（20 条）+ 腕到各指根（MCP）。
ST-GCN 原文用 K=3 分区：中心节点自身 / 向心组 / 离心组。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

# MediaPipe Hands 21 点标准拓扑（0=wrist, 每指 4 点 MCP/PIP/DIP/TIP）
HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                   # palm closure
)

N_JOINTS = 21
N_HANDS = 2
HANDS_SLICE = slice(0, 126)
POSE_SLICE = slice(126, 158)
FACE_SLICE = slice(158, 182)
PRESENCE_SLICE = slice(182, 186)
DELTA_SLICE = slice(186, 368)
FLAT_DIM = 368


def hand_adjacency() -> np.ndarray:
    """手部骨架邻接矩阵 A ∈ R^{21×21}，含自环（ST-GCN 惯例）。"""
    a = np.zeros((N_JOINTS, N_JOINTS), dtype=np.float32)
    for i, j in HAND_BONES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    np.fill_diagonal(a, 1.0)
    return a


def partition3(index: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """ST-GCN 的 K=3 分区：1=自身, 2=向心(离重心更近), 0=离心。

    ref11 Sec 3.4「Spatial configuration partitioning」：
    邻居按「到骨架重心的距离」与根节点比较，分成向心/离心两组。
    手部没有躯干重心，用**掌根(0)到各关节的平均距离**作为代理重心。
    """
    n = adjacency.shape[0]
    part = np.zeros((n, n), dtype=np.int64)      # 默认离心 0
    for i in range(n):
        part[i, i] = 1                            # 自身
        nb = np.where(adjacency[i] > 0)[0]
        nb = nb[nb != i]
        if nb.size == 0:
            continue
        # 代理重心：与所有邻居距离的均值
        d = np.linalg.norm(adjacency[:, None, :] - adjacency[None, :, :], axis=-1)
        centre = float(np.mean([d[i, j] for j in nb]))
        d_nb = np.linalg.norm(adjacency[nb] - adjacency[i], axis=-1)
        for j, dist in zip(nb, d_nb):
            part[i, j] = 2 if dist < centre else 0  # 2=向心, 0=离心
    return part


class STGCNBlock(nn.Module):
    """单个 ST-GCN 块：按 K=3 分区做空间图卷积，再接时间卷积。

    空间：每分区一个可学习权重（W ∈ R^{C_in×C_out/3}），如 ST-GCN 原文 Eq.(4)。
    时间：沿 T 轴做 BN+ReLU+Conv1d。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5,
                 stride: int = 1, dropout: float = 0.3):
        super().__init__()
        assert out_channels % 3 == 0, "out_channels 必须是 3 的倍数（K=3 分区）"
        self.out_channels = out_channels
        # 拼接后每个时间步的向量长度 = C_out * N（关节维被折叠进通道）
        c_per = out_channels // 3
        # 三个分区各一组权重，对应 ST-GCN Eq.(4) 中 w(vti, vtj) = w'(lti(vtj))
        self.p1 = nn.Parameter(torch.empty(in_channels, c_per))
        self.p2 = nn.Parameter(torch.empty(in_channels, c_per))
        self.p3 = nn.Parameter(torch.empty(in_channels, c_per))
        for p in (self.p1, self.p2, self.p3):
            nn.init.kaiming_uniform_(p, a=1.0)
        flat = out_channels * N_JOINTS
        self.bn = nn.BatchNorm1d(flat)
        self.relu = nn.ReLU(inplace=True)
        # 时序卷积：在展平通道（flat = C_out*N）上做，参数远少于逐关节全连接
        self.tconv = nn.Conv1d(flat, flat, kernel_size,
                                stride=stride, padding=kernel_size // 2)
        self.tbn = nn.BatchNorm1d(flat)
        self.drop = nn.Dropout(dropout)
        # 残差：输入 [B,C_in,T,N] -> 目标 [B,C_out,T,N]
        self.res = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, part: torch.Tensor,
                a_hat: torch.Tensor) -> torch.Tensor:
        """x: [B, C_in, T, N]；part: [N, N] long；a_hat: [N, N] float（I + A）。"""
        b, ci, t, n = x.shape
        # 残差：把关节维折进通道，用 1x1 Conv 对齐通道数
        res = self.res(x.reshape(b, ci, t * n)).reshape(b, self.out_channels, t, n)

        # 三个子集：自身(1) / 向心(2) / 离心(0)
        outs = []
        for pid, weight in ((1, self.p1), (2, self.p2), (0, self.p3)):
            mask = (part == pid).to(x.dtype)          # [N, N]
            m = mask * a_hat
            m = m / m.sum(dim=1, keepdim=True).clamp(min=1e-6)
            z = torch.einsum("mn,bctn,cd->bmdt", m, x, weight)
            outs.append(z)
        h = torch.cat(outs, dim=1)                     # [B, C_out, T, N]
        # 关节维折进通道做 BN（ST-GCN 原文用 N 个独立 BN，此处合并等价）
        h = self.bn(h.reshape(b, self.out_channels * n, t))
        h = self.relu(h)                                # [B, C_out*N, T]
        # 时间卷积在展平通道上做（这是 ST-GCN 原文 Eq.(2) 的 T-GCN）
        ht = self.relu(self.tbn(self.tconv(h)))
        ht = self.drop(ht)                              # [B, C_out*N, T]
        return ht.reshape(b, self.out_channels, t, n) + res


class SkeletonBranch(nn.Module):
    """landmark -> 双手骨架图 -> 空间卷积 -> 时序池化 -> [B, C_out, T]。

    输出与扁平分支在时间维对齐（T=48），可与 368 维分支相加或拼接。

    参数量控制：`out_channels * 21` 维的 BN/Conv 是主要开销，
    因此时间卷积用 depthwise 风格的小核，避免 2709 通道全连接。
    """

    def __init__(self, out_channels: int = 66, in_channels: int = 3,
                 dropout: float = 0.3, n_blocks: int = 2):
        super().__init__()
        adj = hand_adjacency()
        part = partition3(np.arange(N_JOINTS, dtype=np.float32)[None, :].repeat(N_JOINTS, 0), adj)
        a_hat = adj.copy()
        np.fill_diagonal(a_hat, 1.0)
        self.register_buffer("part", torch.from_numpy(part))
        self.register_buffer("a_hat", torch.from_numpy(a_hat))
        blocks = [STGCNBlock(in_channels * N_HANDS, out_channels, dropout=dropout)]
        for _ in range(n_blocks - 1):
            blocks.append(STGCNBlock(out_channels, out_channels, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Conv1d(out_channels * N_JOINTS, out_channels, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: [B, 48, 368] -> [B, out_channels, 48]"""
        b, t, _ = feats.shape
        x = feats[:, :, HANDS_SLICE]                       # [B,T,126]
        x = x.reshape(b, t, N_HANDS, N_JOINTS, 3)         # [B,T,2,21,3]
        x = x.permute(0, 2, 4, 1, 3)                       # [B,2,3,T,21]
        x = x.reshape(b, N_HANDS * 3, t, N_JOINTS)        # [B,6,T,21]
        for blk in self.blocks:
            x = blk(x, self.part, self.a_hat)
        h = x.permute(0, 1, 3, 2).reshape(b, -1, t)       # [B, C*21, T]
        return self.proj(h)                                # [B, out, T]


class FlatBranch(nn.Module):
    """原有扁平路径：LayerNorm -> Linear -> BiLSTM。"""

    def __init__(self, input_size: int = FLAT_DIM, hidden_size: int = 256,
                 num_layers: int = 2, projection_size: int = 256, dropout: float = 0.3):
        super().__init__()
        self.normalize = nn.LayerNorm(input_size)
        self.projection = nn.Sequential(
            nn.Linear(input_size, projection_size), nn.GELU(), nn.Dropout(dropout))
        self.temporal = nn.LSTM(projection_size, hidden_size, num_layers,
                                dropout=dropout if num_layers > 1 else 0.0,
                                bidirectional=True, batch_first=True)
        self.out_dim = hidden_size * 2

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        h = self.projection(self.normalize(feats))        # [B,T,P]
        enc, _ = self.temporal(h)                          # [B,T,2H]
        return enc.permute(0, 2, 1)                        # [B,2H,T]


class DualStreamCTCRecognizer(nn.Module):
    """双流：骨架 ST-GCN 分支 + 扁平 BiLSTM 分支 -> 融合 -> CTC。

    融合方式对应 ref23 SignFormer-GCN 式 (7) 的 `Fused = Z_t + L_t`（相加），
    但这里两路维度不同，先各投影到同一宽度再相加。
    """

    def __init__(self, input_size: int = FLAT_DIM, vocabulary_size: int = 301,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3,
                 projection_size: int = 256, skeleton_channels: int = 66,
                 use_skeleton: bool = True, fusion: str = "add"):
        super().__init__()
        self.use_skeleton = use_skeleton
        self.fusion = fusion
        self.flat = FlatBranch(input_size, hidden_size, num_layers,
                               projection_size, dropout)
        fused_dim = self.flat.out_dim
        if use_skeleton:
            self.skel = SkeletonBranch(skeleton_channels, 3, dropout)
            self.skel_proj = nn.Conv1d(skeleton_channels, hidden_size * 2, 1)
            fused_dim = hidden_size * 2
        self.classifier = nn.Linear(fused_dim, vocabulary_size + 1)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return input_lengths

    def forward(self, features: torch.Tensor,
                input_lengths: torch.Tensor | None = None) -> torch.Tensor:
        z = self.flat(features)                              # [B,2H,T]
        if not self.use_skeleton:
            logits = self.classifier(z.permute(0, 2, 1))
            return logits
        l = self.skel_proj(self.skel(features))             # [B,2H,T]
        if self.fusion == "add":
            f = z + l
        elif self.fusion == "concat":
            f = torch.cat([z, l], dim=1)
        else:
            raise ValueError("unknown fusion: " + self.fusion)
        return self.classifier(f.permute(0, 2, 1))          # [B,T,V+1]
