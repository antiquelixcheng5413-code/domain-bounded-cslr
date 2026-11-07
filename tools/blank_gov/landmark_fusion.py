"""Landmark 逐帧融合模块（P206 / E4）。

用途
────
把已落盘的 MediaPipe landmark 特征（data/processed/ce-csl-landmark48，
每条 clip 固定 (48, 368) float32）作为**残差**注入 RGB 主干的逐帧特征。

做法：官方 Net.py 的 TFNet 分支里 `self.conv2d(x)` 返回
`[B*temp, 512]` 的逐帧特征（随后被 reshape 成 [B, temp, 512]）。
在 `model.conv2d` 上注册 forward hook，把 delta 加回 tuple 的第 0 项，
即可让 landmark 信息从第一帧开始参与后续 TemporalConv / 频域 / BiLSTM，
而**不改一行官方代码**。

零初始化理由
──────────
融合头 `proj` 的 weight / bias 全部用 `nn.init.zeros_()` 初始化，
所以接上融合模块后的第 0 步 delta ≡ 0 ⇒ 训练初始点与 P184 基线**严格等价**。
否则随机初始化会在 ep0 就把主干的输入特征打偏，无法区分增益来自
landmark 还是来自「多加了一层随机网络」。LSTM 之前的层无需零初始化，
因为它们只影响 delta 的数值大小（proj=0 时整体恒 0）。

注意：landmark 只覆盖 train + dev（无 test 文件）。缺失文件的 sid 返回
全 0 行，并计入 stats() 的 miss 计数。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FRAMES = 48
DIM = 368
OUT_DIM = 512          # ResNet34MAM 的逐帧输出通道数


class LandmarkBank:
    """按 sid 懒加载 (48, 368) 的 landmark，并用 train 子集统计 mu/sd。

    - 只在 CPU 上做归一化，再 `.to(device)`；GPU 上只做后续融合。
    - 缺文件返回全 0 行（归一化后是常量 -mu/sd，会正常参与后续层）。
    """

    def __init__(self, root, n_stat: int = 200):
        self.root = Path(root)
        self._cache: dict[str, np.ndarray] = {}
        self._miss = 0
        self._mu, self._sd = self._compute_stats(n_stat)

    # ---- 归一化统计（确定性采样 200 条 train sid）----
    def _compute_stats(self, n_stat: int):
        files = sorted(self.root.glob("train-*.npy"))
        if not files:
            return (np.zeros((1, 1, DIM), np.float32),
                    np.ones((1, 1, DIM), np.float32))
        n = min(n_stat, len(files))
        # 等间隔取，避免只看开头（不同 sids 的数值分布可能略有差异）
        idx = np.unique(np.linspace(0, len(files) - 1, n).round().astype(int))
        arr = np.stack([np.load(files[i]).astype(np.float32) for i in idx])
        mu = arr.mean(axis=(0, 1), keepdims=True)
        sd = arr.std(axis=(0, 1), keepdims=True) + 1e-6
        return (mu.astype(np.float32), sd.astype(np.float32))

    def _load(self, sid: str) -> np.ndarray:
        a = self._cache.get(sid)
        if a is None:
            p = self.root / ("%s.npy" % sid)
            if p.exists():
                a = np.load(p).astype(np.float32)
            else:
                a = np.zeros((FRAMES, DIM), np.float32)
                self._miss += 1
            self._cache[sid] = a
        return a

    def get(self, sids, device) -> torch.Tensor:
        """返回 float32 [B, 48, 368] 的归一化特征（已搬到 device）。"""
        arr = np.zeros((len(sids), FRAMES, DIM), np.float32)
        for i, sid in enumerate(sids):
            arr[i] = self._load(str(sid))
        arr = (arr - self._mu) / self._sd       # CPU 上归一化
        return torch.from_numpy(arr).to(device)

    def stats(self) -> dict:
        return {"miss": self._miss, "cached": len(self._cache),
                "root": str(self.root)}


class LandmarkFusion(nn.Module):
    """(B,48,368) → 残差 delta (B,temp,512)，通过 conv2d 的 forward hook 注入。"""

    def __init__(self, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.bank = None
        self._sids = None
        self.fc = nn.Linear(DIM, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden, hidden // 2, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden, OUT_DIM)
        # ★ 关键：零初始化 ⇒ ep0 的 delta 恒为 0，与 P184 严格等价
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def set_bank(self, bank: LandmarkBank) -> None:
        self.bank = bank

    def set_sids(self, sids) -> None:
        self._sids = None if sids is None else list(sids)

    def hook(self, module, inputs, output):
        if self._sids is None or self.bank is None:
            return output
        fw = output[0]                       # conv2d 原样输出 [B*temp, 512]
        temp = fw.shape[0] // len(self._sids)   # 序列时间步（含 padding）
        # cuDNN LSTM 在 bf16 autocast 下不可靠 ⇒ 融合整段强制 fp32
        with torch.autocast(device_type=fw.device.type, enabled=False):
            lm = self.bank.get(self._sids, fw.device).float()
            h = self.drop(self.gelu(self.ln(self.fc(lm))))     # [B, 48, hidden]
            h, _ = self.lstm(h)                               # [B, 48, 256]
            if h.shape[1] != temp:
                h = F.interpolate(h.transpose(1, 2), size=temp,
                                  mode="linear", align_corners=False)
                h = h.transpose(1, 2)
            delta = self.proj(h)                              # [B, temp, 512]
            out0 = fw + delta.reshape(fw.shape).to(fw.dtype)
        return (out0,) + tuple(output[1:])
