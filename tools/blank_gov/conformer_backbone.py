"""P200：Conformer 时序主干（CONTROL 实验，替换 TFNet 原生 BiLSTM）。

与 P198 适配器的区别
──────────────────
- P198：冻结 BiLSTM，只在输出上挂**残差** Conformer 增量（零初始化 ⇒ 恒等起步），
  本质是给已饱和的表征打补丁；
- P200：用 Conformer **完全替换** `temporal_model` / `temporal_model1`，
  没有 BiLSTM 通路，时序建模必须从零学起。

接口与 `BiLSTMLayer` 完全一致
──────────────────────────
    forward(x[T,B,D], lens) -> {"predictions": [T,B,D], "hidden": None}
padding 位置输出恒为 0（与 pack_padded_sequence 的行为一致），
视频级 `lgt` 直接复用，调用方（官方 Net.py）无需改动。

结构
────
    1024 --LN+Linear--> 256 --正弦位置编码--> N×Conformer Block
         --LN+Linear--> 1024
块实现复用 `conformer_adapter._Block`（macaron FFN / MHSA / DWConv k=15），
保证 P198 与 P200 的结构定义完全同源。
"""

import math

import torch
import torch.nn as nn

from conformer_adapter import _Block


class ConformerTemporal(nn.Module):
    def __init__(self, input_size: int = 1024, hidden_size: int = 1024,
                 d: int = 256, layers: int = 4, heads: int = 4,
                 kernel: int = 15, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(input_size), nn.Linear(input_size, d))
        self.blocks = nn.ModuleList(
            [_Block(d, heads, kernel, dropout) for _ in range(layers)])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, hidden_size))

        pos = torch.zeros(max_len, d)
        idx = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d))
        pos[:, 0::2] = torch.sin(idx * div)
        pos[:, 1::2] = torch.cos(idx * div)
        self.register_buffer("pos", pos, persistent=False)

        # 输出层不零初始化：它不是残差补丁，必须从第一步就提供有效表征
        nn.init.xavier_uniform_(self.out_proj[1].weight)
        nn.init.zeros_(self.out_proj[1].bias)

    def forward(self, src_feats: torch.Tensor, src_lens=None, hidden=None):
        T, B = src_feats.shape[0], src_feats.shape[1]
        device = src_feats.device
        if src_lens is None:
            lens = torch.full((B,), T, dtype=torch.long, device=device)
        else:
            lens = src_lens.to(device).long().view(-1)
        m = (torch.arange(T, device=device)[None, :]
             < lens[:, None]).to(src_feats.dtype)     # [B,T]
        m3 = m.t().unsqueeze(-1)                      # [T,B,1]

        y = self.in_proj(src_feats) * m3
        y = y + self.pos[:T, None, :].to(y.dtype)
        for blk in self.blocks:
            y = blk(y, m3)
        return {"predictions": self.out_proj(y) * m3, "hidden": None}
