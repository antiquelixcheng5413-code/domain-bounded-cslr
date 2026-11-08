"""P198：Conformer 时序适配器（插在 BiLSTM temporal_model 输出上，主干全冻）。

设计要点
─────────
1. **bottleneck**：1024 → 256 → 1024。BiLSTM 输出喂给**冻结的** classifier11，
   所以适配器必须保持维度不变，只能改特征不能改形状；bottleneck 同时把
   参数量压到 ~2.4M（2 层），比全宽 1024 的 26M 小一个量级。
2. **零初始化 ⇒ 初始严格恒等**：`out_proj.weight/bias = 0`，且 forward 返回
   **残差增量**（调用方做 `pred + adapter(pred, lens)`）。因此接上适配器后的
   第 0 步模型输出与 p184 基线**逐元素相同**，dev WER 不会因为"插入模块"而变化，
   收益/损失完全来自训练。
3. **掩码**：BiLSTM 用 pack_padded_sequence，padding 位置输出为 0。适配器内部
   全程 mask 掉 padding；注意力对"整行都被 mask"的 query 会算出 NaN（softmax
   分母为 0），所以每步 attention 后 `nan_to_num` 再乘 mask —— depthwise conv
   的 k=15 会把 NaN 沿时间轴扩散到有效位置，这一步不能省。

维度约定
────────
    x: [T, B, D]   （与 pad_packed_sequence 输出一致，T 在前）
    lengths: [B]   （官方 lgt，CPU int64）
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MHSA(nn.Module):
    """多头自注意力（带 padding mask）。"""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout,
                                         batch_first=False)
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
        # x: [T,B,d]  m3: [T,B,1]
        h = self.ln(x)
        pad = (m3[:, :, 0] < 0.5)                     # [T,B] True=忽略
        a, _ = self.att(h, h, h, key_padding_mask=pad.transpose(0, 1))
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return self.do(a) * m3


class _ConvModule(nn.Module):
    """Conformer 卷积模块：LN → PWConv → GLU → DWConv(k=15) → LN → PWConv。"""

    def __init__(self, d: int, kernel: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.ln2 = nn.LayerNorm(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x).permute(1, 2, 0)              # [T,B,d] -> [B,d,T]
        y = F.glu(self.pw1(y), dim=1)                 # [B,d,T]
        y = self.dw(y)
        y = self.ln2(y.transpose(1, 2)).transpose(1, 2)
        y = self.pw2(y).permute(2, 0, 1)              # [B,d,T] -> [T,B,d]
        return self.do(y) * m3


class _FFN(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mult * d)
        self.fc2 = nn.Linear(mult * d, d)
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.do(self.fc2(self.do(F.silu(self.fc1(y)))))
        return y * m3


class _Block(nn.Module):
    """Conformer block（macaron 0.5 → MHSA → Conv → macaron 0.5 → LN）。"""

    def __init__(self, d: int, heads: int, kernel: int, dropout: float):
        super().__init__()
        self.ffn1 = _FFN(d, 4, dropout)
        self.mhsa = _MHSA(d, heads, dropout)
        self.conv = _ConvModule(d, kernel, dropout)
        self.ffn2 = _FFN(d, 4, dropout)
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x, m3)
        x = x + self.mhsa(x, m3)
        x = x + self.conv(x, m3)
        x = x + 0.5 * self.ffn2(x, m3)
        return self.ln(x) * m3


class ConformerAdapter(nn.Module):
    """返回**残差增量**（调用方 `pred + adapter(pred, lens)`），零初始化。"""

    def __init__(self, in_dim: int = 1024, d: int = 256, layers: int = 2,
                 heads: int = 4, kernel: int = 15, dropout: float = 0.1,
                 max_len: int = 4096):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList(
            [_Block(d, heads, kernel, dropout) for _ in range(layers)])
        self.out_proj = nn.Linear(d, in_dim)

        # 固定正弦位置编码（不参与训练）
        pos = torch.zeros(max_len, d)
        idx = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d))
        pos[:, 0::2] = torch.sin(idx * div)
        pos[:, 1::2] = torch.cos(idx * div)
        self.register_buffer("pos", pos, persistent=False)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, lengths) -> torch.Tensor:
        T, B = x.shape[0], x.shape[1]
        lens = lengths.to(x.device).long().view(-1)
        m = (torch.arange(T, device=x.device)[None, :]
             < lens[:, None]).to(x.dtype)               # [B,T]
        m3 = m.t().unsqueeze(-1)                        # [T,B,1]

        y = self.in_proj(x) * m3
        y = y + self.pos[:T, None, :].to(y.dtype)
        for blk in self.blocks:
            y = blk(y, m3)
        return self.out_proj(y)
