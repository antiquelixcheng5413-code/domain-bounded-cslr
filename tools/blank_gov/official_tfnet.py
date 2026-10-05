"""TFNet —— 复刻 arXiv:2409.11960v2 的官方模型

═��═ 论文原文规格（逐条对照，全部有出处）═══

【架构】式(1)~(4)：
  式(1) f_frame = F_f(V) ∈ R^{T×C'}
         帧级特征提取器 F_f = **MAM-FSD 的 backbone**（论文 Implementation rules）
  式(2) f_temporal = F_s^t(f_frame) ∈ R^{T'×C''}
         时域分支 = **1D CNN + BiLSTM**
  式(3) f_freq = F_s^f(DFT(f_frame)) ∈ R^{T'×C''}
         频域分支 = **DFT + 1D CNN + BiLSTM**
         「两个序列特征提取器用的方法是相同的（都是 1D CNN + BiLSTM）」
  式(4) f_classification = F_linear(f_temporal + f_freq) ∈ R^{T'×l}
         **相加**后过全连接；l = 词表大小

【损失】
  L_sum = L_VAE_t + L_VAE_f + L_CTC
  两个辅助 VAE 损失（分别作用于时域/频域序列特征）+ 最终 CTC

【训练超参】Implementation rules：
  - 优化器 Adam，初始 lr = 1e-4，weight decay 1e-4
  - batch size 2
  - 硬件 RTX3090Ti 24GB
  - 增强：随机裁剪 256×256 → 224×224（翻转与裁剪作用于**视频序列**）
  - 增强：随机翻转 p = 0.5
  - 增强：时序增强 **±20%**（随机加长/缩短视频序列长度）
  - 共 **55 epoch**，lr 在第 35、45 epoch **降 80%**
  - 测试：仅中心裁剪；**CTC beam search width = 10**

【评估】式(6)：
  WER = 100% × (ins + del + sub) / sum

【官方基准】CE-CSL Dev/Test WER %：
  MSTNet 54.4/53.0 · CorrNet 47.2/46.5 · SEN 46.5/45.3
  VAC 45.1/43.3 · MAM-FSD 44.9/44.7 · **TFNet 42.1/41.9**

═══ 与我们现有管线的差异（必须诚实标注）═══
  论文 F_f 是 **RGB + CNN backbone**；我们只有 MediaPipe landmark。
  ⇒ 本复刻把 F_f 替换为 landmark 投影层，**时频双域的核心贡献完整保留**。
  ⇒ 这不是「复现 42.1%」，而是「把 TFNet 的时频双域机制用到 landmark 特征上」。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SeqBranch(nn.Module):
    """论文式(2)/(3) 共用的序列特征提取器：**1D CNN + BiLSTM**

    原文：'The methods used by these two sequence feature extractors are the same
    (both being 1D CNN + Bi-LSTM)'
    """

    def __init__(self, in_ch: int, hidden: int = 256, conv_ch: int = 256,
                 dropout: float = 0.3, kernel: int = 3):
        super().__init__()
        # 1D CNN：沿时间轴做局部建模
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, conv_ch, kernel_size=kernel,
                      padding=kernel // 2),
            nn.BatchNorm1d(conv_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # BiLSTM：全局时序
        self.lstm = nn.LSTM(conv_ch, hidden, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden * 2

    def forward(self, x):                    # x: (B, T, C)
        h = self.conv(x.transpose(1, 2))     # (B, conv_ch, T)
        h = h.transpose(1, 2)                # (B, T, conv_ch)
        h, _ = self.lstm(h)                  # (B, T, hidden*2)
        return self.drop(h)


class TFNet(nn.Module):
    """Time-Frequency Network（复刻官方结构）

    forward 返回 logits (B, T', l)，供 CTC 使用。
    """

    def __init__(self, feat_dim: int = 368, hidden: int = 256,
                 conv_ch: int = 256, vocab: int = 302, dropout: float = 0.3,
                 use_vae: bool = True, vae_latent: int = 64):
        super().__init__()
        self.use_vae = use_vae

        # ---- 式(1) 帧级特征 F_f ----
        # 论文用 MAM-FSD backbone（RGB）；我们用 landmark，故用一个线性投影代替
        # **这是与论文唯一的结构性差异**，已在模块 docstring 中声明
        self.frame_proj = nn.Sequential(
            nn.Linear(feat_dim, conv_ch),
            nn.LayerNorm(conv_ch),
            nn.ReLU(inplace=True),
        )

        # ---- 式(2) 时域分支 ----
        self.temporal = SeqBranch(conv_ch, hidden, conv_ch, dropout)

        # ---- 式(3) 频域分支：DFT 后送入同一个结构的提取器 ----
        self.spectral = SeqBranch(conv_ch, hidden, conv_ch, dropout)

        out_ch = self.temporal.out_dim         # 两分支输出维度相同（论文原文）

        # ---- 式(4) 相加 + 全连接 ----
        self.classifier = nn.Linear(out_ch, vocab)

        # ---- 辅助 VAE（论文 L_VAE_t + L_VAE_f）----
        if use_vae:
            self.vae_t = _VAE(out_ch, vae_latent)
            self.vae_f = _VAE(out_ch, vae_latent)

    @staticmethod
    def _dft(x):                             # (B, T, C) -> (B, T, C)
        """时域→频域（论文式3）。

        论文原文：频域序列特征的尺寸与时域**保持一致**
        （'The size of the spectral domain sequence features is consistent
        with that of the temporal domain sequence features'）。
        因此 DFT 后必须把长度还原到 T'，否则式(4) 的 `f_t + f_f` 无法相加
        （rfft 会把 T=48 变成 25，直接相加会报 shape mismatch）。

        ⚠️ 论文只说 'transform ... via DFT'，未说明长度还原方式。
        这里用**零填充回T**（最简且不引入额外参数），记为复刻偏差。
        """
        T = x.size(1)
        spec = torch.fft.rfft(x, dim=1, norm="ortho").real   # (B, T//2+1, C)
        if spec.size(1) < T:                    # 零填充到原长度
            spec = torch.nn.functional.pad(spec, (0, 0, 0, T - spec.size(1)))
        else:
            spec = spec[:, :T, :]
        return spec

    def forward(self, x, input_lengths=None):
        """x: (B, T, feat_dim)"""
        f = self.frame_proj(x)                                # 式(1)

        f_t = self.temporal(f)                                # 式(2)
        f_f = self.spectral(self._dft(f))# 式(3)

        logits = self.classifier(f_t + f_f)                   # 式(4) 相加

        aux = {}
        if self.training and self.use_vae:
            # VAE 在序列特征上做，形状与 f_* 一致：(B, C, T')
            aux["vae_t"] = self.vae_t(f_t.transpose(1, 2), f_t.transpose(1, 2))
            aux["vae_f"] = self.vae_f(f_f.transpose(1, 2), f_f.transpose(1, 2))
        return logits, aux

    @staticmethod
    def loss(logits, aux, targets, input_lengths, target_lengths,
             w_ctc: float = 1.0, w_vae: float = 0.1):
        """L_sum = L_CTC + w*(L_VAE_t + L_VAE_f)

        权重 w 未在论文中给出（原文只写 L_sum 由三部分构成），
        这里默认 0.1 并**显式标注为我们的选择**。
        """
        logp = logits.log_softmax(-1).transpose(0, 1)        # (T,B,C)
        ctc = torch.nn.functional.ctc_loss(
            logp, targets, input_lengths, target_lengths,
            blank=0, reduction="mean", zero_infinity=False)
        vae = logits.new_zeros(())
        n_vae = 0
        for k, out in aux.items():
            rec, mu, logvar, x_in = out     # rec (B,C,T)  mu/logvar (B,L,T)
            # ⚠️ **必须按维度求平均，不能 sum。**
            #   第一次实现用了 sum，量级达到 -1982 而 CTC 只有 61，
            #   VAE 项会完全压过 CTC（实测 total 变成负数）。
            #   标准 VAE 写法是对 batch 求和再除以 batch 数（对时间维求和）。
            kl = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp())       # (B,L,T) -> mean
            rec_mse = torch.mean((rec - x_in).pow(2))          # (B,C,T) -> mean
            vae = vae + kl + rec_mse
            n_vae += 1
        if n_vae:
            vae = vae / n_vae
        return w_ctc * ctc + w_vae * vae, float(ctc), float(vae)


class _VAE(nn.Module):
    """时域/频域分支各自的辅助 VAE（论文 L_VAE_t / L_VAE_f）。

    论文只说「two auxiliary VAE loss functions, L_VAE_t and L_VAE_f」，
    未给网络细节；此处用标准 VAE（encoder → mu/logvar，decoder → rec），
    **作为复刻偏差记录在案**。
    """

    def __init__(self, ch: int, latent: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(ch, 128, 1), nn.ReLU(True))
        self.mu = nn.Conv1d(128, latent, 1)
        self.logvar = nn.Conv1d(128, latent, 1)
        self.dec = nn.Sequential(
            nn.Conv1d(latent, 128, 1), nn.ReLU(True), nn.Conv1d(128, ch, 1))

    def forward(self, x, x_in=None):
        """x: (B, C, T')。x_in 为 VAE 的重构目标，默认等于 x。"""
        h = self.enc(x)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(z), mu, logvar, (x if x_in is None else x_in)


# ════════════════════════════════════════════════════════════
#  官方训练配置（论文 Implementation rules）
# ════════════════════════════════════════════════════════════
OFFICIAL_CONFIG = {
    "optimizer": "Adam",
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 2,
    "epochs": 55,
    "lr_decay_epochs": [35, 45],
    "lr_decay_factor": 0.2,          # 「降80%」= ×0.2
    "augment": {
        "random_crop": "256x256 -> 224x224，作用于视频序列",
        "horizontal_flip_p": 0.5,
        "temporal_jitter": "±20%",
    },
    "test_augment": "仅中心裁剪",
    "decode": "CTC beam search, width=10",
    "hardware": "RTX3090Ti 24GB",
    "frame_extractor": "MAM-FSD backbone (RGB)",
}

OFFICIAL_BENCHMARK_DEV = {
    "MSTNet": 54.4, "CorrNet": 47.2, "SEN": 46.5,
    "VAC": 45.1, "MAM-FSD": 44.9, "TFNet": 42.1,
}


if __name__ == "__main__":
    print("=" * 70)
    print("TFNet 结构自检")
    print("=" * 70)
    net = TFNet(feat_dim=368, vocab=3517, hidden=256)
    n = sum(p.numel() for p in net.parameters())
    print("  参数 %.2f M" % (n / 1e6))
    print("  时域分支输出维度 %d" % net.temporal.out_dim)
    print("  频域分支输出维度 %d（论文：两分支尺寸一致）" % net.spectral.out_dim)

    net.train()
    x = torch.randn(2, 48, 368)
    logits, aux = net(x)
    print("\n  式(1) 输入 (2,48,368)")
    print("  式(4) logits shape = %s  （B, T', l）" % (tuple(logits.shape),))
    print("  aux keys = %s（训练时启用 VAE）" % list(aux))
    rec, mu, logvar, xin = aux["vae_t"]
    print("  VAE 输出 rec%s mu%s logvar%s x%s" % (tuple(rec.shape),
                                             tuple(mu.shape),
                                             tuple(logvar.shape),
                                             tuple(xin.shape)))

    tg = torch.randint(1, 3517, (2, 6), dtype=torch.long)
    tl = torch.full((2,), 6, dtype=torch.long)
    il = torch.full((2,), 48, dtype=torch.long)
    total, ctc, vae = TFNet.loss(logits, aux, tg, il, tl)
    print("\n  损失：total %.4f = ctc %.4f + 0.1*vae %.4f"
          % (total, ctc, vae))

    net.eval()
    with torch.no_grad():
        logits, aux = net(x)
    print("\n  eval 模式 aux 为空 = %s（VAE 仅训练时用）" % (aux == {}))

    print("\n" + "=" * 70)
    print("官方配置")
    print("=" * 70)
    for k, v in OFFICIAL_CONFIG.items():
        print("  %-22s %s" % (k, v))
    print("\n官方基准（CE-CSL Dev WER %%）：")
    for k, v in sorted(OFFICIAL_BENCHMARK_DEV.items(), key=lambda kv: kv[1]):
        print("  %-9s %.1f" % (k, v))
    print("\n⚠️ 复刻偏差（必须诚实标注）：")
    print("  1. F_f：论文用 MAM-FSD RGB backbone，我们用 landmark 投影")
    print("     ⇒ 本实现**不是**复现 42.1%，而是把时频双域机制用于 landmark")
    print("  2. DFT 取实部；论文未说明取幅值/实部/复数")
    print("  3. VAE 网络结构为标准实现；论文未给细节")
    print("  4. L_CTC 与 L_VAE 的权重比论文未给，默认 1.0 : 0.1")
    print("  5. 增强作用于 landmark 时退化为「时序抖动±20%」有效，")
    print("     裁剪/翻转对已抽好的 368 维特征不适用")
    print("     （P63b 已提的特征无法再做像素级增强 —— 这是重要限制）")