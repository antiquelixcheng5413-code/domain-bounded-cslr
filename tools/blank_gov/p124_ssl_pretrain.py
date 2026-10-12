"""P124a：域内自监督预训练（SimCLR）—— 为 VAC 的 2D 主干换一个更好的初始化。

═══════════════════════════════════════════════════════════════
为什么做这件事（目标）
═══════════════════════════════════════════════════════════════
痛点：低频 gloss 判别力不足（48% gloss 在训练集只出现 1 次；P36 实测
      hands 视图 Cohen's d ≈ 0）。P107B2 实测 CLIP 帧特征 d = +0.4119，
      说明「帧级表征」本身有判别力，但我们的主干是 **ImageNet 物体分类**
      初始化，从未见过手形。

做法：不用任何 gloss 标签，只在 CE-CSL 自己的 RGB 帧上做
      SimCLR 对比学习（域内自适应预训练），再把主干权重塞回 VAC。

═══════════════════════════════════════════════════════════════
依据等级
═══════════════════════════════════════════════════════════════
【有】VAC 的主干是 **2D ResNet18 + 逐帧前向**（external/TFNet/Net.py:88
      `self.conv2d = getattr(models,"resnet18")(pretrained=True)`，
      前向 Net.py:331 `seqData.reshape(B*T,C,H,W)` 后送 conv2d）
      ⇒ torchvision resnet18 的 state_dict 可以**直接**替换，无需 inflate。
【有】ref12（CCL-SLR, Wu et al.）原文：
      "Pre-training has been proven to be effective in boosting the
       performance of Isolated Sign Language Recognition (ISLR)"
      且它用的正是「单模态对比 + 跨模态对比」的自监督预训练。
      ⚠️ 但 ref12 的验证任务是 **ISLR（孤立词）**，不是 CSLR（连续）。
【间接】SignBERT（Hu et al. ICCV'21，ref24 参考文献 [841 行]）也是
      自监督预训练，同样只在孤立词上验证。
【无】**没有任何论文在 CE-CSL 的连续识别 WER 上验证过自监督预训练**。
      反向证据：官方表中 CorrNet 47.2 反而比 VAC 45.1 差 2.1 pp，
      而 CorrNet 的核心正是「自监督一致性约束」（【有】，负证据）。

⇒ 所以本脚本只产出权重，不做结论；是否真的有用由 P124b 的
  判别力探针（Cohen's d + bootstrap CI）判定，判据预先写死。

═══════════════════════════════════════════════════════════════
与官方/下游的口径一致性
═══════════════════════════════════════════════════════════════
- 输入归一化：x/127.5 - 1 → [-1,1]（与 p78 一致）
- 分辨率 160（与 P92 结论一致：160 比 224 好 5.60 pp）
- 水平翻转 p=0.5 作为不变性（P119 已证 hflip 作为增强有效 −3.42 pp）
- **不用**随机大比例裁剪（scale 0.5~1.0）：手的位置相对身体是有意义的，
  过强的裁剪不变性可能抹掉判别信息（这是我自己加的约束，依据【无】，
  但风险方向是保守的）
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
RGB = REPO / "artifacts/official_rgb/train"
CKDIR = REPO / "artifacts/checkpoints"


def log(*a):
    print("[P124a %s] %s" % (time.strftime("%H:%M:%S"),
                             " ".join(str(x) for x in a)), flush=True)


# ---------------------------------------------------------------- 增强
def augment(x: torch.Tensor, size: int) -> torch.Tensor:
    """x: [3,H,W] float in [0,1] → 一个增强视图，仍 [0,1]。"""
    # 随机裁剪（尺度保守：0.5~1.0，保住手相对身体的位置信息）
    i, j, h, w = _crop_params(x, scale=(0.5, 1.0), ratio=(0.75, 1.33))
    x = TF.resized_crop(x, i, j, h, w, size=[size, size])
    if random.random() < 0.5:
        x = TF.hflip(x)
    if random.random() < 0.8:
        # 颜色抖动（亮度/对比/饱和/色相）
        x = TF.adjust_brightness(x, 1.0 + random.uniform(-0.4, 0.4))
        x = TF.adjust_contrast(x, 1.0 + random.uniform(-0.4, 0.4))
        x = TF.adjust_saturation(x, 1.0 + random.uniform(-0.4, 0.4))
        x = TF.adjust_hue(x, random.uniform(-0.1, 0.1))
    if random.random() < 0.2:
        x = TF.rgb_to_grayscale(x, num_output_channels=3)
    if random.random() < 0.5:
        x = TF.gaussian_blur(x, kernel_size=[5, 5], sigma=[0.1, 2.0])
    return x


def _crop_params(img, scale, ratio):
    """复刻 torchvision RandomResizedCrop.get_params（避免版本差异）。"""
    import math
    _, H, W = img.shape
    area = H * W
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect = math.exp(random.uniform(*log_ratio))
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
        if 0 < w <= W and 0 < h <= H:
            i = random.randint(0, H - h)
            j = random.randint(0, W - w)
            return i, j, h, w
    # fallback：中心裁剪
    in_ratio = W / H
    if in_ratio < min(ratio):
        w, h = W, int(round(W / min(ratio)))
    elif in_ratio > max(ratio):
        w, h = int(round(H * max(ratio))), H
    else:
        w, h = W, H
    i, j = (H - h) // 2, (W - w) // 2
    return i, j, h, w


# ---------------------------------------------------------------- 数据
class FramePool(Dataset):
    """从每个视频里均匀抽 K 帧，构成一个「无标注帧池」。

    无标签 ⇒ 不存在任何 label leakage。
    """

    def __init__(self, root: Path, k: int, size: int, max_videos: int = 0):
        self.size = size
        self.items = []
        vids = []
        for tr in sorted(os.listdir(root)):
            d = root / tr
            if not d.is_dir():
                continue
            for sid in sorted(os.listdir(d)):
                vids.append(d / sid)
        if max_videos:
            vids = vids[:max_videos]
        for v in vids:
            fs = sorted(f for f in os.listdir(v) if f.endswith(".jpg"))
            if not fs:
                continue
            kk = min(k, len(fs))
            idx = np.linspace(0, len(fs) - 1, num=kk, dtype=int)
            self.items.extend(str(v / fs[i]) for i in idx)
        log("帧池：%d 视频 → %d 帧" % (len(vids), len(self.items)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = cv2.imread(self.items[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size),
                         interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1)  # [3,H,W]
        v1 = augment(x, self.size)
        v2 = augment(x, self.size)
        to = lambda t: t * 2.0 - 1.0          # [0,1] → [-1,1]（= x/127.5-1）
        return to(v1), to(v2)


# ---------------------------------------------------------------- 模型
class SimCLR(nn.Module):
    def __init__(self, from_scratch: bool, proj_dim: int = 256):
        super().__init__()
        w = None if from_scratch else ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=w)
        self.backbone.fc = nn.Identity()          # 与官方 VAC 一致（512 维）
        self.proj = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim))

    def forward(self, x):
        f = self.backbone(x)                     # [B,512]
        return f, F.normalize(self.proj(f), dim=1)


def nt_xent(z1, z2, temp=0.2):
    """SimCLR 的 NT-Xent（InfoNCE）。z 已 L2 归一化。"""
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)               # [2N,d]
    s = z @ z.t()                                # 余弦相似度
    s = s / temp
    eye = torch.eye(2 * N, dtype=torch.bool, device=s.device)
    s = s.masked_fill(eye, -1e9)                 # 去掉自身
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(s.device)
    return F.cross_entropy(s, labels)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--frames-per-video", type=int, default=16)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--from-scratch", action="store_true",
                    help="不用 ImageNet 初始化（纯从头对比学习）")
    ap.add_argument("--tag", default="p124a-simclr")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    ds = FramePool(RGB, a.frames_per_video, a.size, a.max_videos)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True,
                    num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=a.workers > 0)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SimCLR(a.from_scratch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=False)   # bf16 不安全时用 fp32/bf16 autocast
    use_amp = True

    log("device=%s  batch=%d  steps/epoch=%d  epochs=%d  init=%s"
        % (dev, a.batch, len(dl), a.epochs,
           "scratch" if a.from_scratch else "ImageNet"))

    t0 = time.time()
    hist = []
    for ep in range(1, a.epochs + 1):
        net.train()
        run, nb = 0.0, 0
        t_ep = time.time()
        try:
            for v1, v2 in dl:
                v1 = v1.to(dev, non_blocking=True)
                v2 = v2.to(dev, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp,
                                        dtype=torch.bfloat16):
                    _, z1 = net(v1)
                    _, z2 = net(v2)
                    loss = nt_xent(z1.float(), z2.float(), a.temp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                run += loss.item(); nb += 1
        except Exception as e:
            log("ep%02d 训练中断：%r" % (ep, e))
            CKDIR.mkdir(parents=True, exist_ok=True)
            out = CKDIR / ("%s-final.pt" % a.tag)
            torch.save({"state_dict": net.backbone.state_dict(),
                        "epoch": ep, "hist": hist, "config": vars(a),
                        "note": "⚠️ 中断应急存盘（未完成 %d 轮），实际停在 ep%d"
                                % (a.epochs, ep)},
                       out)
            log("应急存盘 -> %s（ep%d）" % (out, ep))
            sys.exit(1)
        sched.step()
        dt = (time.time() - t_ep) / 60
        el = (time.time() - t0) / 60
        eta = el / ep * (a.epochs - ep)
        log("ep%02d  loss %.4f  (%.1f min/轮, 已用 %.0f min, ETA %.0f min)"
            % (ep, run / max(nb, 1), dt, el, eta))
        hist.append({"epoch": ep, "loss": round(run / max(nb, 1), 5)})
        if ep % 3 == 0 or ep == a.epochs:
            CKDIR.mkdir(parents=True, exist_ok=True)
            p = CKDIR / ("%s-ep%02d.pt" % (a.tag, ep))
            torch.save({"state_dict": net.backbone.state_dict(),
                        "epoch": ep, "hist": hist, "config": vars(a)},
                       p)
            log("已存 %s" % p)

    CKDIR.mkdir(parents=True, exist_ok=True)
    out = CKDIR / ("%s-final.pt" % a.tag)
    torch.save({"state_dict": net.backbone.state_dict(), "epoch": a.epochs,
                "hist": hist, "config": vars(a),
                "note": "torchvision resnet18，fc=Identity（与官方 VAC 主干同构）"},
               out)
    rec = REPO / "artifacts/metrics/blank-gov/p124a-simclr.json"
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text(__import__("json").dumps(
        {"experiment": "P124a", "date": time.strftime("%Y-%m-%d"),
         "method": "SimCLR (NT-Xent) 域内自监督预训练",
         "target": "替换官方 VAC 的 2D ResNet18 主干初始化",
         "evidence_level": {"VAC主干是2D resnet18": "【有】Net.py:88/331",
                            "自监督预训练提升手语识别": "【有】但只在 ISLR（ref12 CCL-SLR 原文）",
                            "CSLR/CE-CSL 上有效": "【无】",
                            "反向证据": "【有】官方 CorrNet 47.2 > VAC 45.1（自监督一致性反而更差）"},
         "config": vars(a), "n_frames": len(ds), "history": hist,
         "minutes": round((time.time() - t0) / 60, 1),
         "ckpt": str(out)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    log("完成 -> %s（%.0f min）" % (out, (time.time() - t0) / 60))
    log("收据 -> %s" % rec)


if __name__ == "__main__":
    main()
