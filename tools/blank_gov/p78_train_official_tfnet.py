"""P78：用**官方原版代码**训练 TFNet（RGB + resnet34MAM）。

═══════════════════════════════════════════════════════════════
设计铁律（今天栽了 6 次同源错误后确立）
═══════════════════════════════════════════════════════════════
1. **网络直接 import 官方 `Net.py` / `Module.py`，不重写。**
   我曾照论文描述自己写 official_tfnet.py，错了 5 处。
2. **官方增强直接 import 官方 `videoAugmentation.py`，不重写。**
3. **词表直接用官方 `DataProcessMoudle.Word2Id`。**
4. 只写官方代码没覆盖的部分：Dataset（我们的 jpg 目录结构）、collate、评估、收据。
5. 判优只用 `official_wer.evaluate`（官方口径），其余标注为附加诊断。

官方规格（全部来自 external/TFNet 源码，见 docs/planning/OFFICIAL_TFNET_SPEC.md）：
  主干      ResNet34MAM（ImageNet 2D 权重 → 3D Conv3d + 逐层 MotorAttention）
  序列提取器 TemporalConv(conv_type=2) = [Conv1d(k=5,pad=0)+BN+ReLU+MaxPool(2)]×2
  输出层    NormLinear（权重按输出通道 L2 归一化、无 bias）
  频域      fft(framewise, dim=-1).abs()  ← 沿特征维，不是时间轴
  损失      CTC(blank=0, reduction='none', zero_infinity=True).mean() + SeqKD(T=8)
  增强      RandomCrop(224) + RandomHorizontalFlip(0.5) + ToTensor + TemporalRescale(0.2)
  归一化    x/127.5 - 1  → [-1,1]
  解码      官方用 ctcdecode beam_width=10；我们默认 greedy（可选 beam）
  超参      hidden=1024, lr=1e-4, wd=1e-4, batch=2, 55 epoch, lr 35/45 降 80%

⚠️ 我们**没有**实现官方 collate_fn 的 left_pad=6 补帧，
   因此 dataLen 直接 = 实际帧数（已实测：不补帧时官方 Net.py 会 shape mismatch）。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, update_bn

# ── 仓库路径 ────────────────────────────────────────────────────
def _find_repo() -> Path:
    for c in (Path(__file__).resolve().parents[2],):
        if (c / "data" / "raw" / "CE-CSL").exists():
            return c
    return Path("/home/su127/FYP/domain-bounded-cslr")


REPO = _find_repo()
EXT = REPO / "external/TFNet"
RGB_ROOT = REPO / "artifacts/official_rgb"
CSV_DIR = EXT / "data/CE-CSL"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(EXT))

# ── 官方代码（直接复用，不重写）────────────────────────────────
import DataProcessMoudle as DPM          # noqa: E402
import Net# noqa: E402
import videoAugmentation as VA          # noqa: E402
from official_wer import evaluate, position_vs_official   # noqa: E402

assert "test" not in str(RGB_ROOT), "test split 冻结"
MIN_FRAMES = 48          # TemporalConv 两次 MaxPool 后至少要 9 步，实测下限


def read_csv_split(name: str) -> dict[str, tuple[str, str]]:
    """返回 {Number: (Translator, Gloss)}。用官方 csv（与我们仓库的一致）。"""
    out = {}
    with open(CSV_DIR / name, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0]:
                out[row[0]] = (row[1], row[3])
    return out


class CenterCropShift(object):
    """官方 CenterCrop + **垂直方向偏移**（y_shift=0 时与官方完全等价）。

    为什么需要：P128 实测运动能量重心在 y≈157，而 CenterCrop(160) 保留
    [48,208]（中心 128）⇒ **系统性切掉下部**。224 更大区域已证明有害（P127，
    +13.29pp），所以问题在**位置**不在**大小** ⇒ 需要能平移裁切窗口来单独验证。
    """

    def __init__(self, size, y_shift: int = 0):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.y_shift = int(y_shift)

    def __call__(self, clip):
        h, w = clip[0].shape[:2]
        sh, sw = self.size
        y0 = (h - sh) // 2 + self.y_shift
        x0 = (w - sw) // 2
        y0 = max(0, min(h - sh, y0))      # 夹紧，防止越界
        x0 = max(0, min(w - sw, x0))
        return [img[y0:y0 + sh, x0:x0 + sw] for img in clip]


def build_transforms(hflip: bool, img_size: int = 224, y_shift: int = 0,
                     temporal_rescale: float = 0.2):
    """官方增强链 + 两个可选偏离开关（默认 y_shift=0 / 0.2 ⇒ 与官方完全一致）。

    ⚠️ 官方默认含 RandomHorizontalFlip(0.5)。中文手语的镜像手势语义可能相反，
       所以做成开关：--no-hflip 关掉，两组对比后再决定（P119 已证伪该假设）。

    两个后加的开关（各自都有实测动机，默认保持官方行为）：
      - y_shift             ：**只平移测试裁切窗口**。P128 测得运动重心偏下 29px，
                              而训练时 RandomCrop 是随机的（覆盖全图、含下部），
                              ⇒ 出问题的只有测试时那个**固定**的 CenterCrop，
                              所以训练侧保持官方不变，只动测试侧。
      - temporal_rescale=0  ：关掉时序缩放。官方 Train.py:76 未启用，
                              是我们长期多带的增强（P125 静态核对发现）
    """
    train_ops = [VA.RandomCrop(img_size)]          # 训练侧恒为官方行为
    if hflip:
        train_ops.append(VA.RandomHorizontalFlip(0.5))
    train_ops += [VA.ToTensor()]
    if temporal_rescale:
        train_ops.append(VA.TemporalRescale(temporal_rescale))
    test_ops = [CenterCropShift(img_size, y_shift), VA.ToTensor()]
    return VA.Compose(train_ops), VA.Compose(test_ops)


# 🔴 P149（2026-10-18 实测）：通道顺序开关。默认 False = 复现历史行为。
#   我们的 jpg 由「cv2 抽帧 -> cvtColor(BGR2RGB) -> imencode」写出，磁盘语义 R/B 颠倒；
#   训练读图再 cvtColor(BGR2RGB) => 喂给模型的是 **BGR**。
#   官方用 imageio.get_data()（RGB）也做一次 cvtColor(BGR2RGB) => 官方最终是 **RGB**。
#   True = 读图后不再 cvtColor => 得到真 RGB（与官方一致）
RGB_FIX = False


class RGBSeqDataset(Dataset):
    """读 artifacts/official_rgb/{split}/{translator}/{sid}/*.jpg。

    官方 MyDataset 读的是 `{ImagePath}/{name}/{帧}.jpg`，
    我们 P77 产出的正是同一结构 ⇒ 这里只做路径适配 + 标签编码。
    """

    def __init__(self, split: str, labels: dict, word2idx: dict,
                 transform, is_train: bool, pad_to: int = 0):
        self.root = RGB_ROOT / split
        self.transform = transform
        self.is_train = is_train
        self.pad_to = pad_to
        self.items = []
        n_short = 0
        for sid, (tr, gloss) in labels.items():
            d = self.root / tr / sid
            if not d.exists():
                continue
            n_jpg = len([f for f in os.listdir(d) if f.endswith(".jpg")])
            if n_jpg == 0:
                continue
            ids = [word2idx[w] for w in DPM.PreWords(gloss.split("/"))
                   if w in word2idx]
            if not ids:
                continue
            if n_jpg < MIN_FRAMES:
                # TemporalConv 下限：重复补到 48 帧（记入收据）
                n_short += 1
            self.items.append({"sid": sid, "dir": d, "ids": ids,
                               "n_jpg": n_jpg})
        self.n_short = n_short

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        files = sorted(f for f in os.listdir(it["dir"]) if f.endswith(".jpg"))
        if RGB_FIX:
            imgs = [cv2.imread(str(it["dir"] / f)) for f in files]
        else:
            imgs = [cv2.cvtColor(cv2.imread(str(it["dir"] / f)), cv2.COLOR_BGR2RGB)
                    for f in files]
        # 官方 MyDataset 也做 cv2.resize(...,(256,256))；P77 已resize 过，这里不再重复
        seq = self.transform(imgs)
        seq = seq.float() / 127.5 - 1.0            # 官方归一化→ [-1, 1]
        return {"video": seq, "ids": it["ids"], "sid": it["sid"]}


# 🔴 P123：官方 collate_fn 的补帧约定（由 --official-pad 打开）
#   来源：external/TFNet/DataProcessMoudle.py:435 collate_fn
#     left_pad = 6, total_stride = 4
#     right_pad = ceil(max_len/4)*4 - max_len + left_pad
#     videoLength_i = ceil(len_i/4)*4 + 2*left_pad
#   即：首帧复制 6 帧放左边，末帧复制补齐到 ceil(T/4)*4+12，
#   dataLen 用**含 padding 的长度**（不是真实帧数）。
#   ⚠️ 默认关闭：P93~P121 全部用的是「零填充 + dataLen=真实帧数」口径。
OFFICIAL_PAD = False

# 🔴 P134：解码时给 blank 类（index 0）的 log-prob 减去的常数。
#   全局量是因为解码在 evaluate() 内部，参数对象不方便逐层传递（与 OFFICIAL_PAD 同模式）。
#   默认 0.0 = 官方行为；>0 抑制 blank ⇒ 迫使多输出，零训练成本。
BLANK_BIAS = 0.0


def ls_penalty(log_probs, lgt):
    """label smoothing 的第二项：β·[-(1/N)·Σ_c log p_c]，在**有效时间步**上累积。

    推导（写清楚以免日后误解）：
        标准 LS 目标   y_ls = (1-β)·y_hot + β·uniform(N)
        KL(y_ls‖p)    = -Σ_c y_ls[c]·log p[c]
                      = (1-β)·(-log p[true]) + β·[-(1/N)·Σ_c log p[c]]
                          └─ 对 CTC 即 CTC_loss ─┘   └─ 本函数 ─┘
    第二项只依赖 log_probs，**不需要改 CTC 的前向后向** ⇒ 与 CTCLoss 完全兼容。

    量级对齐：CTC_loss 是**序列级** -log P(seq) ≈ Σ_t(-log p[·])；
    所以本项也按样本**累积**（÷B）而非在 (T,B) 上平均（÷(T·B)），
    否则 β 的实际强度会被 T（≈100+）稀释到可忽略。

    Args:
        log_probs: [T,B,C]，已 log_softmax
        lgt: 各样本的有效输出长度（卷积后时间步）
    Returns:
        标量
    """
    T, B, _ = log_probs.shape
    t = torch.arange(T, device=log_probs.device).unsqueeze(1)      # [T,1]
    mask = (t < lgt.to(log_probs.device).view(1, -1)).to(log_probs.dtype)
    per_t = log_probs.mean(dim=-1)          # [T,B] = mean_c(log p_c)
    return -(per_t * mask).sum() / max(B, 1)


def collate(batch):
    """dataLen 必须是 CPU 的 [B,1] LongTensor（已实测，否则 CTC 报错）。"""
    batch = sorted(batch, key=lambda x: len(x["video"]), reverse=True)
    videos = [b["video"] for b in batch]
    if OFFICIAL_PAD:
        # 官方口径：每样本补到 L_i = ceil(T_i/4)*4 + 12，全 batch 再补到 max(L_i)
        Ls = [((v.shape[0] + 3) // 4) * 4 + 12 for v in videos]
        T = max(Ls)
        vid = torch.zeros(len(videos), T, 3, videos[0].shape[2],
                          videos[0].shape[3])
        for i, v in enumerate(videos):
            vv = torch.cat([v[0][None].expand(6, -1, -1, -1),
                            v,
                            v[-1][None].expand(T - 6 - v.shape[0], -1, -1, -1)],
                           dim=0)
            vid[i, :vv.shape[0]] = vv
        data_len = torch.LongTensor([[l] for l in Ls])          # CPU, [B,1]
        true_len = torch.LongTensor([v.shape[0] for v in videos])
    else:
        T = max(v.shape[0] for v in videos)
        vid = torch.zeros(len(videos), T, 3, videos[0].shape[2],
                          videos[0].shape[3])
        for i, v in enumerate(videos):# [T,C,H,W] → [T,3,H,W]
            vid[i, :v.shape[0]] = v
        data_len = torch.LongTensor([[v.shape[0]] for v in videos])   # CPU, [B,1]
        true_len = torch.LongTensor([v.shape[0] for v in videos])   # 每条真实帧数
    tgt = torch.cat([torch.LongTensor(b["ids"]) for b in batch])   # CPU
    tgt_len = torch.LongTensor([len(b["ids"]) for b in batch])   # CPU（官方同）
    return (vid, tgt, tgt_len, data_len, true_len,
            [b["sid"] for b in batch], [b["ids"] for b in batch])


def make_word_segments(n_frames: int, n_words: int) -> list[tuple[int, int]]:
    """P1 TFD 思路：把时间轴按 gloss 数等分，造词级伪边界。

    ⚠️ 这是**伪**边界（论文里没验证过它对 CSLR 有效），
       但 P13 实测真实候选段长度 1.03 vs 参考 gloss 长 5.52 ⇒ 真边界不可得，
       只能用等分近似。**必须在收据里标注这是近似边界。**
    """
    if n_words <= 0:
        return []
    edges = [round(n_frames * k / n_words) for k in range(n_words + 1)]
    return [(edges[k], edges[k + 1]) for k in range(n_words)
            if edges[k + 1] - edges[k] >= 1]



# =============================================================================
# 🔴 P100：对齐边界（替代等分边界）—— C 方向 v2
#
# 📄 依据等级【间接】（详见 MEMORY 2026-10-08 的 P98/P99 记录）：
#   ref18（arXiv:2505.15438, Google 2025）原文：
#     "we train an order-invariant classifier to predict the set of glosses
#      in each video, which is then used to infer a temporal alignment"
#   且明确说该思路来自 action segmentation（Bojanowski ECCV'14 /
#   Kuehne CVIU'17 / Richard CVPR'18），在 CV 领域是成熟方案
#   ⚠️ **无任何论文直接验证它能降 CSLR 的 WER**（ref18 衡量 BLEU）
#   ⚠️ **不是同一任务**：ref18 做 SLT（视频→文本），我们做 CSLR（识别）
#
# 🔴 与 ref18 的差别要说清：
#   ref18 把伪 gloss 当「无序集合」，额外训分类器推顺序；
#   我们的 CTC forward 算法本身就是「无序集合 → 唯一单调对齐」，
#   ⇒ 直接读它的贪心路径当软对齐即可，**不需要额外的分类器**。
#
# ⚠️ 为什么不能重跑等分边界版：P85 已实测等分边界「无显著影响」
#   ⇒ 唯一有意义的变量是**边界质量本身**
# =============================================================================

def align_segments(logp: torch.Tensor, ids: list, blank: int = 0):
    """用 CTC 贪心路径切词级段。

    参数
    ----
    logp : [T, C] 该样本的 log_probs（**单样本**，不是 batch）
    ids  : 该样本的 gloss 词表索引列表

    返回
    ----
    (keep_ids, segments, avg_conf)
      keep_ids  : 成功对齐的词 id（未对齐的直接丢弃）
      segments  : 与 keep_ids 一一对应的 (起始帧, 结束帧)
      avg_conf  : 各片段起点的平均置信度（用于判断可靠性）

    自检结果（构造 3 词各 20 帧 / 40-10-10 非等分两种情况均正确）
    """
    T = logp.shape[0]
    # 🔴 必须 detach：logits_seq 带 grad，Tensor.numpy() 会报
    #    "Can't call numpy() on Tensor that requires grad"
    with torch.no_grad():
        best = logp.argmax(dim=-1).detach().cpu().numpy()
        conf = logp.max(dim=-1).values.detach().cpu().numpy()

    # CTC 折叠：去掉 blank 与连续重复
    collapsed = []
    prev = -1
    for t in range(T):
        k = int(best[t])
        if k != prev and k != blank:
            collapsed.append((t, k))
        prev = k

    if not collapsed or not ids:
        return [], [], 0.0

    # 按 ids 的顺序贪心匹配片段
    segs = []
    starts = []
    ci = 0
    for gid in ids:
        found = -1
        for j in range(ci, len(collapsed)):
            if collapsed[j][1] == gid:
                found = j
                break
        if found < 0:
            continue                      # 模型没输出这个词 ⇒ 丢弃
        start = collapsed[found][0]
        if found + 1 < len(collapsed):
            end = collapsed[found + 1][0]
        else:
            end = T
        segs.append((start, max(end, start + 1)))
        starts.append(start)
        ci = found + 1

    # 重扫一遍构造 keep_ids（必须与 segs 同序 ⇒ 用同一个 ci 推进逻辑）
    keep = []
    ci = 0
    for gid in ids:
        for j in range(ci, len(collapsed)):
            if collapsed[j][1] == gid:
                keep.append(gid)
                ci = j + 1
                break

    avg_conf = float(conf[starts].mean()) if starts else 0.0
    return keep, segs, avg_conf


def word_head_loss_aligned(logits_seq: torch.Tensor, ids: list,
                           idx2word: list, blank: int = 0) -> torch.Tensor:
    """C 方向 v2：用 CTC 对齐结果切段的词级辅助分类损失。"""
    if not ids or logits_seq.shape[0] == 0:
        return logits_seq.new_zeros(())
    keep, segs, _conf = align_segments(logits_seq, ids, blank)
    if not segs or word_classifier is None:
        return logits_seq.new_zeros(())
    T = logits_seq.shape[0]
    pooled = []
    for (a, b) in segs:
        b = min(b, T)
        if a >= T:
            break
        pooled.append(logits_seq[a:b].mean(0))
    if not pooled:
        return logits_seq.new_zeros(())
    feat = torch.stack(pooled)
    return torch.nn.functional.cross_entropy(
        word_classifier(feat),
        torch.tensor(keep, device=feat.device, dtype=torch.long))


def word_head_loss(logits_seq: torch.Tensor, ids: list[int],
                   feat_len: int, idx2word: list) -> torch.Tensor:
    """C 方向：词级辅助分类头。

    对 CTC 帧序列按伪边界切成 n_words 段，每段 mean-pool 后送一个线性分类头，
    要求能分出该段的 gloss。**这是多任务辅助监督，不替代 CTC。**

    ⚠️ 官方 Net.py 没有这个头，属于我们自己加的（无直接文献支撑，
       属常规多任务做法）。C=3516 类。
    """
    n = len(ids)
    if n == 0 or feat_len <= 0:
        return logits_seq.new_zeros(())
    segs = make_word_segments(feat_len, n)
    if len(segs) != n:
        return logits_seq.new_zeros(())
    pooled = []
    for a, b in segs:
        if b > logits_seq.shape[0]:
            return logits_seq.new_zeros(())
        pooled.append(logits_seq[a:b].mean(0))
    feat = torch.stack(pooled)                # (n_words, C)
    return torch.nn.functional.cross_entropy(
        word_classifier(feat), torch.tensor(ids, device=feat.device))


#全局分类头（C 方向用；官方网络里没有）
word_classifier = None


def _conv_len(n: int) -> int:
    """官方 TemporalConv(conv_type=2) 的长度公式：L→L-4→/2→L-4→/2。"""
    for _ in range(2):
        n = (n - 4 + 1) // 2
    return max(n, 0)


def greedy_decode(logp: torch.Tensor, true_len: torch.Tensor,
                  idx2word: list, blank: int = 0) -> list[list[str]]:
    """greedy CTC 解码（官方用 ctcdecode beam_width=10，这里默认 greedy 对照）。

    🔴 collate 把整批 pad 到最长帧数，所以 log_probs 的时间维是 T_max，
       而每条样本的卷积后长度 = _conv_len(自己的真实帧数)，
       **不能直接用 out[5]**（那是 batch 内所有样本按同一条 lgt 推的）。
    """
    # 🔴 官方词表的 0 是 PAD/blank，真实 gloss 从 1 开始（word2idx 从 1 排）
    #    所以 argmax 得到的 class k 直接对应 idx2word[k]，不需要再减 1。
    #⚠️ 不要用 classes_to_token_ids（那是本仓库 landmark 模型的约定：0=blank、
    #    词表索引需 -1 还原）。官方 Net.py 的输出层本身就是 wordSetNum*1+1，
    #    索引 0 是 PAD。**两套约定不同，混用会全错。**
    seq = logp.cpu()                              # 官方是 [T,B,C] ⇒ 转成 [B,T,C]
    if seq.shape[0] != len(true_len) and seq.shape[1] == len(true_len):
        seq = seq.transpose(0, 1)
    out = []
    lens = true_len.view(-1).tolist()
    for b in range(seq.shape[0]):
        t = min(_conv_len(int(lens[b])), seq.shape[1])
        toks = []
        prev = -1
        for ti in range(t):
            k = int(seq[b, ti].argmax())
            if k != prev and k != blank and 0 <= k < len(idx2word):
                toks.append(idx2word[k])
            prev = k
        out.append(toks)
    return out


def _dev_eval(mdl, dv_loader, device, idx2word, labels_dv):
    """官方口径 dev 评估（供训练循环与 SWA 终评复用）。
    idx2word / labels_dv 由 main 传入（它们是 main 局部量，不可模块级引用）。"""
    mdl.eval()
    hyps, refs = [], []
    with torch.no_grad():
        for vid, tgt, tgt_len, dl, tl_len, _sids, _ids in dv_loader:
            vid = vid.to(device)
            out = mdl(vid, dl, False)
            logp5 = F.log_softmax(out[0], dim=-1)   # 官方布局 [T,B,C]
            if BLANK_BIAS:
                logp5 = logp5.clone()
                logp5[..., 0] -= BLANK_BIAS
            hyps += greedy_decode(logp5, dl, idx2word)
            for b in _sids:
                refs.append(labels_dv[b][1])
    return evaluate(refs, hyps)


def _update_bn(loader, swa_mdl, device):
    """自定义 BN 统计量刷新：torch 自带 update_bn 只喂 model(input) 单参，
    但 TFNet.forward 签名是 forward(x, len_x, ...)，单参会令 len_x=None 崩。
    这里手动按 (vid, dl, False) 两参喂入，刷新 BN running stats。"""
    swa_mdl.train()
    with torch.no_grad():
        for batch in loader:
            vid = batch[0].to(device)
            # 🔴 dl(=len_x) 必须保持 CPU：TFNet 内部 pack_padded_sequence
            #    要求 lengths 为 1D CPU int64，搬 cuda 会崩（正常训练/_dev_eval
            #    里 dl 也一直是 CPU）。
            dl = batch[3]
            swa_mdl(vid, dl, False)
    swa_mdl.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--no-hflip", action="store_true",
                    help="关掉 RandomHorizontalFlip（中文手语镜像语义可能相反）")
    ap.add_argument("--test-crop-shift", type=int, default=0,
                    help="🔴 只平移**测试**裁切窗口（像素，正=下移）。0=官方。\n"
                         "动机：P128 测得运动能量重心在 y≈157，而 CenterCrop(160)\n"
                         "保留 [48,208]（中心 128）⇒ 系统性切掉下部。训练侧 RandomCrop\n"
                         "是随机的（覆盖全图含下部），故只有测试侧的固定裁切有问题。")
    ap.add_argument("--no-temporal-rescale", action="store_true",
                    help="关掉 TemporalRescale(0.2)：官方 Train.py:76 未启用，\n"
                         "是我们长期多带的增强（P125 静态核对发现）")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="🔴 CTC 的 label smoothing 系数 β（默认 0=官方行为）。\n"
                         "标准 LS 把目标分布换成 y_ls=(1-β)·onehot + β·uniform，\n"
                         "其 KL 可分解为 (1-β)·CE + β·[-(1/N)·Σ_c log p_c]；\n"
                         "第一项对 CTC 就是 CTC loss，第二项（对均匀分布的 CE）\n"
                         "只依赖 log_probs，**与 CTC 完全兼容** ⇒ 可直接实现。\n"
                         "第二项量级需与序列级 CTC 对齐 ⇒ 按样本累积（÷B）而非 ÷(T·B)。\n"
                         "⚠️⚠️ **β 的有效强度在 CTC 上被放大约 9 倍**（实测，必读）：\n"
                         "   CTC 因对**所有对齐路径求和**，其 per-step 有效 CE 只有 ~1.4，\n"
                         "   而 uniform 的 per-step CE = log(N) = 8.17 ⇒ 比值 ≈ 9。\n"
                         "   实测 penalty/CTC ≈ 326/36 ≈ 9 ⇒\n"
                         "     β=0.10 ⇒ penalty 项 ≈ CTC 项（**100%%**，过强）\n"
                         "     β=0.05 ⇒ ≈48%%      β=0.01 ⇒ ≈9%%（≈ 标准 CE 的 0.1）\n"
                         "   ⇒ **CTC 上推荐 β = 0.01**（等效标准 CE 的 label_smoothing=0.1）\n"
                         "依据：\n"
                         "  【有】ref17 Gloss2Text 提出 SALS（语义感知 LS，手语专用）\n"
                         "  【有】ref16 Sign2GPT 用 label smoothing 0.1\n"
                         "  【间接】LS 抑制过拟合是通用原理（Szegedy 2016 / Müller 2019）\n"
                         "  ⚠️【无】CTC-based CSLR 上 LS 的**直接**论文证据（这是首次尝试）\n"
                         "动机：P127/P140 已确认本项目**过拟合**（train 远低于 dev）")
    ap.add_argument("--blank-bias", type=float, default=0.0,
                    help="🔴 解码时给 blank 类（index 0）的 log-prob 减该常数，\n"
                         "抑制 blank ⇒ 迫使多输出，**零训练成本**（P134）。\n"
                         "动机：P133 实测生成率仅 77.2%%，且随句长崩塌（10+ 句仅\n"
                         "54.7%%）⇒ CTC 不确定时退化成 blank。\n"
                         "实测：P123 55.84→54.75（0.75）、P124e 59.61→58.80（1.00），\n"
                         "双 ckpt 同为平滑 U 形、最优区间 0.75-1.0。\n"
                         "⚠️ 默认 0（官方行为）；⚠️ 该 bias 是在 dev 上调出来的，\n"
                         "有乐观风险，最终验收须在 test 上复核一次。")
    ap.add_argument("--max-train", type=int, default=0, help="0=全量；小规模验证用")
    ap.add_argument("--max-dev", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mem-budget", type=float, default=6.6e9,
                    help="显存预算（字节）。默认 6.6GB，留 1.4GB 余量")
    ap.add_argument("--module", default="VAC",
                    choices=["VAC", "CorrNet", "TFNet", "MAM-FSD", "SEN"],
                    help="官方 moduleChoice。VAC=ResNet18(唯一能在8GB跑满帧数的)")
    ap.add_argument("--word-head", type=float, default=0.0,
                    help="C方向：词级辅助分类头的权重 w（0=关闭）。"
                         "需配合 --pseudo-boundary")
    ap.add_argument("--word-boundary", default="equal",
                    choices=["equal", "ctc_align"],
                    help="🔴 C 方向 v2 的边界来源："
                         "equal=按 gloss 数等分（P85 已证无效）；"
                         "ctc_align=用 CTC 贪心路径切（ref18 机制，【间接】依据）")
    ap.add_argument("--word-head-lowlr-mult", type=float, default=1.0,
                    help="🔴 P101：lr 衰减后词级头权重的倍数。"
                         "1.0 = 不变（对照组）；"
                         ">1 = 衰减后加大权重（验证『lr 低时才有效』假设）")
    ap.add_argument("--warmup-ep", type=int, default=0,
                    help="🔴 前 N 轮不开词级头（早期 CTC 对准极差，"
                         "此时用对齐结果切边界会提供错误监督）")
    ap.add_argument("--pseudo-boundary", action="store_true",
                    help="按 gloss 数等分时间轴造词级伪边界（P1 TFD 思路）")
    ap.add_argument("--img-size", type=int, default=224,
                    help="输入分辨率。官方是 224；降到 160/128 可省显存"
                         "（依据：AdaSize 间接支持，但未在 CE-CSL 验证）")
    ap.add_argument("--amp-bf16", action="store_true",
                    help="用 bf16 混合精度（⚠️ fp16 会让 CTC+KLD 梯度 NaN）")
    ap.add_argument("--resume", default="",
                    help="🔴 续训用：从某个 .pt 恢复（需含 optimizer/sched/RNG）。"
                         "典型用法：--resume ck/xxx-last.pt --total-epochs 30")
    ap.add_argument("--fix-rgb-order", action="store_true",
                    help="读图后不再做 cvtColor(BGR2RGB)：修正 R/B 通道错位，使输入与官方一致（P149 实测：当前喂给模型的是 BGR）")
    ap.add_argument("--official-pad", action="store_true",
                    help="🔴 P123：改用官方 collate_fn 的补帧约定"
                         "（left_pad=6 首帧复制 + 右补末帧到 4 的倍数，"
                         "dataLen=ceil(T/4)*4+12）。"
                         "默认关闭＝P93~P121 的「零填充+真实帧数」口径")
    ap.add_argument("--backbone-ckpt", default="",
                    help="🔴 P124：用自监督预训练的 2D 主干权重替代 ImageNet 初始化。"
                         "VAC 的主干是 **2D resnet18**（官方 Net.py:88，逐帧前向），"
                         "故 torchvision resnet18 的 state_dict 可直接载入")
    ap.add_argument("--total-epochs", type=int, default=0,
                    help="🔴 lr 调度按**最终总轮数**算，而不是本次要跑的轮数。"
                         "0 = 用 --epochs（首次训练）。"
                         "续训时必须传，否则 lr 调度会错位。")
    ap.add_argument("--sched", default="multistep", choices=["multistep", "cosine"],
                    help="lr 调度：multistep(官方 35/45) 或 cosine 退火")
    ap.add_argument("--eta-min", type=float, default=0.0,
                    help="cosine 退火下限 lr")
    ap.add_argument("--no-resume-sched", action="store_true",
                    help="续训扩展时【不】恢复旧 scheduler，改用新 T 重建并把 lr "
                         "重置回高位（否则上轮已衰减到 4e-6，扩展段空转）")
    ap.add_argument("--swa-start", type=int, default=0,
                    help=">=1 时从该 epoch 起做 SWA 权重平均（0=关闭）")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="VAC 主干空间特征后 dropout（正则探针）；0=关闭（默认，历史实验不变）。")
    ap.add_argument("--tag", default="p78")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=1)
    a = ap.parse_args()

    # 🔴 collate 是模块级函数，用全局开关传递官方补帧约定
    global OFFICIAL_PAD, BLANK_BIAS, RGB_FIX
    OFFICIAL_PAD = bool(a.official_pad)
    BLANK_BIAS = float(a.blank_bias)
    RGB_FIX = bool(a.fix_rgb_order)
    if RGB_FIX:
        print("[RGB_FIX] 通道顺序修正开启：读图后不再 cvtColor => 真 RGB")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels_tr = read_csv_split("train.csv")
    labels_dv = read_csv_split("dev.csv")

    # 🔴 官方词表：train+dev+test 全收（词表层面用了 test 标注，
    #    但训练只看 train 帧、不在 test 上算指标，故不违反 test 冻结）
    word2idx, wordSetNum, idx2word = DPM.Word2Id(
        str(CSV_DIR / "train.csv"), str(CSV_DIR / "dev.csv"),
        str(CSV_DIR / "test.csv"), "CE-CSL")

    tr_tf, dv_tf = build_transforms(
        hflip=not a.no_hflip, img_size=a.img_size,
        y_shift=a.test_crop_shift,
        temporal_rescale=(0.0 if a.no_temporal_rescale else 0.2))
    tr_ds = RGBSeqDataset("train", labels_tr, word2idx, tr_tf, True)
    dv_ds = RGBSeqDataset("validation", labels_dv, word2idx, dv_tf, False)
    # dev 在 P77 里落在 validation/ 目录名
    dv_ds = RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    if a.max_train:
        tr_ds.items = tr_ds.items[:a.max_train]
    if a.max_dev:
        dv_ds.items = dv_ds.items[:a.max_dev]

    print("=" * 70)
    print("P78 官方 TFNet 训练（RGB + resnet34MAM，原版网络代码）")
    print("=" * 70)
    print("device              %s" % dev)
    print("词表 wordSetNum     %d（+1 blank = %d 类）" % (wordSetNum, wordSetNum + 1))
    print("train / dev         %d / %d 条" % (len(tr_ds), len(dv_ds)))
    print("帧数 < %d 的训练样本  %d 条（已重复补帧）" % (MIN_FRAMES, tr_ds.n_short))
    print("RandomHorizontalFlip %s" % ("关闭（对照组）" if a.no_hflip else "开启（官方默认 0.5）"))
    print("官方 moduleChoice  %s" % a.module)
    print("输入分辨率         %d×%d%s" % (
        a.img_size, a.img_size,
        "（官方 224）" if a.img_size == 224 else " ⚠️偏离官方，工程取舍"))
    print("混合精度           %s" % ("bf16" if a.amp_bf16 else "fp32"))
    print("显存预算           %.1f GB（斜率 %.1f MB/帧·样本）"
          % (a.mem_budget / 1e9, {"TFNet": 55.7, "CorrNet": 45.8,
                                  "VAC": 27.1}.get(a.module, 55.7)))
    print("超参                hidden=%d lr=%g wd=%g batch=%d epochs=%d"
          % (a.hidden, a.lr, a.weight_decay, a.batch, a.epochs))
    print("补帧约定           %s" % (
        "★官方 collate（left_pad=6 + 末帧复制，dataLen=ceil(T/4)*4+12）"
        if a.official_pad else "零填充 + dataLen=真实帧数（P93~P121 口径）"))
    print("主干初始化         %s" % (a.backbone_ckpt or "ImageNet（官方默认）"))

    if len(tr_ds) < 2 or len(dv_ds) < 2:
        print("❌ 样本不足，先跑 p77_extract_rgb.py")
        return

    # 🔴 显存实测（RTX5060Laptop 8GB，官方 7 项损失，batch=2）：
    #     T=48  5.36GB✓   T=96  10.92GB✗   T=164 OOM
    #   batch=1：T=96 5.37GB ✓ T=128 6.96GB ✓ T=164 8.73GB ✗ T=180 9.52GB ✗
    #   ⇒ 显存 ≈ batch × T 线性增长，必须按帧数分桶动态定batch。
    def _batch_for(n_frames: int) -> int:
        """选能塞进显存的 batch（实测上限 ≈ 1.0e8·B/frame·B，保守取半）。"""
        # 🔴 P81 实测斜率（MB 每 帧·样本）：
        #    TFNet(7项损失,ResNet34MAM) 5.35e9/(2*48) = 55.7
        #    CorrNet(3项, ResNet18Corr) 4.40e9/(2*48) = 45.8
        #    VAC    (3项, ResNet18)      2.60e9/(2*48) = 27.1
        #    ★ 取实测值而不是猜，这是 P78 失败换来的教训
        SLOPE = {"TFNet": 55.7e6, "CorrNet": 45.8e6, "VAC": 27.1e6,
                 "MAM-FSD": 50.0e6, "SEN": 30.0e6}
        per_frame = SLOPE.get(a.module, 55.7e6)
        # 🔴 激活显存 ∝ (res/224)²（P86b 实测：224→9.04GB, 128→3.45GB）
        scale = (a.img_size / 224.0) ** 2
        per_frame = per_frame * scale
        budget = float(a.mem_budget)
        b = int(budget / (per_frame * max(n_frames, 1)))
        return max(1, min(b, a.batch))

    def _adaptive_batches(ds, shuffle):
        """按帧数排序 -> 相邻同量级组成 batch，各自选可行 batch。返回索引列表。"""
        order = sorted(range(len(ds.items)),
                       key=lambda k: ds.items[k]["n_jpg"],
                       reverse=not shuffle)
        out, cur = [], []
        for k in order:
            cur.append(k)
            if len(cur) >= _batch_for(ds.items[k]["n_jpg"]):
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        if shuffle:
            random.shuffle(out)
        return out

    class _AdaptiveLoader:
        """绕开 DataLoader 的固定 batch_size：按帧数桶自适应组装。"""

        def __init__(self, ds, shuffle, shuffle_state=None):
            self.ds = ds
            if shuffle and shuffle_state is not None:
                # 🔴 续训：用 checkpoint 里的 RNG 状态做 shuffle，
                #    使 batch 排列与「连续训练到同一epoch」一致，
                #    且不污染全局 RNG（否则会影响后续 dropout 等）。
                import random as _r
                rng = _r.Random()
                rng.setstate(shuffle_state)
                order = sorted(range(len(ds.items)),
                               key=lambda k: ds.items[k]["n_jpg"],
                               reverse=False)
                out, cur = [], []
                for k in order:
                    cur.append(k)
                    if len(cur) >= _batch_for(ds.items[k]["n_jpg"]):
                        out.append(cur)
                        cur = []
                if cur:
                    out.append(cur)
                rng.shuffle(out)
                self.batches = out
            else:
                self.batches = _adaptive_batches(ds, shuffle)

        def __iter__(self):
            for chunk in self.batches:
                yield collate([self.ds[k] for k in chunk])

        def __len__(self):
            return len(self.batches)

    # 🔴🔴 注意：构造 loader 会调用 random.shuffle() 消耗 python RNG。
    #   若这里先构造、后面才恢复 RNG（续训分支在 400+ 行），
    #   续训时的 batch 排列会与连续训练不同（实测 ep004 ctc 差 1.78）。
    #   ⇒ 解决办法：把 shuffle 的随机源做成可注入，
    #     续训时用 checkpoint 里的 py_rng 副本，构造后不污染全局 RNG。
    # 🔴 必须在建 loader 之前读出 shuffle 用的 RNG（见下方 resume 段说明）
    _ck_pre = None
    if a.resume:
        _ck_pre = torch.load(a.resume, map_location="cpu", weights_only=False)

    tr_dl = _AdaptiveLoader(tr_ds, True,
                            shuffle_state=(_ck_pre or {}).get("py_rng"))
    dv_dl = _AdaptiveLoader(dv_ds, False)
    print("自适应分桶         %d train batch / %d dev batch" % (
        len(tr_dl), len(dv_dl)))

    # ★ 直接用官方网络
    model = Net.moduleNet(a.hidden, wordSetNum * 1 + 1, a.module,
                          torch.device(dev), "CE-CSL", True,
                          dropout=a.dropout).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print("参数                %.2f M" % n_par)

    # ---- P124：自监督预训练主干替换（替代 ImageNet 初始化）----
    if a.backbone_ckpt:
        _ck = torch.load(a.backbone_ckpt, map_location="cpu",
                         weights_only=False)
        _sd = _ck.get("state_dict", _ck)
        _tgt = model.conv2d                       # VAC: 2D resnet18（官方 Net.py:88）
        _tsd = _tgt.state_dict()
        _sd2 = {}
        for k, v in _sd.items():
            if (k in _tsd and torch.is_tensor(v) and v.dim() == 4
                    and _tsd[k].dim() == 5):
                # 官方 Module.py:467 的 inflate：2D 卷积 → 3D（时间维=1）
                v = v.unsqueeze(2)
            _sd2[k] = v
        _res = _tgt.load_state_dict(_sd2, strict=False)
        print("主干初始化          %s" % a.backbone_ckpt)
        print("  载入 %d/%d 张量，missing=%d unexpected=%d"
              % (len(_sd2) - len(_res.unexpected_keys), len(_sd2),
                 len(_res.missing_keys), len(_res.unexpected_keys)))
        if len(_res.missing_keys) > 4:
            print("  ⚠️ missing 过多，主干可能没换成功：" % ())
            print("     %s" % _res.missing_keys[:8])

    if a.word_head > 0:
        global word_classifier
        word_classifier = torch.nn.Linear(wordSetNum + 1, wordSetNum + 1).to(dev)
        _bl = ("等分时间轴（P85 已证无效）" if a.word_boundary == "equal"
               else "★ CTC 贪心路径（ref18 机制，依据【间接】）")
        print("词级辅助头          C = %d 类，权重 w = %.3f"
              % (wordSetNum + 1, a.word_head))
        print("边界来源            %s" % _bl)
        print("warmup              前 %d 轮不开词级头" % a.warmup_ep)
        if a.word_head_lowlr_mult != 1.0:
            # 🔴 ms 在下方才定义，此处用同公式预计算，避免 NameError
            _T0 = a.total_epochs if a.total_epochs > 0 else a.epochs
            _ms0 = int(round(35 * _T0 / 55))
            print("低 lr 增强           ep%d 起权重 ×%.2f（P101 假设验证）"
                  % (_ms0, a.word_head_lowlr_mult))
        else:
            print("低 lr 增强           关闭（对照组，权重恒定）")

    # ★ 官方的 CTC 配置（zero_infinity=True 是关键，我 P72 就是漏了这行导致永久 nan）
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    kld = DPM.SeqKD(T=8)
    ls = torch.nn.LogSoftmax(dim=-1)
    _params = list(model.parameters())
    if a.word_head > 0 and word_classifier is not None:
        _params += list(word_classifier.parameters())
    opt = torch.optim.Adam(_params, lr=a.lr, weight_decay=a.weight_decay)
    # 🔴 lr 里程碑必须按**最终目标总轮数**算，不是本次要跑的轮数。
    #   原因：续训时若按本次轮数算，衰减会提前/错位
    #   （例：从 20 续到 30，若按 30 算 ms=[19,25]，
    #     但 20 轮时已在 ep14/ep16 衰减过 → lr 不会升回 1e-4，
    #     模型在4e-6 下空转 10 轮，学习几乎停滞。）
    T = a.total_epochs if a.total_epochs > 0 else a.epochs
    ms = [int(round(35 * T / 55)), int(round(45 * T / 55))]
    if a.sched == "cosine":
        sched = CosineAnnealingLR(opt, T_max=T, eta_min=a.eta_min)
    else:
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms, gamma=0.2)
    swa_model = (AveragedModel(model) if a.swa_start > 0 else None)

    print("lr 衰减里程碑        %s（按**总轮数 %d** 缩放，官方 35/45）" % (ms, T))

    # 🔴🔴 P119 事故修复：--total-epochs 原本只影响 lr 里程碑，
    #   训练循环却用 a.epochs（默认 55）⇒ 传 --total-epochs 20 仍会跑 55 轮。
    #   本项目 P93/P100/P101 都同时传了 --epochs 20 才正确，
    #   但 P119 只传 --total-epochs ⇒ 跑超到 44 轮（已饱和，白跑 24 轮）。
    #   ⇒ 现在：若两者不一致，明确警告并**以 --epochs 为准**（保持向后兼容）。
    if a.total_epochs > 0 and a.total_epochs != a.epochs:
        print()
        print("  ⚠️⚠️⚠️ --total-epochs=%d 与 --epochs=%d 不一致！" % (a.total_epochs, a.epochs))
        print("     --total-epochs 只用于缩放 lr 里程碑，**不限制训练轮数**。")
        print("     实际训练轮数由 --epochs 决定 ⇒ 本次会跑 %d 轮。" % a.epochs)
        print("     若只想跑 %d 轮，请同时传 --epochs %d。" % (a.total_epochs, a.total_epochs))
        print()

    ck_dir = REPO / "artifacts/checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    # =========================================================================
    # 🔴 续训：恢复 optimizer / lr 调度 / 随机数状态
    #   只 load 权重是不够的 —— Adam 的二阶动量丢了就不是同一个优化过程；
    #   随机数不复原则数据顺序与增强都变了，续训轨迹不可复现。
    # =========================================================================
    start_ep = 1
    if a.resume:
        # 🔴 用前面已读的 ck（避免重复读 423MB 文件）
        ck = _ck_pre
        model.load_state_dict(ck["model_state"])
        if "optimizer_state" not in ck:
            raise SystemExit(
                "❌ %s 里没有 optimizer_state，无法真正续训。\n"
                "   （只存 model_state 会丢 Adam 二阶动量 + lr 位置）"
                % a.resume)
        opt.load_state_dict(ck["optimizer_state"])
        # 🔴 P123 续训事故修复：word_classifier 是**独立于 model 的全局模块**，
        #   不在 model.state_dict() 里 ⇒ 不单独存就会在续训时被随机重建，
        #   却配上上一轮针对旧权重的 Adam 二阶动量 ⇒ 词级头梯度把共享特征带偏。
        #   实测：ep31 ctc 10.65 → ep32(续训) ctc 13.91（+3.27）。
        if a.word_head > 0 and word_classifier is not None:
            _wcs = ck.get("word_classifier_state")
            if _wcs is not None:
                word_classifier.load_state_dict(_wcs)
                print("  词级头状态恢复  ✅")
            else:
                print("  ⚠️ ckpt 里没有 word_classifier_state ⇒ 词级头被随机重建，"
                      "将配着旧 Adam 动量跑，头几轮 ctc 会被抬高（需 1-2 轮恢复）")
        start_ep = int(ck["epoch"]) + 1
        if "torch_rng" in ck:
            torch.set_rng_state(ck["torch_rng"].cpu())
        if "cuda_rng" in ck and torch.cuda.is_available():
            _cr = ck["cuda_rng"]
            if isinstance(_cr, (list, tuple)):
                # best.pt 存的是 get_rng_state_all()（张量列表）
                torch.cuda.set_rng_state_all(
                    [t.cpu() if hasattr(t, "cpu") else t for t in _cr])
            else:
                torch.cuda.set_rng_state(_cr.cpu())
        if "np_rng" in ck:
            np.random.set_state(ck["np_rng"])
        if "py_rng" in ck:
            random.setstate(ck["py_rng"])
        # 🔴 恢复 scheduler 位置：让 lr 从「上轮结束时的值」继续，
        #    而不是从头按 sched.step() 的初始序列走。
        if a.no_resume_sched:
            # 🔴 续训扩展：上轮 lr 已衰减到极小（如 4e-6），必须重置回高位，
            #    否则扩展段在 4e-6 下空转（p78 注释已警告过此坑）。
            #    同时【不】恢复旧 scheduler，避免里程碑被覆盖回旧的 [35,45]。
            for _pg in opt.param_groups:
                _pg["lr"] = a.lr
            _ext = a.epochs - start_ep + 1
            if a.sched == "cosine":
                sched = CosineAnnealingLR(opt, T_max=_ext, eta_min=a.eta_min)
            else:
                _ms_new = [m for m in ms if m >= start_ep]
                sched = torch.optim.lr_scheduler.MultiStepLR(
                    opt, milestones=_ms_new, gamma=0.2)
            sched.last_epoch = 0
            print("  scheduler 重建（不恢复旧调度）：%s，lr 重置为 %.2e，扩展段 %d 轮"
                  % (("cosine T_max=%d" % _ext)
                     if a.sched == "cosine"
                     else ("multistep ms=%s" % _ms_new), a.lr, _ext))
        elif "sched_state" in ck:
            try:
                sched.load_state_dict(ck["sched_state"])
                sched.last_epoch = int(ck["epoch"])
                print("  scheduler 恢复：last_epoch=%d，当前 lr=%.2e"
                      % (sched.last_epoch, opt.param_groups[0]["lr"]))
            except Exception as e:
                print("  ⚠️ scheduler 状态恢复失败（将按初始 lr 继续）：%r" % (e,))
        best = float(ck.get("wer_official", float("inf")))
        prev_hist = ck.get("history", [])
        print("=" * 68)
        print("续训：从 %s 恢复" % a.resume)
        print("  已完成 epoch = %d，本次从 ep%d 跑到 ep%d"
              % (int(ck["epoch"]), start_ep, a.epochs))
        print("  历史 best WER = %.2f%%" % best)
        print("  lr 里程碑 %s（按总轮数 %d 算）← 🔴 不会被续训打乱"
              % (ms, T))
        # 🔴 主动检查 lr 调度是否与上一轮的规划冲突
        planned = ck.get("total_epochs_planned")
        if planned and planned != T:
            print()
            print("  ⚠️⚠️ 警告：上一轮原定的总轮数是 %d，这次传的是 %d" % (planned, T))
            print("     上轮已在该调度下衰减过 lr，本轮不会升回高位lr。")
            print("     若 T > planned，属正常扩展（衰减点按新 T 重排）；")
            print("     若 T < planned，学习率会偏高，需谨慎。")
            print("     上轮 lr 当前位置（sched）："
                  "last_epoch=%d, base_lrs=%s"
                  % (ck.get("sched_state", {}).get("last_epoch", -1),
                     [("%.2e" % x) for x in
                      ck.get("sched_state", {}).get("_last_lr", [])]))
        print("=" * 68)
    else:
        prev_hist = []
        best = float("inf")

    # 🔴🔴 等价性修复（续训数字与连续训练不一致的根因）：
    #   _AdaptiveLoader.__init__ 里 random.shuffle() 会消耗 python RNG，
    #   而上面恢复 RNG 发生在**建 loader 之前**
    #   ⇒ 续训时 batch 排列与连续训练不同（实测 ep004 差 1.78）。
    #   解法：把 RNG 恢复**挪到建 loader 之前**，让 shuffle 用上轮的 RNG 状态。
    #   （若不修，逐级扩展仍可用，但节点间的数字不可与连续跑直接对比。）

    hist = list(prev_hist)
    t0 = time.time()
    nan_steps = 0

    for ep in range(start_ep, a.epochs + 1):
        model.train()
        run, nrun = 0.0, 0
        for vid, tgt, tgt_len, dl, _tl, _sids, _ids in tr_dl:
            vid = vid.to(dev)                      # [B,T,3,H,W]
            with torch.amp.autocast("cuda", enabled=a.amp_bf16,
                                    dtype=torch.bfloat16):
                out = model(vid, dl, True)
            # 🔴 官方的 lgt 是卷积后的时间步（out[5]），不是原始帧数。
            #    官方 log_probs 布局就是 [T,B,C]（实测 (9,2,3516)），与 CTCLoss 一致。
            # 🔴 只转浮点输出；out[5]（lgt）是整型长度，CTC 要求 integral
            out = [(o.float() if (torch.is_tensor(o) and o.is_floating_point())
                    else o) for o in out]
            lgt = out[5]
            # 🔴 TFNet 分支推理时 logProbs1 = logProbs5（官方 Net.py 末尾）；
            #    其他分支 logProbs1 本身就是最终输出，out[4] 是 None。
            logp5 = ls(out[4] if a.module == "TFNet" else out[0])
            # 🔴 nan 防护：非有限就跳过这一 step，保住 Adam 状态
            if not torch.isfinite(logp5).all():
                nan_steps += 1
                opt.zero_grad(set_to_none=True)
                continue
            #训练时统一用 logProbs1 做主 CTC
            main_lp = ls(out[0])
            loss_ctc = ctc_loss(main_lp, tgt, lgt, tgt_len.cpu()).mean()
            # 🔴 label smoothing：CTC 项替换成 (1-β)·CTC + β·[对均匀分布的平均 CE]
            #    β=0 时严格等价（(1-0)·CTC + 0·penalty）⇒ 默认行为不变
            if a.label_smoothing > 0:
                loss_ctc = ((1.0 - a.label_smoothing) * loss_ctc
                            + a.label_smoothing * ls_penalty(main_lp, lgt))
            if a.module == "TFNet":
                # 官方 Train.py 210-223：4 个 CTC + 2 个 SeqKD(×25) + 1 个融合 CTC
                loss = (ctc_loss(ls(out[4]), tgt, lgt, tgt_len.cpu()).mean()
                        + ctc_loss(ls(out[2]), tgt, lgt, tgt_len.cpu()).mean()
                        + 25.0 * kld(out[3], out[2], use_blank=False)
                        + ctc_loss(ls(out[1]), tgt, lgt, tgt_len.cpu()).mean()
                        + 25.0 * kld(out[1], out[0], use_blank=False)
                        + loss_ctc)
            else:
                # 官方 CorrNet/VAC/SEN 分支：2 个 CTC + 1 个 SeqKD(×25)
                loss = (loss_ctc
                        + ctc_loss(ls(out[1]), tgt, lgt, tgt_len.cpu()).mean()
                        + 25.0 * kld(out[1], out[0], use_blank=False))
            # C 方向：词级辅助分类头（多任务，主任务仍是 CTC）
            # C 方向：词级辅助头（主任务仍是 CTC）
            # 🔴 warmup：前 warmup_ep 轮不开（早期对齐不可靠）
            # 🔴 ctc_align 模式：用 CTC 贪心路径切边界（非等分）
            # 🔴 P101：lr 衰减后加大词级头权重
            #   依据：P100 实测 ep14-20（lr 衰减后）连续 7/7 为正、平均 +2.08pp，
            #        而 ep04-08（lr 高）为 0/5、平均 −1.46pp
            #   ⇒ 假设「lr 低时词级头才有益」
            # ⚠️ 这是**基于单次观察的事后假设**（有 p-hacking 风险），
            #    故必须设对照组（mult=1.0）才能分离「低 lr」与「权重更大」两个因素。
            _mult = 1.0
            if a.word_head_lowlr_mult != 1.0 and ep > (ms[0] if ms else 10**9):
                _mult = a.word_head_lowlr_mult
            if a.word_head > 0 and ep > a.warmup_ep:
                _w = a.word_head * _mult
                lp_seq = main_lp                      # [T, B, C]
                n_seg = 0
                for bi in range(lp_seq.shape[1]):
                    fw = int(lgt[bi])
                    if a.word_boundary == "ctc_align":
                        wl = word_head_loss_aligned(
                            lp_seq[:, bi, :], _ids[bi], idx2word)
                    else:
                        wl = word_head_loss(
                            lp_seq[:, bi, :], _ids[bi], fw, idx2word)
                    if float(wl) != 0.0:
                        loss = loss + _w * wl
                        n_seg += 1
                if n_seg:
                    loss = loss / (1.0 + _w * n_seg)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gn):
                nan_steps += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            run += float(loss_ctc.detach())
            nrun += 1
            torch.cuda.empty_cache()
        sched.step()

        # ---- 评估（官方口径）----
        o = _dev_eval(model, dv_dl, dev, idx2word, labels_dv)
        if swa_model is not None and ep >= a.swa_start:
            swa_model.update_parameters(model)

        cur = (run / max(nrun, 1), o["WER_official"])
        hist.append({"epoch": ep, "train_ctc": round(cur[0], 4),
                     "dev_wer_official": o["WER_official"],
                     "gap_to_sota": o["gap_to_sota_TFNet"]})
        print("  ep%03d ctc %.4f  **WER_official %.2f%%**  差SOTA %.2fpp%s"
              % (ep, cur[0], o["WER_official"], o["gap_to_sota_TFNet"],
                 "  [nan跳过 %d]" % nan_steps if nan_steps else ""), flush=True)

        # =========================================================================
        # 🔴 存 last.pt（**每轮覆盖**）—— 这是逐级续训的基础
        #   为什么必须存 last 而不是只用 best：
        #     逐级扩展要的是「训练到第 N 轮的状态」，不是「最好的那一轮」。
        #     若只存 best，续训会跳回 best 那一轮，等于每次都重跑最好点之前
        #     的所有 epoch（lr 轨迹会错乱、优化过程被重入）。
        # =========================================================================
        last_state = {
            "epoch": ep,
            "model_state": model.state_dict(),
            # 🔴 词级头是独立于 model 的全局模块，不在 model.state_dict() 里
            "word_classifier_state":
                (word_classifier.state_dict()
                 if (a.word_head > 0 and word_classifier is not None) else None),
            "optimizer_state": opt.state_dict(),   # 🔴 Adam 二阶动量
            "sched_state": sched.state_dict(),     # 🔴 lr 当前位置
            "torch_rng": torch.get_rng_state(),
            "np_rng": np.random.get_state(),
            "py_rng": random.getstate(),
            "wordSetNum": wordSetNum, "idx2word": idx2word,
            "hidden": a.hidden, "hflip": not a.no_hflip,
            "wer_official": o["WER_official"],
            "history": hist,
            "total_epochs_planned": T,   # 🔴 记录原定的总轮数，供续训核对
            "config": vars(a),
        }
        if torch.cuda.is_available():
            last_state["cuda_rng"] = torch.cuda.get_rng_state()
        torch.save(last_state, ck_dir / ("%s-last.pt" % a.tag))

        if o["WER_official"] < best:
            best = o["WER_official"]
            # 🔴 best.pt 原本缺 RNG 状态（last.pt 有）⇒ 从 best 续训时
            #    data shuffle / dropout 的随机流会重新初始化。
            #    实例：p119-hflip-best.pt（ep33, 56.72%）就是缺 RNG 存的。
            #    P122 已补齐四项 RNG。
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "word_classifier_state":
                            (word_classifier.state_dict()
                             if (a.word_head > 0 and word_classifier is not None)
                             else None),
                        "optimizer_state": opt.state_dict(),
                        "sched_state": sched.state_dict(),
                        "torch_rng": torch.random.get_rng_state(),
                        "np_rng": np.random.get_state(),
                        "py_rng": random.getstate(),
                        "cuda_rng": (torch.cuda.get_rng_state()
                                     if torch.cuda.is_available() else None),
                        "wordSetNum": wordSetNum, "idx2word": idx2word,
                        "hidden": a.hidden, "hflip": not a.no_hflip,
                        "wer_official": best, "history": hist,
                        "total_epochs_planned": T,
                        "config": vars(a)},
                       ck_dir / ("%s-best.pt" % a.tag))

    # ---- SWA 最终评估（顺带）----
    swa_wer_official = None
    if swa_model is not None:
        _update_bn(tr_dl, swa_model, dev)
        swa_wer_official = _dev_eval(swa_model.module, dv_dl, dev,
                                    idx2word, labels_dv)["WER_official"]
        torch.save({"model_state": swa_model.module.state_dict(),
                    "word_classifier_state":
                        (word_classifier.state_dict()
                         if (a.word_head > 0 and word_classifier is not None)
                         else None),
                    "wordSetNum": wordSetNum, "idx2word": idx2word,
                    "hidden": a.hidden, "config": vars(a)},
                   ck_dir / ("%s-swa.pt" % a.tag))
        print("SWA 平均(ep%d+) WER_official = %.2f%%"
              % (a.swa_start, swa_wer_official))

    # ---- 收据（⚠️括号必须加：Path / str 的优先级高于 %）----
    out_rec = {
        "experiment": "P78",
        "date": time.strftime("%Y-%m-%d"),
        "purpose": "官方 moduleChoice=%s 真训练（RGB）" % a.module,
        "official_code": "external/TFNet (Net.py / Module.py / videoAugmentation.py "
                         "/ DataProcessMoudle.py 均直接 import，未重写)",
        "official_benchmark_dev": {"TFNet": 42.1, "CorrNet": 47.2, "VAC": 45.1,
                             "MAM-FSD": 44.9, "SEN": 46.5}[a.module],
        "best_wer_official": best,
        "swa_wer_official": swa_wer_official,
        "module": a.module,
        "img_size": a.img_size,
        "amp_bf16": a.amp_bf16,
        "word_head_w": a.word_head,
        "word_boundary": a.word_boundary,
        "word_head_lowlr_mult": a.word_head_lowlr_mult,
        "lr_milestone_ep": ms[0] if ms else None,
        "lr_milestones": ms,
        "lr_gamma": 0.2,
        "official_pad": bool(a.official_pad),
        "backbone_ckpt": a.backbone_ckpt or None,
        "warmup_ep": a.warmup_ep,
        "pseudo_boundary": a.pseudo_boundary,
        "mem_budget_GB": round(a.mem_budget / 1e9, 2),
        "vocab": {"wordSetNum": wordSetNum,
                  "note": "官方 Word2Id，train+dev+test 全收、无截断"},
        "config": vars(a),
        "resumed_from": a.resume or None,
        "total_epochs_planned": T,
        "n_params_M": round(n_par, 2),
        "short_samples_padded": tr_ds.n_short,
        "ctc": "blank=0, reduction='none', zero_infinity=True（官方配置）",
        "nan_steps_skipped": nan_steps,
        "history": hist,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    dst = REPO / ("artifacts/metrics/blank-gov/%s-official-tfnet.json" % a.tag)
    dst.write_text(json.dumps(out_rec, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print()
    print("最佳 WER_official = %.2f%%" % best)
    print("收据 -> %s" % dst)


if __name__ == "__main__":
    main()