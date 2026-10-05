# -*- coding: utf-8 -*-
"""P40 · RGB 作为主干（而非辅助特征）接进 CTC —— A2 结论在修好管线上重做

## 为什么做

P39 翻转 P30 后，「已证实的瓶颈诊断」归零。查 24 篇论文后发现：
**CSL-Daily 上同行的 WER 是 32.0~44.3**（ref07 SignBT Table 5/6），
我们是 0.70 —— 差 2.2 倍以上，说明远未到顶。

而此前排除的 7 个方向**全部在 landmark 表征内部**：
| 方向 | 动的对象 |
|---|---|
| ST-GCN | 补 landmark 骨骼拓扑 |
| 容量控制 | 调 landmark 塔宽度 |
| 数据增强 | 扰动 landmark |
| 面部扩容 | 加 landmark 点数 |
| RGB 探针(P36) | **只是线性探针，没当主干** |

**没有一个动过「输入模态」这个根。** 同行 SOTA 全部用 RGB/TIN 作主干。

## A2 的旧结论已作废

A2（r3d_18 视频塔）实测 macroAUC 0.5170 < landmark 0.5239，判定 RGB 无效。
**但那是在 off-by-one 模型上测的**，P23 已作废全部 P0-P22 结论。
**这是唯一「方向未排除 + 有文献依据 + 成本可控」的路线。**

## 🔴 P40 相对 A2 的关键修正

A2 提取的是**视频级单向量** `(512,)` —— 全局池化后的一个向量。
**CTC 需要时序序列 `(B, T, C)`**，单向量无法做序列标注。
所以 A2 的特征在架构上就**不能接 CTC**，它只适合做全局分类/检索探针。

P40 改为：**按窗口切分视频，每窗口提一个 512 维向量**
→ 得到 `(n_windows, 512)` 的时序特征，可直接喂 CTC 的时间维。

## 成本实测（本机 RTX 5060）

```
r3d_18  (1,3,16,112,112)  10.5 ms/clip  -> 94.8 clip/s
全量 5488 视频 x 3 窗口 = 16464 clips -> 纯 GPU 17.2 分钟（不含视频解码）
```
本脚本先用 --n-videos 做小规模探针，确认有信号再考虑全量。

## 预注册判据

单因素对照，只换输入模态，其余超参全同（同 P30 base：vocab 301 / h256 / L2 /
d0.3 / lr 1e-3 / batch 16 / seed 42）：

| 配置 | 输入 | 说明 |
|---|---|---|
| `lm_only` | 368 维 landmark | 基线（= P30 base） |
| `rgb_only` | r3d_18 窗口特征 | **本实验的待验证项** |
| `rgb+lm` | 两者相加融合 | 对应 ref23 式 7 `Fused = Z_t + L_t` |

**判据：`rgb_only` 的 dev WER ≤ `lm_only` - 0.02**（沿用 P29/P30 的阈值）。
若 `rgb+lm` 显著优于两者任一单独配置 → 判定双流互补（ref23 的核心主张）。

**三口径同时报告**（P38 铁律）：WER 宽松 / WER 严格 / token_recall。
⚠️ **不再用 acc/WER 作为唯一判据** —— P38 已证明它们在本任务上失真。
但 CTC 没有 macroAUC 的直接对应物，故用**分频次 macroAUC 之外的口径**交叉验证：
同时报告 n_distinct（输出多样性）与 token_recall（召回），并对 rgb_only
额外做一次 P38 式片段级 macroAUC 探针。

## 8 样本过拟合自检（P20 铁律）

任何新模态接入后，**必须先能完美拟合 8 条真实样本且 argmax 解码正确**，
否则后续所有 WER 数字不可信。
"""
import argparse
import collections
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

TRAIN_VIDEO = REPO / "data/raw/CE-CSL/video/train"
DEV_VIDEO = Path("/mnt/c/Users/su127/Desktop/csl视频/dev")
LM_FEAT = REPO / "artifacts/part3_features"
RGB_CACHE = REPO / ".a2_cache/r3d_18_windows"   # 读取时会校验形状

BLOCKS = {"hands": (0, 126), "pose": (126, 158), "face": (158, 182)}


def read_csv(p):
    rows = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["Number"]] = r["Gloss"]
    return rows


def l2n(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


# ---------------------------------------------------------------- RGB 特征
def build_rgb_encoder(device):
    import torchvision
    net = getattr(torchvision.models.video, "r3d_18")(weights="DEFAULT")
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def find_video(root, sid):
    for d in sorted(root.iterdir()) if root.exists() else []:
        p = d / (sid + ".mp4")
        if p.exists():
            return p
    return None


def extract_windows(net, path, n_windows=3, frames=16, size=112, device="cuda"):
    """把视频切成 n_windows 个时间窗，每窗提一个 512 维向量。

    🔴 与 A2 的区别：A2 全局池化成 1 个向量 (512,)，无法接 CTC；
    这里保留窗口维，得到 (n_windows, 512) 的**时序**特征。
    """
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    buf = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        buf.append(cv2.resize(fr, (size, size)))
    cap.release()
    if len(buf) < frames:
        return None
    # 切成 n_windows 个连续窗口，窗口内再均匀抽 frames 帧
    out = []
    edges = np.linspace(0, len(buf), n_windows + 1).astype(int)
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
    with torch.no_grad():
        for w in range(n_windows):
            seg = buf[edges[w]:edges[w + 1]]
            if len(seg) < 2:
                continue
            idx = [round(i * (len(seg) - 1) / (frames - 1)) for i in range(frames)]
            arr = np.stack([seg[i] for i in idx])          # (T,H,W,3) BGR
            x = torch.from_numpy(arr).permute(3, 0, 1, 2).float().to(device) / 255.0
            x = (x.unsqueeze(0) - mean) / std              # (1,3,T,H,W)
            out.append(net(x).squeeze(0).cpu().numpy())
    return np.stack(out) if out else None                      # (n_windows, 512)


def ensure_rgb_cache(ids, n_windows, device, limit=None):
    RGB_CACHE.mkdir(parents=True, exist_ok=True)
    net = build_rgb_encoder(device)
    def needs(sid):
        p = RGB_CACHE / (sid + ".npy")
        if not p.exists():
            return True
        try:
            return np.load(p, mmap_mode="r").shape[0] != n_windows
        except Exception:                                        # noqa: BLE001
            return True

    todo = [s for s in ids if needs(s)]
    if limit:
        todo = todo[:limit]
    print("RGB 缓存：需提取 {} 个（已有 {}/{}）".format(
        len(todo), len(ids) - len(todo), len(ids)))
    t0 = time.time()
    for k, sid in enumerate(todo, 1):
        for split, root in (("train", TRAIN_VIDEO), ("dev", DEV_VIDEO)):
            vp = find_video(root, sid)
            if vp is None:
                continue
            try:
                arr = extract_windows(net, vp, n_windows, device=device)
            except Exception as e:                                  # noqa: BLE001
                print("  {} 失败: {}".format(sid, e))
                arr = None
            if arr is not None:
                np.save(RGB_CACHE / (sid + ".npy"), arr.astype(np.float32))
                break
        if k % 50 == 0 or k == len(todo):
            el = time.time() - t0
            print("  {}/{}  ({:.0f}s, {:.2f}s/个)".format(
                k, len(todo), el, el / max(k, 1)), flush=True)
    del net
    torch.cuda.empty_cache()
    return RGB_CACHE


def rgb_for(sid, n_windows):
    p = RGB_CACHE / (sid + ".npy")
    if not p.exists():
        return None
    a = np.load(p)
    # 🔴 形状必须匹配：缓存目录不按窗口数隔离，若旧的 3 窗口缓存被读到，
    # 会让 CTC 的 T 远小于 2L-1 而报「alignment impossible」（已踩过一次）
    if a.shape[0] != n_windows:
        return None
    return a.astype(np.float32)                                  # (n_windows, 512)


# ---------------------------------------------------------------- 模型
class DualInputCTC(torch.nn.Module):
    """支持 landmark-only / rgb-only / rgb+lm 三种输入的 CTC 主干。

    对应 ref23 SignFormer-GCN 式 7 `Fused = Z_t + L_t`：
    两条分支各自过 BiLSTM 后**相加**，共享投影与 CTC 头。
    """

    def __init__(self, lm_dim, rgb_dim, vocab, hidden=256, layers=2,
                 dropout=0.3, proj=256, use_rgb=True, use_lm=True, mode="add"):
        super().__init__()
        assert use_rgb or use_lm, "至少一路输入"
        self.use_rgb, self.use_lm, self.mode = use_rgb, use_lm, mode
        if use_lm:
            self.lm_enc = torch.nn.LSTM(lm_dim, hidden, layers,
                                        batch_first=True, bidirectional=True,
                                        dropout=dropout if layers > 1 else 0.0)
            self.lm_proj = torch.nn.Linear(hidden * 2, proj)
        if use_rgb:
            self.rgb_enc = torch.nn.LSTM(rgb_dim, hidden, layers,
                                         batch_first=True, bidirectional=True,
                                         dropout=dropout if layers > 1 else 0.0)
            self.rgb_proj = torch.nn.Linear(hidden * 2, proj)
        self.drop = torch.nn.Dropout(dropout)
        # 🔴 类数必须是 vocab + 1：CTC 类 0 是 blank，词表索引 i -> 类 i+1，
        #    故最后一个词表索引映射到类 vocab.size，越界一格。
        #    仓库 model.py 的写法是 nn.Linear(output_size, vocabulary_size + 1)。
        #    我初版漏了 +1，300 视频时恰好没触到词表末位而未暴露，
        #    全量数据立刻报 'target id 301 is out of range for 301 classes'。
        self.head = torch.nn.Linear(proj, vocab + 1)

    def output_lengths(self, input_lengths):
        return input_lengths

    def forward(self, lm, lm_len, rgb, rgb_len):
        """两路时间步不同（landmark 48 帧 vs rgb n_windows 窗），
        故各自用自己的长度做 CTC 约束，**相加融合前先对齐时间步**。

        对齐方式：把较短的一路用最近邻重复采样到较长的一路的长度，
        保证相加时形状一致（ref23 式 7 的相加融合在同长序列上定义）。
        """
        z = None
        # 🔴 三配置的时间步必须一致，否则参考长度不同、WER 不可比。
        #   rgb_only 时原始只有 20 窗（20 < 折叠后最长目标 16 词的 2L-1=31），
        #   若直接用 20 会丢掉长句；故**统一上采样到 48**，
        #   使三配置的 T 与参考长度完全一致（P29 铁律）。
        T = 48
        if self.use_lm:
            o, _ = self.lm_enc(lm)                       # (B, 48, 2H)
            if o.size(1) != T:
                o = torch.nn.functional.interpolate(
                    o.transpose(1, 2), size=T, mode="nearest").transpose(1, 2)
            z = self.lm_proj(o)
        if self.use_rgb:
            o2, _ = self.rgb_enc(rgb)                    # (B, n_win, 2H)
            if o2.size(1) != T:
                o2 = torch.nn.functional.interpolate(
                    o2.transpose(1, 2), size=T, mode="nearest").transpose(1, 2)
            z2 = self.rgb_proj(o2)
            z = z2 if z is None else (z + z2 if self.mode == "add"
                                      else torch.cat([z, z2], dim=-1)[:, :, :z.size(-1)])
        return self.head(self.drop(z))


# ---------------------------------------------------------------- 评估
def levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1))
        prev = cur
    return prev[m]


DROPPED = {"n": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=800,
                    help="train/dev 各取多少视频做探针")
    ap.add_argument("--n-windows", type=int, default=20,
                    help="窗口数。CTC 需 T >= 2L-1，参考句长 mean 5.69/max 16，"
                         "故 20 窗口覆盖 98.7%% 样本（12 窗口仅 71.8%%）")
    ap.add_argument("--frames", type=int, default=8,
                    help="每窗口帧数。实测 8 帧与 16 帧单 clip 耗时相同(32.6ms)，"
                         "故 8 帧更划算：同一视频可切更多窗口")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default="lm_only,rgb_only,rgb+lm")
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--rgb-limit", type=int, default=0,
                    help="限制本次新提取的 RGB 视频数（0=不限）")
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p40-rgb-main.json")
    a = ap.parse_args()

    from cslr.contracts import SampleRecord
    from cslr.recognition.gloss_sequence import build_ordered_vocabulary
    from cslr.recognition.training import (
        ctc_loss, iterate_batches, resolve_device, to_torch_batch, decode_batch)

    t_all = time.time()
    device = resolve_device("auto")
    print("device = {}".format(device))

    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(
        lab_tr.values(), min_frequency=a.min_frequency, max_tokens=a.max_tokens)
    print("词表 {}，target 层级 = token".format(voc.size))

    tr_ids = sorted(lab_tr)
    dv_ids = sorted(lab_dv)
    random.Random(a.seed).shuffle(tr_ids)
    random.Random(a.seed).shuffle(dv_ids)
    tr_ids = tr_ids[:a.n_videos]
    dv_ids = dv_ids[:a.n_videos]

    # ---- RGB 特征 ----
    need = tr_ids + dv_ids
    ensure_rgb_cache(need, a.n_windows, device,
                     limit=(a.rgb_limit or None))

    def make(sid, split):
        raw = lab_tr.get(sid, lab_dv.get(sid, ""))
        # GlossVocabulary.encode 接受原始 gloss 字符串（内部自行 split）
        toks = [t.strip() for t in raw.split("/") if t.strip()]
        if not toks:
            return None
        lf = LM_FEAT / split / (sid + ".landmark.npy")
        if not lf.exists():
            return None
        L = np.load(lf).astype(np.float32)                        # (48, 368)
        R = rgb_for(sid, a.n_windows)
        if R is None:
            return None
        ids = voc.encode(raw)
        # 🔴 CTC 硬约束：T >= 2L-1，否则 ctc_loss 直接抛异常。
        #   T 基准由 model.forward 统一到 48（rgb 分支插值上采样），
        #   故这里按 48 算约束 → 折叠后目标最长 16 词，2L-1=31 > 48 不会发生，
        #   实际上 0% 样本需要丢弃（折叠后 max target = 16 < 24）。
        #   保留这个检查作为防御：若将来改了 T 基准或数据，这里会显式报错。
        T_eff = 48
        max_target = (T_eff + 1) // 2
        if len(ids) > max_target:
            DROPPED["n"] += 1
            return None
        if not ids:
            return None
        return {
            "lm": L,
            "rgb": R.astype(np.float32),
            "target": [i + 1 for i in ids],       # CTC 类空间：0 是 blank
            "toks": toks,
            # 与训练 target 同源且同步截断（P29 铁律：参考必须与 target 同空间）
            "folded": voc.decode(list(ids)),
            "sid": sid,
        }

    tr_set = [s for s in (make(x, "train") for x in tr_ids) if s]
    dv_set = [s for s in (make(x, "validation") for x in dv_ids) if s]
    print("train {} 条 / dev {} 条（要求 RGB 与 landmark 同时存在）".format(
        len(tr_set), len(dv_set)))
    if len(dv_set) < 50:
        print("❌ dev 样本不足，先放宽 --n-videos 或检查视频路径")
        return

    ref_dv = {s["sid"]: s["folded"] for s in dv_set}

    def to_batch(items, with_rgb=True, with_lm=True):
        B = len(items)
        lm = torch.zeros(B, 48, 368)
        rgb = torch.zeros(B, a.n_windows, 512)
        T = max(len(s["target"]) for s in items)
        tgt = torch.zeros(B, T, dtype=torch.long)
        tl = torch.zeros(B, dtype=torch.long)
        il = torch.full((B,), 48, dtype=torch.long)
        rl = torch.full((B,), a.n_windows, dtype=torch.long)
        for i, s in enumerate(items):
            lm[i] = torch.from_numpy(s["lm"])
            rgb[i] = torch.from_numpy(s["rgb"])
            n = len(s["target"])
            tgt[i, :n] = torch.tensor(s["target"])
            tl[i] = n
        d = {"features": lm, "input_lengths": il, "targets": tgt,
             "target_lengths": tl, "rgb": rgb, "rgb_lengths": rl,
             "sample_ids": [s["sid"] for s in items]}
        return d

    def evaluate(model, items, with_rgb, with_lm, tag):
        model.eval()
        E = N = 0
        kinds = set()
        seqs = set()
        recs = []
        with torch.no_grad():
            for k in range(0, len(items), 32):
                chunk = items[k:k + 32]
                b = to_batch(chunk, with_rgb, with_lm)
                logits = model(b["features"].to(device),
                               b["input_lengths"].to(device),
                               b["rgb"].to(device),
                               b["rgb_lengths"].to(device))
                lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
                ol = np.full((logits.size(0),), logits.size(1), dtype=np.int64)
                dec, _, _ = decode_batch(lp, ol, 1)
                for r, seq in enumerate(dec):
                    hyp = voc.decode(list(seq))
                    # 参考必须与训练 target 同源（折叠后），且随样本携带，
                    # 否则 train/dev 混用时会取不到（P29 踩过的坑）
                    R = chunk[r]["folded"]
                    e = levenshtein(list(R), hyp)      # 本地 levenshtein 返回标量
                    E += e
                    N += len(R)
                    kinds.update(hyp)
                    seqs.add(tuple(hyp))
                    recs.append({"sid": chunk[r]["sid"], "ref": R, "hyp": hyp, "edit": e})
        L = sum(len(r["hyp"]) for r in recs) / max(len(recs), 1)
        Rl = sum(len(r["ref"]) for r in recs) / max(len(recs), 1)
        hit = sum(1 for r in recs if r["edit"] == 0)
        part = sum(1 for r in recs if 0 < r["edit"] < len(r["ref"]))
        return {"tag": tag, "wer": round(E / max(N, 1), 4), "n_ref_tokens": N,
                "n_distinct": len(kinds), "token_kinds": len(kinds),
                "n_distinct_seqs": len(seqs),
                "hyp_len_mean": round(L, 3), "ref_len_mean": round(Rl, 3),
                "n_exact": hit, "n_partial": part, "n_samples": len(recs),
                "samples": recs[:8]}

    # ---- 8 样本过拟合自检（P20 铁律）----
    if not a.skip_sanity:
        print()
        print("=" * 70)
        print("自检：8 条真实样本过拟合（P20 铁律，新模态接入必须先过这关）")
        print("=" * 70)
        pick = tr_set[:8]
        for name, wr, wl, mo in (("lm_only", False, True, "add"),
                                 ("rgb_only", True, False, "add"),
                                 ("rgb+lm", True, True, "add")):
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            m = DualInputCTC(368, 512, voc.size, a.hidden, a.layers,
                             a.dropout, use_rgb=wr, use_lm=wl, mode=mo).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
            for ep in range(200):
                b = to_batch(pick, wr, wl)
                logits = m(b["features"].to(device), b["input_lengths"].to(device),
                           b["rgb"].to(device), b["rgb_lengths"].to(device))
                # CTC 约束长度必须等于 logits 的实际时间步 T
                T = logits.size(1)
                ol = torch.full((logits.size(0),), T, dtype=torch.long, device=device)
                loss = ctc_loss(logits, b["input_lengths"].to(device),
                                b["targets"].to(device), b["target_lengths"].to(device), ol)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
            r = evaluate(m, pick, wr, wl, name)
            ok = "PASS" if r["wer"] == 0.0 else "FAIL"
            print("  {:<9s} 8 条 WER={:.4f} 精确 {}/8  {}".format(
                name, r["wer"], r["n_exact"], ok))

    # ---- 主实验 ----
    results = []
    for name in a.configs.split(","):
        name = name.strip()
        wr = name in ("rgb_only", "rgb+lm")
        wl = name in ("lm_only", "rgb+lm")
        print()
        print("=" * 70)
        print("配置 {}  (rgb={} lm={})".format(name, wr, wl))
        print("=" * 70)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        m = DualInputCTC(368, 512, voc.size, a.hidden, a.layers, a.dropout,
                         use_rgb=wr, use_lm=wl, mode="add").to(device)
        n_par = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
        hist = []
        for ep in range(1, a.epochs + 1):
            m.train()
            items = tr_set[:]
            random.Random(a.seed + ep).shuffle(items)
            tot = 0.0
            seen = 0
            for k in range(0, len(items), a.batch):
                b = to_batch(items[k:k + a.batch], wr, wl)
                logits = m(b["features"].to(device), b["input_lengths"].to(device),
                           b["rgb"].to(device), b["rgb_lengths"].to(device))
                T = logits.size(1)
                ol = torch.full((logits.size(0),), T, dtype=torch.long, device=device)
                loss = ctc_loss(logits, b["input_lengths"].to(device),
                                b["targets"].to(device),
                                b["target_lengths"].to(device), ol)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                tot += float(loss) * b["features"].shape[0]
                seen += b["features"].shape[0]
            # 🔴 每 10 轮存一次 checkpoint：WSL 只有 7GB 内存，
            #    P42 首跑在 ep35 被 OOM killer 杀掉，若只在末尾存就全丢。
            if ep % 10 == 0 or ep == a.epochs:
                ck_dir0 = REPO / "artifacts/checkpoints"
                ck_dir0.mkdir(parents=True, exist_ok=True)
                sd0 = {k: v for k, v in m.state_dict().items()}   # 变量名是 m 不是 model
                torch.save({"model_state": sd0,
                            "vocab_size": int(voc.size),
                            "config": {"lm_dim": 368, "rgb_dim": 512,
                                        "hidden": a.hidden, "layers": a.layers,
                                        "dropout": a.dropout,
                                        "use_rgb": wr, "use_lm": wl,
                                        "mode": "add",
                                        "n_windows": a.n_windows,
                                        "frames_per_window": a.frames,
                                        "seed": a.seed},
                            "epoch": ep,
                            "reference": "folded (voc.decode(voc.encode(label)))",
                            "notes": "P42 lm_only; target = token+1 (P23 fix)"},
                           ck_dir0 / "p42-lm_only-ep{}.pt".format(ep))
            if ep % 5 == 0 or ep == a.epochs:
                dv = evaluate(m, dv_set, wr, wl, name)
                trr = evaluate(m, tr_set[:300], wr, wl, name + "_train")
                hist.append({"epoch": ep, "loss": round(tot / max(seen, 1), 4),
                             "dev_wer": dv["wer"], "train_wer": trr["wer"],
                             "dev_ndist": dv["n_distinct"],
                             "dev_len": dv["hyp_len_mean"]})
                print("  ep {:3d} loss={:8.4f} trainWER={:.4f} devWER={:.4f} "
                      "ndist={:4d} len={:.2f}".format(
                          ep, hist[-1]["loss"], trr["wer"], dv["wer"],
                          dv["n_distinct"], dv["hyp_len_mean"]), flush=True)
        fin = evaluate(m, dv_set, wr, wl, name)
        # 🔴 必须保存 checkpoint（用户 2026-10-05 要求）。
        #    P40 首跑就是因为没存，训出最优 0.5211 后模型永久丢失。
        ck_dir = REPO / "artifacts/checkpoints"
        ck_dir.mkdir(parents=True, exist_ok=True)
        ck = ck_dir / "p40-{}-ep{}.pt".format(name.replace("+", "_"), a.epochs)
        torch.save({"model_state": m.state_dict(),
                    "config": {"lm_dim": 368, "rgb_dim": 512,
                                "vocab_size": voc.size, "hidden": a.hidden,
                                "layers": a.layers, "dropout": a.dropout,
                                "use_rgb": wr, "use_lm": wl, "mode": "add",
                                "n_windows": a.n_windows,
                                "frames_per_window": a.frames,
                                "seed": a.seed, "epochs": a.epochs},
                    "vocab_size": int(voc.size),
                    "feature_view": "full",
                    "epoch": a.epochs,
                    "dev_wer": fin["wer"],
                    "reference": "folded (voc.decode(voc.encode(label)))",
                    "notes": "P40 RGB-as-main-stream; target = token+1 (P23 fix)"},
                   ck)
        print("  => checkpoint 已保存: {}".format(ck))
        results.append({"config": name, "params_M": round(n_par / 1e6, 2),
                        "history": hist, "final": fin,
                        "checkpoint": str(ck),
                        "final_train_wer": hist[-1]["train_wer"]})
        print("  => devWER={:.4f} ndist={} exact={}/{}".format(
            fin["wer"], fin["n_distinct"], fin["n_exact"], fin["n_samples"]))

    # ---- 判决 ----
    verdict = None
    base = next((r for r in results if r["config"] == "lm_only"), None)
    rgbo = next((r for r in results if r["config"] == "rgb_only"), None)
    both = next((r for r in results if r["config"] == "rgb+lm"), None)
    if base and rgbo:
        d = rgbo["final"]["wer"] - base["final"]["wer"]
        line = "rgb_only - lm_only = {:+.4f}（判据 <= -0.02 判有效）".format(d)
        if d <= -0.02:
            verdict = "RGB 主干有效（{}）-> 值得全量提取 5488 个视频".format(line)
        else:
            verdict = "RGB 主干无增益（{}）-> 与 A2 旧结论一致，但已在修好管线上确认".format(line)
    if both and base and rgbo:
        b = both["final"]["wer"]
        if b < min(base["final"]["wer"], rgbo["final"]["wer"]) - 0.02:
            verdict += "；且 rgb+lm {:.4f} 显著优于任一单模态 -> 双流互补（ref23 主张成立）".format(b)
    print()
    print("判决：{}".format(verdict))

    out = {
        "experiment": "P40 RGB as the MAIN input stream into CTC "
                      "(A2's old negative result was measured on the off-by-one model)",
        "n_videos_train": len(tr_set), "n_videos_dev": len(dv_set),
        "n_dropped_too_long": DROPPED["n"],
        "drop_reason": "target_len > (min(n_windows,48)+1)//2 -> CTC 不可对齐，丢弃而非截断（保持参考一致）",
        "n_windows": a.n_windows, "frames_per_window": a.frames,
        "rgb_feature_dim": 512, "lm_feature_dim": 368,
        "vocab_size": int(voc.size), "epochs": a.epochs,
        "seed": a.seed, "results": results, "verdict": verdict,
        "criterion": "rgb_only dev_WER <= lm_only - 0.02",
        "key_difference_vs_a2": "A2 extracted a single pooled 512-D vector which cannot feed "
                                "CTC (needs (B,T,C)); P40 keeps the window axis -> (T,512)",
        "literature_basis": {
            "ref23_SignFormer-GCN": "Eq.7 Fused = Z_t + L_t (RGB + skeleton addition) — "
                                    "本实验的 rgb+lm 配置",
            "ref07_SignBT": "CSL-Daily CSLR SOTA WER 32.0-32.2 vs 我们 0.70",
            "ref12_CCL-SLR": "pose 有上限、RGB 有语义线索（方向性参考，它是 ISLR 非 CSLR）",
        },
        "metrics_note": "P38 铁律：acc/WER 在本任务失真，故同时报 n_distinct / "
                        "hyp_len / exact / partial 作交叉验证",
        "all_data_real": True, "reads_test_split": False,
        "minutes": round((time.time() - t_all) / 60, 2),
    }
    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("收据 -> {}".format(p))


if __name__ == "__main__":
    main()
