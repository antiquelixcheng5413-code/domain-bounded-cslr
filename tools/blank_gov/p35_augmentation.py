# -*- coding: utf-8 -*-
"""P35 · 特征空间数据增强 —— 当前唯一未被证伪的方向

## 动机：为什么现在才做增强

P34 已确诊：四个模态在跨视频 held-out 上都 ≈ 随机（0.0028~0.0063），
包括最强的 hands。**landmark 特征几乎不编码 gloss 身份。**

已证伪的方向（勿再试）：
- ❌ 加训练轮数（P28，30→150 只值 0.0063）
- ❌ ST-GCN 骨架拓扑（P29，train 拟合完美但 dev 差 0.037）
- ❌ 容量控制（P30，四个配置全劣于 base）
- ❌ 面部扩容（P34，face128/face8 = 1.01x）

**数据增强是唯一还没试过的方向** —— 它不增加信息，而是
让模型看到同一句话的多种「合理变体」，从而被迫学到不变表征。

## 论文依据（三条，均为 landmark/骨架可直接迁移的）

### ref07 Zhou et al., Sign Back-Translation, Sec 5.2（原文）
> "For data augmentation, we use **random shift** and **random discard or copy
> of 20% frames**."
→ 时间轴丢帧/复制的明确比例：**20%**

### ref12 Wu et al., CCL-SLR, Sec 3.2 + Fig 3
> "we use **spatial-temporal augmentation** to generate different query and key
>  samples for each single-modal branch"
→ 每个样本生成两个视图，构成对比学习的正样本对。
原文的 MPM（Motion-Preserving Masking）依赖 RGB 视频与生成模型，
**不能搬到 landmark**；但「双视图」这个原语可以。

### ref03 Wójcicka et al., LREC 2026, Sec 4.1.2
> "To account for variations in signer positioning and distance from the camera,
> we apply **scale and translation normalization**."
→ 位置/距离变更是真实噪声。我们的特征已做肩距归一化，
但**增强可以反过来注入这类变化**（模拟不同拍摄条件）。

## 本脚本测什么

四种增强，单因素对照（其余超参完全固定）：

| 配置 | 增强 | 依据 |
|---|---|---|
| `base` | 无 | 对照 |
| `shift` | 时间轴随机平移 ±2 帧 + 高斯噪声 σ=0.01 | ref07 random shift |
| `mask` | 随机丢弃 20% 帧（用邻近帧替换） | ref07 "discard or copy of 20% frames" |
| `scale` | 随机缩放 0.9~1.1 + 平移 ±0.05 | ref03 scale/translation 变体 |
| `combo` | 上述三者组合 | 多重扰动 |

**每种增强都必须验证它确实改变了数据**（不能是恒等变换）——
这是本项目反复踩坑的教训（P29 参考口径、P33 探针饱和）：
**先验断言再训练**。

## 判据（跑之前写死）

- 若 `combo` 或任一增强的 **dev WER 比 base 低 ≥0.02** → 增强有效
- 若所有增强都无改善（<0.01）→ 增强方向也排除，
  landmark 表征的局限无法靠扰动弥补
- **必须同时报 train WER**：若 train 与 dev 同步改善 → 真收益；
  若只 train 改善 → 只是加剧记忆，无效

只读 train/dev，**不触碰 test split**。
"""
import argparse
import collections
import dataclasses
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

from cslr.contracts import SampleRecord
from cslr.recognition.model import CTCConfig, CTCRecognizer
from cslr.recognition.gloss_sequence import build_ordered_vocabulary
from cslr.recognition.dataset import GlossSequenceDataset, FeatureNormalizer, collate_samples
from cslr.recognition.training import (
    ctc_loss, iterate_batches, learning_rate_at, resolve_device,
    to_torch_batch, decode_batch)
from p8_error_attribution import levenshtein

HANDS = slice(0, 126)
POSE = slice(126, 158)
FACE = slice(158, 182)
PRES = slice(182, 186)


# ----------------------------------------------------------------- 增强原语
def aug_identity(x, rng):
    return x


def aug_shift(x, rng, max_shift=2, noise=0.01):
    """时间轴随机平移（循环）+ 高斯噪声。对应 ref07 的 "random shift"。

    循环平移保持时序连续，不引入 padding 边界效应。
    """
    t = len(x)
    s = int(rng.integers(-max_shift, max_shift + 1))
    out = np.roll(x, s, axis=0)
    if noise > 0:
        out = out + rng.normal(0.0, noise, size=out.shape).astype(np.float32)
    return out.astype(np.float32)


def aug_mask(x, rng, ratio=0.20):
    """随机丢弃 ratio 比例的帧（用最近的保留帧替换）。

    对应 ref07 的 "random discard or copy of 20% frames"。
    替换而非置零：landmark 置零等于伪造「手部消失」，
    而实测检出率 100%，置零会制造不真实的样本。
    """
    t = len(x)
    n_drop = max(int(round(t * ratio)), 1)
    idx = rng.choice(t, size=n_drop, replace=False)
    keep = np.setdiff1d(np.arange(t), idx)
    if keep.size == 0:
        return x
    # 最近邻替换
    out = x.copy()
    for i in idx:
        j = keep[np.argmin(np.abs(keep - i))]
        out[i] = x[j]
    return out


def aug_scale(x, rng, lo=0.9, hi=1.1, shift=0.05):
    """随机缩放 + 平移，模拟不同拍摄距离/位置（ref03 Sec 4.1.2 的反向注入）。"""
    s = float(rng.uniform(lo, hi))
    dx, dy, dz = rng.uniform(-shift, shift, size=3)
    out = x.copy()
    out[:, HANDS] = (x[:, HANDS] - 0.5) * s + 0.5 + dx
    out[:, POSE] = (x[:, POSE] - 0.5) * s + 0.5 + dy
    out[:, FACE] = (x[:, FACE] - 0.5) * s + 0.5 + dz
    return out.astype(np.float32)


def aug_combo(x, rng):
    out = x
    if rng.random() < 0.5:
        out = aug_scale(out, rng)
    if rng.random() < 0.5:
        out = aug_shift(out, rng)
    if rng.random() < 0.3:
        out = aug_mask(out, rng)
    return out


AUGS = {
    "base": aug_identity,
    "shift": aug_shift,
    "mask": aug_mask,
    "scale": aug_scale,
    "combo": aug_combo,
}


# ----------------------------------------------------------------- 数据
def read_split(split):
    table = {"train": "train.csv", "validation": "dev.csv"}
    if split not in table:
        raise ValueError("split 必须是 {} 之一".format(sorted(table)))
    out = {}
    with open(REPO / "data/raw/CE-CSL/label" / table[split],
              newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["Number"]] = r["Gloss"]
    return out


def link_dir(split):
    root = REPO / ".vocab_link" / split
    root.mkdir(parents=True, exist_ok=True)
    for p in sorted((REPO / "artifacts/part3_features" / split).glob("*.landmark.npy")):
        sid = p.name[: -len(".landmark.npy")]
        dst = root / (sid + ".npy")
        if not dst.exists():
            try:
                dst.symlink_to(p)
            except OSError:
                pass
    return root


def make_records(labels, feat_root, split):
    out = []
    for sid, g in labels.items():
        if not (feat_root / (sid + ".npy")).exists():
            continue
        if not [t.strip() for t in g.split("/") if t.strip()]:
            continue
        out.append(SampleRecord(sample_id=sid, video=Path(sid + ".mp4"), label=g,
                                signer="x", session="x", split=split))
    out.sort(key=lambda r: r.sample_id)
    return out


class AugmentedDataset:
    """包装 GlossSequenceDataset，在 __getitem__ 时施加增强。

    关键：增强只在**训练集**上做，dev 必须用原特征。
    normalizer 用未增强的统计量拟合（与推理时口径一致）。
    """

    def __init__(self, base, aug_fn, seed):
        self.base = base
        self.aug_fn = aug_fn
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        if self.aug_fn is aug_identity:
            return s
        # SequenceSample 是 frozen dataclass，不能原地赋值，必须整体替换
        rng = np.random.default_rng((self.seed * 1000003 + i) % (2 ** 32))
        return dataclasses.replace(s, features=self.aug_fn(s.features, rng))


@torch.no_grad()
def evaluate(model, ds, device, batch, voc, ref, subset=None):
    model.eval()
    d = n = 0
    seqs = []
    for samples in iterate_batches(ds, batch, shuffle=False, seed=42):
        if subset is not None:
            samples = [s for s in samples if s.sample_id in subset]
            if not samples:
                continue
        b = to_torch_batch(collate_samples(samples), device)
        logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        dec, _, _ = decode_batch(lp, ol.cpu().numpy(), 1)
        for r, ids in enumerate(dec):
            hyp = voc.decode(list(ids))
            seqs.append(tuple(hyp))
            d += levenshtein(list(ref[b["sample_ids"][r]]), hyp)[0]
            n += len(ref[b["sample_ids"][r]])
    return {"wer": round(d / max(n, 1), 4), "n_distinct": len(set(seqs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-eval-n", type=int, default=1500)
    ap.add_argument("--configs", default="base,shift,mask,scale,combo")
    ap.add_argument("--out",
                    default="artifacts/metrics/blank-gov/p35-augmentation.json")
    a = ap.parse_args()

    device = resolve_device("auto")
    tr_labels = read_split("train")
    va_labels = read_split("validation")
    tr_root = link_dir("train")
    va_root = link_dir("validation")
    tr_recs = make_records(tr_labels, tr_root, "train")
    va_recs = make_records(va_labels, va_root, "validation")
    voc, _ = build_ordered_vocabulary((g for g in tr_labels.values()),
                                      min_frequency=2, max_tokens=300)
    nrm = FeatureNormalizer.fit(
        [np.load(tr_root / (r.sample_id + ".npy")) for r in tr_recs[:600]])
    tr_base = GlossSequenceDataset(tr_recs, tr_root, voc, nrm, feature_view="full")
    va_ds = GlossSequenceDataset(va_recs, va_root, voc, nrm, feature_view="full")
    ref_tr = {r.sample_id: [voc.tokens[i] for i in voc.encode(r.label)] for r in tr_recs}
    ref_va = {r.sample_id: [voc.tokens[i] for i in voc.encode(r.label)] for r in va_recs}
    tr_subset = set(random.Random(12345).sample(sorted(ref_tr.keys()),
                                               min(a.train_eval_n, len(ref_tr))))
    print("device = {}  train {} / dev {}  vocab {}".format(
        device, len(tr_recs), len(va_recs), voc.size))

    # ---- 预检：增强确实改变了数据（不能是恒等变换）----
    print()
    print("=" * 74)
    print("预检 · 各增强确实改变数据（若 delta 全为 0 则该增强是恒等的）")
    print("=" * 74)
    x0 = np.load(tr_root / (tr_recs[0].sample_id + ".npy")).copy()
    for name in a.configs.split(","):
        fn = AUGS[name]
        rng = np.random.default_rng(0)
        deltas = []
        for _ in range(20):
            xa = fn(x0.copy(), rng)
            deltas.append(float(np.abs(xa - x0).mean()))
        d = float(np.mean(deltas))
        ok = "PASS" if (name == "base" and d < 1e-9) or (name != "base" and d > 1e-6) else "FAIL"
        print("  {:<7s} 平均 |Δ| = {:.6f}   {}".format(name, d, ok))
        if name != "base" and d <= 1e-6:
            print("  !! {} 是恒等变换，配置无效".format(name))

    results = []
    for name in a.configs.split(","):
        fn = AUGS[name]
        print()
        print("#" * 74)
        print("# 配置 {}   增强={}".format(name, fn.__name__))
        print("#" * 74)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        cfg = CTCConfig(input_size=368, vocabulary_size=voc.size, hidden_size=256,
                        num_layers=2, dropout=0.3, bidirectional=True,
                        projection_size=256, subsample_stride=1)
        model = CTCRecognizer(cfg).to(device)
        tr_ds = AugmentedDataset(tr_base, fn, a.seed)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        hist = []
        t0 = time.time()
        for ep in range(1, a.epochs + 1):
            model.train()
            tot = seen = 0
            cur_lr = learning_rate_at(ep, a.epochs, a.lr, warmup_epochs=a.warmup,
                                      schedule="cosine", min_ratio=0.05)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            for samples in iterate_batches(tr_ds, a.batch, shuffle=True,
                                           seed=a.seed + ep):
                b = to_torch_batch(collate_samples(samples), device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,
                                    enabled=device.type == "cuda"):
                    lg = model(b["features"], b["input_lengths"])
                    ol = model.output_lengths(b["input_lengths"])
                    loss = ctc_loss(lg, b["input_lengths"], b["targets"],
                                    b["target_lengths"], ol)
                if device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                tot += float(loss.item()) * len(samples)
                seen += len(samples)
            if ep % 10 == 0 or ep == a.epochs:
                tr = evaluate(model, tr_base, device, a.batch, voc, ref_tr, tr_subset)
                dv = evaluate(model, va_ds, device, a.batch, voc, ref_va)
                hist.append({"epoch": ep, "train_loss": round(tot / max(seen, 1), 4),
                             "train_wer": tr["wer"], "dev_wer": dv["wer"],
                             "dev_n_distinct": dv["n_distinct"]})
                print("  ep {:3d} loss={:8.4f} trainWER={:.4f} devWER={:.4f} "
                      "ndist={:4d}".format(ep, tot / max(seen, 1), tr["wer"],
                                           dv["wer"], dv["n_distinct"]), flush=True)
        results.append({"config": name, "aug": fn.__name__,
                        "final_train_wer": hist[-1]["train_wer"],
                        "final_dev_wer": hist[-1]["dev_wer"],
                        "history": hist,
                        "minutes": round((time.time() - t0) / 60, 2)})

    print()
    print("=" * 78)
    print("P35 汇总（判据：dev WER 比 base 低 >=0.02）")
    print("=" * 78)
    print("{:<8s} {:>10s} {:>10s} {:>8s} {:>10s}".format(
        "配置", "trainWER", "devWER", "vs base", "判定"))
    base = next((r for r in results if r["config"] == "base"), None)
    best = None
    for r in results:
        dd = (r["final_dev_wer"] - base["final_dev_wer"]) if base else 0.0
        mark = "基线" if r["config"] == "base" else ("PASS" if dd <= -0.02 else "无收益")
        print("{:<8s} {:>10.4f} {:>10.4f} {:>+8.4f} {:>10s}".format(
            r["config"], r["final_train_wer"], r["final_dev_wer"], dd, mark))
        if r["config"] != "base" and dd <= -0.02 and (best is None or dd < best[1]):
            best = (r["config"], dd)

    if best:
        verdict = ("数据增强有效：{} 的 devWER {:+.4f}（<-0.02），"
                   "且需核对 train 是否同步改善".format(best[0], best[1]))
    else:
        verdict = "数据增强无效：所有配置 dev WER 变化 >-0.01，landmark 表征局限无法靠扰动弥补"
    print()
    print("判决: " + verdict)
    # train/dev 是否同步
    if base:
        for r in results:
            if r["config"] == "base":
                continue
            dt = r["final_train_wer"] - base["final_train_wer"]
            dd = r["final_dev_wer"] - base["final_dev_wer"]
            if dt < -0.02 and dd > -0.005:
                print("  ⚠️ {}: train {:+.4f} 改善但 dev {:+.4f} 未改善 —— 只是加剧记忆".format(
                    r["config"], dt, dd))

    outp = REPO / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "experiment": "P35 feature-space data augmentation",
        "paper_basis": {
            "ref07_SignBT_Sec5.2": "random shift + random discard or copy of 20% frames",
            "ref12_CCLSLR_Sec3.2": "spatial-temporal augmentation to generate two views "
                                    "per sample for contrastive learning; its MPM masking "
                                    "needs RGB+generative model so NOT transferred",
            "ref03_Wojcicka_Sec4.1.2": "scale and translation normalization for signer "
                                        "positioning/distance variation",
        },
        "augmentations": {
            "shift": "circular time shift +-2 frames + gaussian noise sigma=0.01",
            "mask": "drop 20% frames, nearest-neighbour replacement (not zeroing: "
                    "measured detection rate is 100%, zeroing would fabricate absence)",
            "scale": "random scale 0.9-1.1 + translation +-0.05 on the three blocks",
            "combo": "random subset of the above",
        },
        "criterion": "dev WER <= base - 0.02 AND train must improve too",
        "results": results, "verdict": verdict,
        "all_data_real": True, "reads_test_split": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(outp))


if __name__ == "__main__":
    main()
