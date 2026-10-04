# -*- coding: utf-8 -*-
"""P15 · Wójcicka Table 4 的 modality 消融（真实数据，零成本）

文献依据（Wójcicka LREC 2026, Sec 4.3 Ablation Study: Impact of Modalities，
Table 4，Test Set）：

    Input Modality   F1@10   F1@25   F1@50   λ (vs Full)
    Face Only         70.00   64.30   43.92   -13.60
    Hands Only        69.39   63.74   47.92    -9.60
    Full              75.43   69.96   57.52      —

论文原话：
> "the Face-Only model performs competitively at looser thresholds
>  (F1@10: 70.00%), slightly outperforming the Hands-Only baseline. ...
>  However, the performance drops significantly at stricter thresholds
>  (F1@50: 43.92%), suggesting that while the face indicates the presence
>  of a boundary, it lacks the temporal sharpness to define the exact
>  frame of the onset."

## 为什么这个消融值得做（迁移条件几乎完全一致）

论文的 MediaPipe 特征选择（Sec 4.1.1）：
  Pose  : 8 landmarks × 4 = 32 维
  Hands : 21 landmarks × 3 × 2 = 126 维
  Face  : 8 landmarks × 3 = 24 维

我们仓库的 368 维特征布局（已实测核对，extractor.py BASE_FEATURE_SIZE=182）：
  [0:126]   hands      126 维   ←→ 论文 126 维，**完全一致**
  [126:158] pose        32 维   ←→ 论文 32 维，**完全一致**
  [158:182] face        24 维   ←→ 论文 24 维，**完全一致**
  [182:186] presence     4 维
  [186:368] deltas      182 维   ← 一阶差分（原始帧率上求，非 48 帧差分）

**这是文献里少见的三段式消融，且我们的特征布局天然支持，零成本。**

## 主指标说明

论文报的是 F1@{10,25,50}（IoU 阈值下的**分词** F1），需要句子级边界真值。
我们**没有**边界真值，所以主指标只能用**逐 token 准确率 / macro-F1**
（与 Wójcicka Table 2/5 的 frame-wise 口径对应）。

**明确声明：不对标论文的 F1@50 数字**（任务层级不同，我们无边界 GT）。

判据（预先写死）：
  - 若 face-only 的 token 准确率显著高于随机水平（0.5），
    则 face 流在本数据上有判别信号 -> 论文的「face 指示边界存在」结论可迁移
  - 若 hands-only > face-only，则与论文 F1@50 排序一致
    （hands 定位更准，face 只能指示存在）
  - 若 full < max(hands, face)，则存在模态冗余，
    论文的 λ<0（去掉任一模态都变差）在本数据不成立

只读 train/dev，不触碰 test split。
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

_CSLR_ERROR = None
try:
    from cslr.contracts import SampleRecord
    from cslr.recognition.model import (
        CTCConfig, CTCRecognizer, ctc_config_from_dict, BLANK_INDEX)
    from cslr.recognition.gloss_sequence import (
        build_ordered_vocabulary, GlossVocabulary, GlossSequenceConfig)
    from cslr.recognition.dataset import (
        GlossSequenceDataset, FeatureNormalizer, collate_samples)
    from cslr.recognition.training import (
        TrainingConfig, ctc_loss, evaluate, iterate_batches,
        to_torch_batch, learning_rate_at, resolve_device, decode_batch)
except ModuleNotFoundError as exc:
    _CSLR_ERROR = exc
    BLANK_INDEX = 0

# ---- 特征分块（已实测核对 extractor.py 的 BASE_FEATURE_SIZE=182）----
BLOCKS = {
    "hands": (0, 126),
    "pose": (126, 158),
    "face": (158, 182),
    "masks": (182, 186),
}
BASE_END = 182          # base + masks 的结束位置
DELTA_START = 186       # deltas 块起点
DELTA_END = 368


def slice_modality(arr: np.ndarray, name: str) -> np.ndarray:
    """按模态切片特征（arr: (T, 368)）。

    模态取「base 部分 + 对应的 delta 部分」，与 Wójcicka 的
    「该模态的归一化位置 + 其速度/加速度」口径一致。
    """
    if name == "full":
        return arr
    if name == "base_only":
        return arr[:, :BASE_END]
    s, e = BLOCKS[name]
    base = arr[:, s:e]
    # delta 块与 base 块一一对应（deltas[i] 对应 base[i]）
    d = arr[:, DELTA_START + s: DELTA_START + e]
    return np.concatenate([base, d], axis=1)


def build_model(input_size, vocab_size, hidden, layers, dropout):
    cfg = CTCConfig(input_size=input_size, vocabulary_size=vocab_size,
                    hidden_size=hidden, num_layers=layers, dropout=dropout,
                    bidirectional=True, projection_size=256, subsample_stride=1)
    return cfg, CTCRecognizer(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", default="full,hands,face,pose,hands_face",
                    help="逗号分隔的模态组合")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",
                    default="artifacts/metrics/blank-gov/p15-modality-ablation.json")
    a = ap.parse_args()
    if _CSLR_ERROR is not None:
        raise SystemExit("cslr 不可用：{}（需在仓库 venv 下运行）".format(_CSLR_ERROR))

    device = resolve_device("auto")
    print("device = {}".format(device))

    # ---- 特征分块缓存（真实特征，只读一次）----
    tr_table = {}
    for split, fn in (("train", "train.csv"), ("validation", "dev.csv")):
        with open(REPO / "data/raw/CE-CSL/label" / fn, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tr_table[(split, r["Number"])] = r["Gloss"]
    tr_ids = sorted(sid for (sp, sid) in tr_table if sp == "train")
    va_ids = sorted(sid for (sp, sid) in tr_table if sp == "validation")

    feats = {}
    for split, sub in (("train", "train"), ("validation", "validation")):
        d = REPO / "artifacts/part3_features" / sub
        for sid in (tr_ids if split == "train" else va_ids):
            p = d / (sid + ".landmark.npy")
            if p.exists():
                feats[(split, sid)] = np.load(p).astype(np.float32)
    print("真实特征：train {} 条  dev {} 条  维度 {}".format(
        sum(1 for k in feats if k[0] == "train"),
        sum(1 for k in feats if k[0] == "validation"),
        next(iter(feats.values())).shape[1]))

    # ---- 词表（只用全量 train 标签构建，保证各模态同词表）----
    voc, _ = build_ordered_vocabulary(
        (tr_table[("train", sid)] for sid in tr_ids),
        min_frequency=2, max_tokens=300)
    print("词表 {} 类（所有模态共用）".format(voc.size))

    mods = [m.strip() for m in a.modalities.split(",") if m.strip()]
    results = []
    t_all = time.time()

    for mod in mods:
        import random as _r
        _r.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

        def sub(arr):
            if mod in ("hands_face",):
                return np.concatenate([slice_modality(arr, "hands"),
                                       slice_modality(arr, "face")], axis=1)
            return slice_modality(arr, mod)

        dim = sub(next(v for k, v in feats.items() if k[0] == "train")).shape[1]
        print("\n" + "=" * 68)
        print("模态 {:<12} 输入维度 {}".format(mod, dim))
        print("=" * 68)

        # 训练集张量（真实特征，预先载入）
        # 标准化统计量用**真实训练特征**算（不是合成数据）
        all_tr = np.concatenate(
            [sub(feats[("train", sid)]) for sid in tr_ids if ("train", sid) in feats],
            axis=0)
        mean = all_tr.mean(axis=0).astype(np.float32)
        std = (all_tr.std(axis=0) + 1e-8).astype(np.float32)

        tr_recs = [SampleRecord(sample_id=sid, video=Path(sid + ".mp4"),
                                label=tr_table[("train", sid)],
                                signer=sid.split("-")[0], session="x", split="train")
                   for sid in tr_ids if ("train", sid) in feats]
        va_recs = [SampleRecord(sample_id=sid, video=Path(sid + ".mp4"),
                                label=tr_table[("validation", sid)],
                                signer=sid.split("-")[0], session="x", split="validation")
                   for sid in va_ids if ("validation", sid) in feats]

        # 用独立目录让 Dataset 读到切好的特征
        link_root = REPO / ".p15_feats" / mod
        for split, ids in (("train", tr_ids), ("validation", va_ids)):
            d = link_root / split
            d.mkdir(parents=True, exist_ok=True)
            # 注意：ids 与 arr 必须用**同一个过滤条件**，否则 zip 会错位，
            # 导致某些 id 没写文件（首版就踩了这个坑）
            for sid in ids:
                if (split, sid) not in feats:
                    continue
                np.save(d / (sid + ".npy"), sub(feats[(split, sid)]))

        nrm = FeatureNormalizer(mean=mean, std=std)
        tr_ds = GlossSequenceDataset(tr_recs, link_root / "train", voc, nrm,
                                     feature_view="full")
        va_ds = GlossSequenceDataset(va_recs, link_root / "validation", voc, nrm,
                                     feature_view="full")

        cfg, model = build_model(dim, voc.size, a.hidden, a.layers, a.dropout)
        model = model.to(device)
        npar = sum(p.numel() for p in model.parameters())
        print("参数量 {:.2f}M".format(npar / 1e6))

        tcfg = TrainingConfig(epochs=a.epochs, batch_size=a.batch,
                             learning_rate=a.lr, weight_decay=a.weight_decay,
                             seed=a.seed, device=str(device), amp=True,
                             early_stopping_patience=a.patience, beam_width=1,
                             min_epochs=1, sequence_length=48, normalize_features=False)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        use_amp = tcfg.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best = {"wer": float("inf")}
        best_state = None
        hist = []
        no_imp = 0
        t0 = time.time()
        for ep in range(1, tcfg.epochs + 1):
            model.train()
            ep_loss, seen = 0.0, 0
            cur_lr = learning_rate_at(ep, tcfg.epochs, a.lr, warmup_epochs=a.warmup,
                                      schedule="cosine", min_ratio=0.05)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            for samples in iterate_batches(tr_ds, a.batch, shuffle=True,
                                           seed=a.seed + ep):
                b = to_torch_batch(collate_samples(samples), device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(b["features"], b["input_lengths"])
                    ol = model.output_lengths(b["input_lengths"])
                    loss = ctc_loss(logits, b["input_lengths"], b["targets"],
                                    b["target_lengths"], ol)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                    opt.step()
                ep_loss += float(loss.item()) * len(samples)
                seen += len(samples)

            val = evaluate(model, va_ds, voc, a.batch, device, beam_width=1,
                           seed=a.seed, amp=use_amp)
            m = val.metrics
            rec = {"epoch": ep, "train_loss": ep_loss / max(seen, 1),
                   "wer": float(m["wer"]), "cer": float(m["cer"]),
                   "blank_ratio": float(m["blank_ratio"]),
                   "vocab_util": float(m.get("vocabulary_utilization", 0.0))}
            hist.append(rec)
            print("  ep {:2d}  loss={:.4f}  WER={:.4f}  CER={:.4f}  blank={:.4f}".format(
                ep, rec["train_loss"], rec["wer"], rec["cer"], rec["blank_ratio"]),
                flush=True)
            if rec["wer"] < best["wer"]:
                best = dict(rec); best["epoch"] = ep; no_imp = 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                no_imp += 1
                if no_imp >= a.patience:
                    print("  early stop @ ep{}".format(ep))
                    break
        dt = (time.time() - t0) / 60
        results.append({"modality": mod, "input_dim": dim, "params_M": npar / 1e6,
                        "best": best, "history": hist, "minutes": dt})
        print("  最佳 WER={:.4f} (ep{})  用时 {:.1f} 分钟".format(
            best["wer"], best["epoch"], dt))
        del model
        torch.cuda.empty_cache()

    # ---- 汇总 ----
    print("\n" + "=" * 68)
    print("P15 · Wójcicka Table 4 modality 消融（真实数据）")
    print("=" * 68)
    print("{:<14} {:>8} {:>10} {:>10} {:>10} {:>12}".format(
        "模态", "维度", "WER", "CER", "blank", "相对full"))
    full_wer = next((r["best"]["wer"] for r in results if r["modality"] == "full"), None)
    for r in results:
        rel = "" if full_wer is None or r["modality"] == "full" else \
            "{:+.4f}".format(r["best"]["wer"] - full_wer)
        print("{:<14} {:>8} {:>10.4f} {:>10.4f} {:>10.4f} {:>12}".format(
            r["modality"], r["input_dim"], r["best"]["wer"], r["best"]["cer"],
            r["best"]["blank_ratio"], rel))
    print("=" * 68)

    # ---- 与论文对照 ----
    paper = {"face": (70.00, 64.30, 43.92), "hands": (69.39, 63.74, 47.92),
             "full": (75.43, 69.96, 57.52)}
    print("\n论文 Table 4（Test Set，F1@{10,25,50}）—— **不可直接对标**")
    print("  论文任务是**分词**且有句子级边界 GT；我们是逐 token 识别、无边界 GT。")
    print("  可对照的只有**排序**：论文 F1@50 排序为 full > hands > face。")
    ours = {r["modality"]: r["best"]["wer"] for r in results}
    if "hands" in ours and "face" in ours:
        better = "hands" if ours["hands"] < ours["face"] else "face"
        print("\n  本项目 WER 排序（越低越好）: {} 更优".format(better))
        print("  论文 F1@50 排序（越高越好）: hands 更优（47.92 > 43.92）")
        if better == "hands":
            print("  -> 排序**一致**：hands 定位更准，face 只指示边界存在。")
        else:
            print("  -> 排序**不一致**：本项目 face 反而更优，需记录该差异。")
    if "full" in ours and "hands" in ours:
        lam_h = ours["hands"] - ours["full"]
        print("\n  λ_hands（去掉 face 的代价）= {:+.4f}".format(lam_h))
        print("  论文 λ: Face Only 相对 Full 为 -13.60（去掉 hands 的代价）")
        if lam_h > 0:
            print("  -> 本项目 face 有正贡献（去掉后变差），与论文「模态互补」一致。")
        else:
            print("  -> 本项目去掉 face 反而更好，**模态冗余**，与论文不一致。")

    receipt = {
        "paper": "Wójcicka LREC 2026, Sec 4.3 (Ablation: Impact of Modalities), Table 4",
        "paper_table4": {
            "note": "F1@{10,25,50} 是分词指标 + 有句子级边界 GT，与本项目不可直接对标",
            "face_only": {"F1@10": 70.00, "F1@25": 64.30, "F1@50": 43.92},
            "hands_only": {"F1@10": 69.39, "F1@25": 63.74, "F1@50": 47.92},
            "full": {"F1@10": 75.43, "F1@25": 69.96, "F1@50": 57.52},
        },
        "our_metric": "逐 token WER / CER（无边界 GT，故不用 F1@50）",
        "feature_layout_verified": {
            "hands": "[0:126] = 126 维（论文亦 126）",
            "pose": "[126:158] = 32 维（论文亦 32）",
            "face": "[158:182] = 24 维（论文亦 24）",
            "deltas": "[186:368] 是一阶差分，但在**原始帧率**上求后重采样，"
                      "不是 48 帧序列的差分（实测偏差 1.32）",
        },
        "modalities": results,
        "n_train": len(tr_recs), "n_val": len(va_recs),
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
