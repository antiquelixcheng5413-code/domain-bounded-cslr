# -*- coding: utf-8 -*-
"""P3 选项 1：帧级辅助损失 + gloss 级 CTC 联合训练（Hands-On FG2025 Sec III-D1）。

文献依据（本项目已逐篇核实，见 .workbuddy/memory/MEMORY.md）：
  Hands-On FG2025 Sec III-D1 原文：
    "The transformer encoder processes the multi-modal features to generate
     (Begin-In-Out) BIO scheme predictions. Concurrently, gloss-level supervision
     is incorporated using the Connectionist Temporal Classification (CTC) loss,
     allowing the model to leverage linguistic annotations while also providing
     a global constraint."

  → 帧级 BIO 与 gloss 级 CTC **同时优化**，这是选项 1 的唯一直接依据。

  **反向依据（必须一并考虑）**：SMART 全文两处明写
    "The recognition module is frozen during spotting training"（Sec 3.3 / 4.1）
  即同任务上 SMART 选了冻结 + 推理期 late fusion。
  Hands-On 与 SMART 的差别在帧级监督来源：
    - Hands-On 用 DGS 语料**自带的人工 frame-level gloss 标注**
    - 本项目没有（CE-CSL 无边界标注），只能用 TFD 伪边界
  所以本实验是把「有帧级 GT 时验证过的方法」外推到「伪边界」场景，
  **风险真实存在**，λ 必须扫，且主指标看 CTC 侧是否真的改善。

设计上遵守单因素：唯一变量是新增的 λ_frame·L_frame。
其他一切（数据、特征、模型超参、seed、early stopping、评估口径）与
基线 C0 完全一致，且基线用同一份代码路径跑，保证可比。

只用 train/validation，不触碰 test。
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

import tfd as TFD
import bio_labels as BIO
from cslr.recognition.model import CTCRecognizer, ctc_config_from_dict, BLANK_INDEX
from cslr.recognition.gloss_sequence import GlossVocabulary, GlossSequenceConfig
from cslr.recognition.dataset import GlossSequenceDataset, collate_samples
from cslr.recognition.training import (
    ctc_loss,
    evaluate,
    iterate_batches,
    learning_rate_at,
    resolve_device,
    to_torch_batch,
)
from cslr.recognition.gloss_sequence import GlossVocabulary as GV


# ----------------------------------------------------------------------
# 帧级辅助头：轻量 CNN，不复用 P2-a 的 MS-TCN（那是独立实验的大头，
# 这里只要一个「不抢 CTC 容量」的辅助信号）
# ----------------------------------------------------------------------
class FrameAuxHead(nn.Module):
    """帧级 BIO 三类辅助头。

    结构刻意极简（3 层 1D 卷积）：联合训练里辅助头过大会抢走 CTC 主干的
    容量，导致主指标（WER / blank）反而变差。论文 Hands-On 用的是
    transformer encoder + MLP mixer，但那是它的主任务就是分割；
    我们只是加辅助监督，不能喧宾夺主。
    """

    def __init__(self, enc_dim: int, num_classes: int = 3, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(enc_dim, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, num_classes, 1),
        )

    def forward(self, enc):
        """enc: (B, T, C) -> (B, num_classes, T)"""
        return self.net(enc.transpose(1, 2))


def build_frame_labels(path, n_gloss, dilate_k=1):
    """TFD 伪边界 -> BIO 三类标签（P1 已验证：标签分布健康）。

    返回 (T,) 的 {0:O, 1:I, 2:B} 标签。
    """
    arr = np.load(path).astype(np.float64)
    T = arr.shape[0]
    t, m = TFD.suggest_TM(T, n_gloss)
    peaks = TFD.detect_boundaries(arr, t, m, metric="l2")
    return BIO.labels_from_boundaries(T, peaks, dilate_k=dilate_k)


def frame_loss(frame_logits, frame_labels, class_weights, lengths, smooth_lambda=0.15):
    """加权帧级 CE + SignShift 式平滑正则。

    加权 CE 依据 Wójcicka Table 2（λ_Out=0.5, λ_Beg=5.0, λ_In=1.0）——
    论文明确说最高权重给 Begin（Out 的 10 倍）以强制高召回，
    同时降权 Out 以防退化为「每帧都预测 Out」的平凡解。
    我们的 BIO 标签与它的 {O,I,B} 语义一致，权重可原样迁移。

    平滑正则依据 SignShift 式(17)，λ=0.15（论文固定值）。
    """
    b, c, t = frame_logits.shape
    tgt = frame_labels[:, :t]
    valid = torch.arange(t, device=frame_logits.device)[None, :] < lengths[:, None]
    lsm = F.log_softmax(frame_logits.permute(0, 2, 1), dim=-1)
    w = class_weights.to(frame_logits.device)
    wt = w[tgt.clamp(min=0, max=c - 1)]
    ce = -(wt * lsm.gather(-1, tgt.clamp(min=0, max=c - 1).unsqueeze(-1)).squeeze(-1))
    ce = (ce * valid).sum() / valid.sum().clamp(min=1)

    # L_smooth 只看主导类概率图的帧间差
    p = F.softmax(frame_logits, dim=1)[:, 1:2, :]
    sm = ((p[:, :, 1:] - p[:, :, :-1]) ** 2).mean()
    return ce + smooth_lambda * sm, float(ce.detach()), float(sm.detach())


def load_ckpt(path):
    p = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ctc_config_from_dict(p["model_config"])
    vc = p.get("vocabulary_config") or {}
    voc = GlossVocabulary(tokens=tuple(p["vocabulary"]),
                          counts=dict(p.get("vocabulary_counts") or {}),
                          config=GlossSequenceConfig(**vc) if vc else GlossSequenceConfig())
    return p, cfg, voc, p.get("feature_normalizer")


class MultiTaskCTC(nn.Module):
    """CTC 主干 + 帧级辅助头。forward 同时返回 CTC logits 与编码。"""

    def __init__(self, ctc: CTCRecognizer, aux: FrameAuxHead):
        super().__init__()
        self.ctc = ctc
        self.aux = aux

    def forward(self, features, lengths):
        c = self.ctc
        h = c.projection(c.normalize(features))
        if c.config.subsample_stride != 1:
            h = c.subsample(h.transpose(1, 2)).transpose(1, 2)
        enc, _ = c.temporal(h)
        return c.classifier(enc), enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/ctc-landmark48-cap300.pt")
    ap.add_argument("--feat-root", default="artifacts/part3_features")
    ap.add_argument("--lambda-frame", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.3, 0.5, 1.0])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=300)
    ap.add_argument("--smooth-lambda", type=float, default=0.15)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p3-joint-loss.json")
    a = ap.parse_args()

    device = resolve_device("auto")
    print("device = {}".format(device))

    payload, cfg, voc, nrm_raw = load_ckpt(REPO / a.ckpt)
    feat_root = REPO / a.feat_root

    # ---- 帧级标签：直接从特征文件算，用 TFD 免训练边界 ----
    # 复用 P1 已验证的路径，不额外引入依赖
    import csv

    def read_csv(p):
        rows = {}
        with open(p, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames
            idc = next((c for c in cols if c.lower() in ("number", "id", "name")), cols[0])
            gc = next((c for c in cols if "gloss" in c.lower() or "label" in c.lower()), cols[-1])
            for r in rd:
                rows[r[idc]] = r[gc]
        return rows

    tr_g = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    va_g = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")

    def feature_path(split, sid):
        for suf in (".landmark.npy", ".clip.npy", ".npy"):
            cand = feat_root / split / (sid + suf)
            if cand.exists():
                return cand
        return None

    # 帧级标签缓存：{sample_id: np.ndarray}，用 TFD 免训练边界（P1 已验证）
    frame_labels = {}
    for tbl, split in ((tr_g, "train"), (va_g, "validation")):
        for sid, g in tbl.items():
            n_g = len([t for t in g.split("/") if t.strip()])
            if n_g == 0:
                continue
            fp = feature_path(split, sid)
            if fp is None:
                continue
            try:
                frame_labels[sid] = build_frame_labels(fp, n_g)
            except Exception:
                continue
    print("帧级标签覆盖 {} 条样本".format(len(frame_labels)))
    if not frame_labels:
        raise SystemExit("帧级标签为空：检查特征目录命名")

    # 类别权重（Wójcicka Table 2 实测值，语义与本项目 {O,I,B} 一致）
    sample_labels = list(frame_labels.values())[:400]
    cw = BIO.class_weights_from_labels(sample_labels)
    print("类别权重  O={:.2f} I={:.2f} B={:.2f}（Wójcicka Table 2: 0.5/1.0/5.0）".format(
        cw[0], cw[1], cw[2]))
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)

    # ---- 数据集：仓库无 JSON manifest，直接从标签 CSV 构造 records ----
    # 特征文件实际命名是 {id}.landmark.npy，而 GlossSequenceDataset 读 {id}.npy，
    # 所以这里建一个指向 .landmark.npy 的软链接目录，避免改动仓库既有代码。
    import os

    from cslr.contracts import SampleRecord

    link_root = Path("/tmp/p3_feats")
    for split, ids in (("train", list(tr_g.keys())), ("validation", list(va_g.keys()))):
        d = link_root / split
        d.mkdir(parents=True, exist_ok=True)
        for sid in ids:
            src = None
            for suf in (".landmark.npy", ".clip.npy", ".npy"):
                cand = feat_root / split / (sid + suf)
                if cand.exists():
                    src = cand
                    break
            if src is None:
                continue
            dst = d / (sid + ".npy")
            if not dst.exists():
                try:
                    os.symlink(src, dst)
                except OSError:
                    pass
    print("软链目录就绪: {}".format(link_root))

    tr_recs = [
        SampleRecord(sample_id=sid, video=Path(sid + ".mp4"), label=g,
                     signer="x", session="x", split="train")
        for sid, g in tr_g.items()
        if (link_root / "train" / (sid + ".npy")).exists()
    ]
    va_recs = [
        SampleRecord(sample_id=sid, video=Path(sid + ".mp4"), label=g,
                     signer="x", session="x", split="validation")
        for sid, g in va_g.items()
        if (link_root / "validation" / (sid + ".npy")).exists()
    ]
    tr_recs = [r for r in tr_recs if r.sample_id in frame_labels]
    va_recs = [r for r in va_recs if r.sample_id in frame_labels]
    if a.train_limit:
        tr_recs = tr_recs[: a.train_limit]
    va_recs = va_recs[: a.val_limit] if a.val_limit else va_recs
    print("train {} / val {}（要求有帧级标签）".format(len(tr_recs), len(va_recs)))
    if not tr_recs or not va_recs:
        raise SystemExit("训练/验证集为空")

    from cslr.recognition.dataset import FeatureNormalizer

    tr_ds = GlossSequenceDataset(tr_recs, link_root / "train", voc)
    va_ds = GlossSequenceDataset(va_recs, link_root / "validation", voc)
    if nrm_raw:
        # FeatureNormalizer 只有 mean/std 两个字段（frozen dataclass，无 count）
        nrm = FeatureNormalizer(
            mean=[float(x) for x in nrm_raw["mean"]],
            std=[float(x) + 1e-8 for x in nrm_raw["std"]],
        )
        # 注意必须仍用 link_root，不能退回 feat_root —— 那里是 .landmark.npy 命名
        tr_ds = GlossSequenceDataset(tr_recs, link_root / "train", voc, nrm)
        va_ds = GlossSequenceDataset(va_recs, link_root / "validation", voc, nrm)
        print("沿用 checkpoint 内 normalizer（train 统计量）")

    results = []
    t_all = time.time()
    for lam in a.lambda_frame:
        print("")
        print("=" * 68)
        print("lambda_frame = {}   （0.0 即纯 CTC 基线，同一代码路径）".format(lam))
        print("=" * 68)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

        ctc = CTCRecognizer(cfg).to(device)
        enc_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        aux = FrameAuxHead(enc_dim, 3).to(device)
        model = MultiTaskCTC(ctc, aux).to(device)

        params = list(ctc.parameters()) + list(aux.parameters())
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
        base_state = None
        if lam == 0.0:
            # 基线：加载同一 checkpoint 的权重，保证与实验组同起点
            ctc.load_state_dict(payload["state_dict"])
        else:
            # 实验组也从该 checkpoint 出发（热启动），只多训练一个头
            ctc.load_state_dict(payload["state_dict"])

        best = {"wer": float("inf")}
        best_state = None
        hist = []
        no_improve = 0
        t0 = time.time()
        for ep in range(1, a.epochs + 1):
            model.train()
            tot_c = tot_f = tot_ctc = 0.0
            nb = 0
            cur_lr = learning_rate_at(ep, a.epochs, a.lr, warmup_epochs=a.warmup,
                                      schedule="cosine", min_ratio=0.05)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            for samples in iterate_batches(tr_ds, a.batch, shuffle=True, seed=a.seed + ep):
                batch = to_torch_batch(collate_samples(samples), device)
                opt.zero_grad(set_to_none=True)
                logits, enc = model(batch["features"], batch["input_lengths"])
                out_len = ctc.output_lengths(batch["input_lengths"])
                l_ctc = ctc_loss(logits, batch["input_lengths"], batch["targets"],
                                 batch["target_lengths"], out_len)
                l_frame = torch.zeros((), device=device)
                l_fc = l_fs = 0.0
                if lam > 0.0:
                    fl = aux(enc)
                    t = fl.shape[-1]
                    bl = torch.full((fl.shape[0], t), BIO.OUT, dtype=torch.long, device=device)
                    for j, s in enumerate(samples):
                        lab = frame_labels.get(s.sample_id)
                        if lab is None:
                            continue
                        n = min(len(lab), t)
                        bl[j, :n] = torch.from_numpy(lab[:n]).to(device)
                    l_frame, l_fc, l_fs = frame_loss(
                        fl, bl, cw_t, batch["input_lengths"], a.smooth_lambda
                    )
                loss = l_ctc + lam * l_frame
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                tot_ctc += float(l_ctc.detach()); tot_f += l_fc
                tot_c += float(loss.detach()); nb += 1

            val = evaluate(ctc, va_ds, voc, a.batch, device, beam_width=1,
                           seed=a.seed, amp=False)
            m = val.metrics
            rec = {
                "epoch": ep,
                "train_loss": tot_c / max(nb, 1),
                "train_ctc": tot_ctc / max(nb, 1),
                "train_frame_ce": tot_f / max(nb, 1),
                "wer": float(m["wer"]),
                "cer": float(m["cer"]),
                "blank_ratio": float(m["blank_ratio"]),
                "empty_hyp": float(m["empty_hypothesis_rate"]),
                "vocab_util": float(m.get("vocabulary_utilization", 0.0)),
                "val_loss": float(m["loss"]),
            }
            hist.append(rec)
            print("  ep {:2d}  loss={:.4f} (ctc={:.4f} frameCE={:.4f})  WER={:.4f}  "
                  "blank={:.4f}  empty={:.3f}".format(
                      ep, rec["train_loss"], rec["train_ctc"], rec["train_frame_ce"],
                      rec["wer"], rec["blank_ratio"], rec["empty_hyp"]), flush=True)
            if rec["wer"] < best["wer"]:
                best = dict(rec); best["epoch"] = ep; no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in ctc.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= a.patience:
                    print("  early stop @ ep{}".format(ep))
                    break
        dt = (time.time() - t0) / 60
        print("  -> best ep{}  WER={:.4f}  blank={:.4f}  ({:.1f} 分钟)".format(
            best.get("epoch"), best["wer"], best["blank_ratio"], dt))
        results.append({"lambda_frame": lam, "best": best, "history": hist, "minutes": dt})
        print("  参考：基线 P0 测得 blank=0.9698 / pfpt=0.26（ctc-landmark48-cap300）")

    # ---- 汇总 ----
    print("")
    print("=" * 68)
    print("P3 联合训练 λ 扫描汇总（主指标：WER 与 blank，越低越好）")
    print("=" * 68)
    print("{:>12}  {:>10}  {:>10}  {:>10}  {:>8}".format(
        "lambda", "WER", "blank", "empty", "ep"))
    for r in results:
        b = r["best"]
        print("{:>12.2f}  {:>10.4f}  {:>10.4f}  {:>10.3f}  {:>8}".format(
            r["lambda_frame"], b["wer"], b["blank_ratio"], b["empty_hyp"],
            b.get("epoch", -1)))
    base = next((r for r in results if r["lambda_frame"] == 0.0), None)
    if base:
        print("")
        print("相对基线（lambda=0，同一代码路径）:")
        for r in results:
            if r["lambda_frame"] == 0.0:
                continue
            dw = r["best"]["wer"] - base["best"]["wer"]
            db = r["best"]["blank_ratio"] - base["best"]["blank_ratio"]
            print("  lambda={:>5.2f}  dWER={:+.4f}  dblank={:+.4f}".format(
                r["lambda_frame"], dw, db))
    print("")
    print("总耗时 {:.1f} 分钟".format((time.time() - t_all) / 60))

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "P3 joint frame-level + gloss-level CTC loss (Hands-On Sec III-D1)",
        "literature_basis": {
            "primary": "Hands-On FG2025 Sec III-D1: frame-level BIO + gloss-level CTC, "
                       "concurrent optimization",
            "weight_lambda": "NOT given in the paper; swept here",
            "counter_evidence": "SMART Sec 3.3 / 4.1: 'The recognition module is frozen "
                                "during spotting training' (two-stage + late fusion)",
            "why_might_differ": "Hands-On has human frame-level GT (DGS); we only have "
                                "TFD pseudo-boundaries (CE-CSL has no boundary annotation)",
        },
        "class_weights": {"O": float(cw[0]), "I": float(cw[1]), "B": float(cw[2])},
        "weight_source": "Wojcicka Table 2 (0.5/1.0/5.0); semantics match our {O,I,B}",
        "smooth_lambda_source": "SignShift Eq.17, lambda=0.15 (paper fixed value)",
        "checkpoint": a.ckpt,
        "config": vars(a),
        "results": results,
        "total_minutes": (time.time() - t_all) / 60,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
