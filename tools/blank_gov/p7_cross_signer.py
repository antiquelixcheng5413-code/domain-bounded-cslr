# -*- coding: utf-8 -*-
"""P7 · cross-signer 泛化分析（零训练分解 + LOSO 三折）

动机：CE-CSL 的 train/dev/test **共享同一批 12 个 signer（A~L）**，
所以当前 WER 0.85 是 **seen-signer** 条件下的结果。真实部署会遇到未见过的手语者，
泛化能力未知。本脚本量化这个差距。

两个部分：

A) **按 signer 分解（零训练成本）**
   直接用已有 cap300 检查点逐 signer 算 dev WER。
   若各 signer WER 差异大 -> signer 身份是重要因素，seen-signer 指标偏乐观。

B) **LOSO（leave-one-signer-out）三折**
   留出 signer S 的**全部** train 样本训练，只在 S 的 dev 样本上评测。
   词表**只从剩余 train 构建**（否则会泄漏 S 的 gloss，协议不干净）。
   对照 = 同一 S 的 dev 样本在「全量 train 模型」上的 WER。
   差值 = 未见 signer 的泛化代价。

判据（预先写死）：
  - 若 LOSO-WER 相对对照的退化 > 0.05（绝对 WER），
    则「seen-signer 划分显著高估性能」成立，属需在论文中报告的限制。
  - 若退化 <= 0.05，则 signer 泛化不是主要瓶颈，可排除该方向。

只读 train/dev，不触碰 test split。
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cslr.contracts import SampleRecord
from cslr.recognition.model import CTCConfig, CTCRecognizer, ctc_config_from_dict
from cslr.recognition.gloss_sequence import (
    build_ordered_vocabulary, GlossVocabulary, GlossSequenceConfig,
)
from cslr.recognition.dataset import GlossSequenceDataset, FeatureNormalizer
from cslr.recognition.training import (
    TrainingConfig, ctc_loss, evaluate, iterate_batches,
    to_torch_batch, learning_rate_at, resolve_device, decode_batch,
)
from cslr.recognition.dataset import collate_samples
from cslr.recognition.metrics import gloss_metrics


def read_split(split):
    """返回 {sample_id: (gloss, signer)}。signer 来自 Translator 列。"""
    out = {}
    fname = "dev.csv" if split == "validation" else "train.csv"
    p = REPO / "data/raw/CE-CSL/label" / fname
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            signer = (r.get("Translator") or "").strip()
            # 兜底：Translator 缺失时用 id 前缀
            if not signer:
                signer = r["Number"].split("-")[0]
            out[r["Number"]] = (r["Gloss"], signer)
    return out


def ensure_link_dir(split, dim=368):
    """特征实际是 {id}.landmark.npy，而 Dataset 读 {id}.npy -> 建软链目录。"""
    root = REPO / ".vocab_link" / split
    root.mkdir(parents=True, exist_ok=True)
    src_dir = REPO / "artifacts/part3_features" / split
    for p in sorted(src_dir.glob("*.landmark.npy")):
        sid = p.name[: -len(".landmark.npy")]
        dst = root / (sid + ".npy")
        if not dst.exists():
            try:
                os.symlink(p, dst)
            except OSError:
                pass
    return root


def make_records(table, feat_root, limit=None, signers=None):
    out = []
    for sid, (g, signer) in table.items():
        if signers is not None and signer not in signers:
            continue
        if not (feat_root / (sid + ".npy")).exists():
            continue
        if not [t for t in g.split("/") if t.strip()]:
            continue
        out.append(SampleRecord(sample_id=sid, video=Path(sid + ".mp4"),
                                label=g, signer=signer, session="x",
                                split="train" if "train" in str(feat_root) else "validation"))
    out.sort(key=lambda r: r.sample_id)
    if limit:
        out = out[:limit]
    return out


# ---------------------------------------------------------------- A) 逐 signer 分解

def per_signer_wer(model, recs, feat_root, voc, device, batch, nrm):
    """用给定模型在 recs 上按 signer 分组算 WER。返回 {signer: {...}}。

    注意：`SequenceSample` 只有 sample_id/tokens，**没有 signer 字段**，
    所以必须从 recs 建 sample_id -> signer 的映射来分组。
    """
    signer_of = {r.sample_id: r.signer for r in recs}
    ds = GlossSequenceDataset(recs, feat_root, voc, nrm)
    buckets = {}
    for samples in iterate_batches(ds, batch, shuffle=False, seed=42):
        b = to_torch_batch(collate_samples(samples), device)
        with torch.no_grad():
            logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        dec, _, _ = decode_batch(lp, ol.cpu().tolist(), 1)
        for row, ids in enumerate(dec):
            sid = samples[row].sample_id
            signer = signer_of[sid]
            buckets.setdefault(signer, ([], []))
            buckets[signer][0].append(voc.decode(ids))
            buckets[signer][1].append(b["tokens"][row])
    out = {}
    for signer, (preds, refs) in sorted(buckets.items()):
        m = gloss_metrics(refs, preds, vocabulary_size=voc.size)
        out[signer] = {
            "wer": float(m["wer"]),
            "cer": float(m["cer"]),
            "blank_ratio": float(m["blank_ratio"]),
            "vocab_util": float(m.get("vocabulary_utilization", 0.0)),
            "n": len(refs),
        }
        print("   signer {:>2}  n={:>3}  WER={:.4f}  CER={:.4f}  blank={:.4f}".format(
            signer, out[signer]["n"], out[signer]["wer"], out[signer]["cer"],
            out[signer]["blank_ratio"]), flush=True)
    ws = [v["wer"] for v in out.values()]
    if len(ws) > 1:
        print("   -> signer 间 WER 极差 {:.4f}（max {:.4f} / min {:.4f}）".format(
            max(ws) - min(ws), max(ws), min(ws)))
        print("   -> 变异系数 CV = {:.3f}".format(
            float(np.std(ws) / max(np.mean(ws), 1e-9))))
    return out


def _root_of(recs):
    """保留占位以防外部引用；实际调用已改为显式传 feat_root。"""
    raise NotImplementedError


_CUR_ROOT = None


# ---------------------------------------------------------------- 训练（LOSO 用）

def train_one(tr_recs, tr_root, va_recs, va_root, voc, nrm, a, device, tag):
    cfg = CTCConfig(input_size=368, vocabulary_size=voc.size,
                    hidden_size=a.hidden, num_layers=a.layers, dropout=a.dropout,
                    bidirectional=True, projection_size=256, subsample_stride=1)
    model = CTCRecognizer(cfg).to(device)
    tr_ds = GlossSequenceDataset(tr_recs, tr_root, voc, nrm, feature_view="full")
    va_ds = GlossSequenceDataset(va_recs, va_root, voc, nrm, feature_view="full")

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
        for samples in iterate_batches(tr_ds, a.batch, shuffle=True, seed=a.seed + ep):
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
        print("   ep {:2d}  loss={:.4f}  WER={:.4f}  CER={:.4f}  blank={:.4f}".format(
            ep, rec["train_loss"], rec["wer"], rec["cer"], rec["blank_ratio"]), flush=True)
        if rec["wer"] < best["wer"]:
            best = dict(rec); best["epoch"] = ep; no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= a.patience:
                print("   early stop @ ep{}".format(ep))
                break
    dt = (time.time() - t0) / 60
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, hist, dt


def main():
    global _CUR_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--loso-signers", default="A,I,L",
                    help="LOSO 折：大样本 signer / 极小样本 signer / 中位 signer")
    ap.add_argument("--baseline-ckpt", default="artifacts/checkpoints/ctc-landmark48-cap300.pt")
    ap.add_argument("--skip-a", action="store_true", help="跳过 A 部分（不加载基线检查点）")
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p7-cross-signer.json")
    a = ap.parse_args()

    device = resolve_device("auto")
    print("device = {}".format(device))

    tr_table = read_split("train")
    va_table = read_split("validation")
    tr_root = ensure_link_dir("train")
    va_root = ensure_link_dir("validation")
    _CUR_ROOT = va_root

    tr_recs = make_records(tr_table, tr_root)
    va_recs = make_records(va_table, va_root)
    print("train {} / dev {}".format(len(tr_recs), len(va_recs)))

    tr_by_signer = Counter(r.signer for r in tr_recs)
    va_by_signer = Counter(r.signer for r in va_recs)
    print("\n=== signer 样本分布 ===")
    print("signer   train    dev")
    for sg in sorted(set(tr_by_signer) | set(va_by_signer)):
        print("   {:>2}   {:>5}  {:>5}".format(sg, tr_by_signer[sg], va_by_signer[sg]))

    receipt = {
        "protocol": {
            "splits": "train + dev only（test 未触碰）",
            "vocab": "max_tokens={} min_freq={}，**每折只用该折训练集构建**".format(
                a.max_tokens, a.min_freq),
            "loso_signers": a.loso_signers,
            "seed": a.seed,
            "criterion": "LOSO-WER 相对「全量 train 模型在同一 signer dev 上」的退化 > 0.05 "
                         "则判定 seen-signer 划分显著高估性能",
        },
        "signer_distribution": {
            "train": dict(tr_by_signer),
            "dev": dict(va_by_signer),
        },
    }

    nrm = FeatureNormalizer.fit(
        [np.load(tr_root / (r.sample_id + ".npy")) for r in tr_recs[:600]]
    )
    voc_full, _ = build_ordered_vocabulary(
        (g for g, _ in tr_table.values()), min_frequency=a.min_freq,
        max_tokens=a.max_tokens)
    print("\n全量 train 词表 {} 类".format(voc_full.size))

    # ---------------- A) 逐 signer 分解（零训练） ----------------
    if not a.skip_a:
        print("\n" + "=" * 70)
        print("A) 按 signer 分解（已有 cap300 检查点，seen-signer 条件）")
        print("=" * 70)
        payload = torch.load(REPO / a.baseline_ckpt, map_location="cpu", weights_only=False)
        cfg = ctc_config_from_dict(payload["model_config"])
        model = CTCRecognizer(cfg)
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        vc = payload.get("vocabulary_config") or {}
        voc_ck = GlossVocabulary(
            tokens=tuple(payload["vocabulary"]),
            counts=dict(payload.get("vocabulary_counts") or {}),
            config=GlossSequenceConfig(**vc) if vc else GlossSequenceConfig())
        nrm_ck = payload.get("feature_normalizer")
        if nrm_ck:
            nrm_a = FeatureNormalizer(
                mean=np.asarray(nrm_ck["mean"], dtype=np.float32),
                std=np.asarray(nrm_ck["std"], dtype=np.float32) + 1e-8)
        else:
            nrm_a = nrm
        print("检查点词表 {} 类".format(voc_ck.size))
        receipt["per_signer_seen"] = per_signer_wer(
            model, va_recs, va_root, voc_ck, device, a.batch, nrm_a)
        receipt["baseline_ckpt"] = a.baseline_ckpt

    # ---------------- B) LOSO ----------------
    loso = [s.strip() for s in a.loso_signers.split(",") if s.strip()]
    folds = []
    for sg in loso:
        print("\n" + "=" * 70)
        print("B) LOSO 折：留出 signer {}".format(sg))
        print("=" * 70)
        tr_sub = [r for r in tr_recs if r.signer != sg]
        va_sub = [r for r in va_recs if r.signer == sg]
        print("训练集 {} 条（去掉 signer {} 的 {} 条）  评测 {} 条 dev".format(
            len(tr_sub), sg, len(tr_recs) - len(tr_sub), len(va_sub)))
        if len(tr_sub) < 100 or len(va_sub) < 5:
            print("样本不足，跳过该折")
            continue

        # 词表只用该折训练集构建，避免泄漏 signer sg 的 gloss
        labels_sub = {r.sample_id: r.label for r in tr_sub}
        voc_sub, _ = build_ordered_vocabulary(
            labels_sub.values(), min_frequency=a.min_freq, max_tokens=a.max_tokens)
        nrm_sub = FeatureNormalizer.fit(
            [np.load(tr_root / (r.sample_id + ".npy")) for r in tr_sub[:600]])
        print("该折词表 {} 类（对照：全量 {} 类）".format(voc_sub.size, voc_full.size))

        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        m_loso, best_loso, hist, dt = train_one(
            tr_sub, tr_root, va_sub, va_root, voc_sub, nrm_sub, a, device,
            "loso-" + sg)
        print("   LOSO 最佳 WER={:.4f} (ep{})  用时 {:.1f} 分钟".format(
            best_loso["wer"], best_loso["epoch"], dt))

        # 对照：同一位 signer 的 dev 样本在「见过他」的全量模型上表现
        # 用同一 LOSO 模型先算一遍对照是不对的（它没见过 sg），
        # 所以对照必须来自全量 train 模型 —— 重新训一遍太贵，
        # 这里改为：直接引用 A 部分或单独训一个 full 模型。
        folds.append({
            "signer": sg,
            "n_train_sub": len(tr_sub),
            "n_dev_eval": len(va_sub),
            "vocab_size": voc_sub.size,
            "loso_wer": best_loso["wer"],
            "loso_cer": best_loso["cer"],
            "loso_blank": best_loso["blank_ratio"],
            "loso_best_epoch": best_loso["epoch"],
            # 对照：同一位 signer 的 dev 样本，在**见过他**的全量模型上的表现。
            # 取自 A 部分（cap300 基线检查点）。这是 seen-signer 条件。
            "seen_wer": (receipt.get("per_signer_seen", {}).get(sg) or {}).get("wer"),
            "history": hist,
            "minutes": dt,
        })
        del m_loso
        torch.cuda.empty_cache()

    receipt["loso_folds"] = folds
    if folds:
        w = [f["loso_wer"] for f in folds]
        receipt["loso_summary"] = {
            "n_folds": len(folds),
            "wer_mean": float(np.mean(w)),
            "wer_min": float(np.min(w)),
            "wer_max": float(np.max(w)),
        }
        # 泛化代价：LOSO（未见该 signer）相对 seen（见过他）的 WER 退化
        deltas = [(f["signer"], f["loso_wer"], f["seen_wer"],
                   f["loso_wer"] - f["seen_wer"])
                  for f in folds if f.get("seen_wer") is not None]
        if deltas:
            receipt["generalization_cost"] = [
                {"signer": s, "loso_wer": lw, "seen_wer": sw, "delta": d}
                for s, lw, sw, d in deltas]
        print("\n" + "=" * 70)
        print("LOSO 汇总（对照 = 同 signer 在见过他的模型上的表现）")
        print("=" * 70)
        print("signer   n_dev   seen-WER   LOSO-WER    退化")
        for f in folds:
            sw = f.get("seen_wer")
            d = (f["loso_wer"] - sw) if sw is not None else None
            print("   {:>2}     {:>4}   {}   {:.4f}    {}".format(
                f["signer"], f["n_dev_eval"],
                "  --  " if sw is None else "{:.4f}".format(sw),
                f["loso_wer"],
                "  --  " if d is None else "{:+.4f}".format(d)))
        print("  LOSO 平均 WER = {:.4f}".format(np.mean(w)))
        ds = [d for _, _, _, d in deltas]
        if ds:
            print("  泛化代价（ΔWER）平均 {:+.4f}  最大 {:+.4f}".format(
                float(np.mean(ds)), max(ds)))
            verdict = ("未见 signer 显著更难（平均退化 > 0.05）—— "
                       "seen-signer 划分显著高估性能，**属需在论文报告的限制**"
                       if float(np.mean(ds)) > 0.05 else
                       "退化 <= 0.05，signer 泛化不是主要瓶颈，可排除该方向")
            print("  判定：{}".format(verdict))
            receipt["verdict"] = verdict

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
