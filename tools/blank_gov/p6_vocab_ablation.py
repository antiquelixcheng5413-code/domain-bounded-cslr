# -*- coding: utf-8 -*-
"""P6 词表规模消融：解除 cap300 裁剪能否降低 WER。

**这是 FYP 场景下最高价值的实验。** 起因是今天五次 blank 方向实验后
发现的一个此前未被记录的事实：

`build_ordered_vocabulary(min_frequency, max_tokens)` 有**两个独立的人为裁剪**：
    min_frequency=2   砍掉所有只出现 1 次的 gloss（train 里有 1893 个）
    max_tokens=300    只保留频次最高的 300 个

对 dev 的覆盖（实测 515 条 / 2842 token / 822 类）：
    配置              词表类数   dev token 覆盖   dev 样本完全覆盖
    cap300 (当前)          301        69.6%            8.9%
    min_freq=2 全量        1828        90.1%           54.8%
    min_freq=1 全量        3517        93.8%           71.1%

→ **当前配置下 91.1% 的 dev 样本其 gloss 不全在词表内**，
   模型无论怎么训练都吐不出词表外的词。这给 WER 0.85 提供了约 0.30 的硬下界。

本实验：固定特征/模型/超参/seed，唯一变量是词表规模。
主指标 WER（dev），辅助：blank 率、vocabulary_utilization、
以及**分桶 WER**（OOV 样本 vs 词汇内样本）—— 后者能直接量化
「WER 里有多少是裁剪造成的」。

严格约束：只用 train/dev，**不触碰 test**。
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.contracts import SampleRecord
from cslr.recognition.model import CTCConfig, CTCRecognizer, BLANK_INDEX
from cslr.recognition.gloss_sequence import (
    GlossVocabulary,
    GlossSequenceConfig,
    build_ordered_vocabulary,
)
from cslr.recognition.dataset import GlossSequenceDataset, FeatureNormalizer
from cslr.recognition.training import (
    TrainingConfig,
    ctc_loss,
    evaluate,
    iterate_batches,
    learning_rate_at,
    resolve_device,
    to_torch_batch,
)
from cslr.recognition.dataset import collate_samples
from cslr.recognition.metrics import gloss_metrics


def read_split(split):
    out = {}
    p = REPO / "data/raw/CE-CSL/label" / ("dev.csv" if split == "validation" else "train.csv")
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["Number"]] = r["Gloss"]
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


def make_records(labels, feat_root, limit=None):
    out = []
    for sid, g in labels.items():
        if not (feat_root / (sid + ".npy")).exists():
            continue
        if not [t for t in g.split("/") if t.strip()]:
            continue
        out.append(SampleRecord(sample_id=sid, video=Path(sid + ".mp4"),
                                label=g, signer="x", session="x",
                                split="train" if "train" in str(feat_root) else "validation"))
    out.sort(key=lambda r: r.sample_id)
    if limit:
        out = out[:limit]
    return out


def bucket_wer(model, dataset, voc, device, batch, nrm):
    """分桶 WER：把 dev 样本按「是否含 OOV gloss」分两组分别算 WER。

    直接回答「WER 里有多少是词表裁剪造成的」。
    OOV = 该样本至少有一个 gloss 不在词表内。
    """
    model.eval()
    preds, refs, is_oov = [], [], []
    for samples in iterate_batches(dataset, batch, shuffle=False, seed=42):
        b = to_torch_batch(collate_samples(samples), device)
        with torch.no_grad():
            logits = model(b["features"], b["input_lengths"])
        out_len = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        from cslr.recognition.training import decode_batch
        dec, _, _ = decode_batch(lp, out_len.cpu().tolist(), 1)
        for row, ids in enumerate(dec):
            preds.append(voc.decode(ids))
            refs.append(b["tokens"][row])
            toks = [t for t in b["tokens"][row]]
            miss = [t for t in toks if t not in set(voc.tokens)]
            is_oov.append(len(miss) > 0)
    res = {}
    for name, sel in (("in_vocab", [i for i, o in enumerate(is_oov) if not o]),
                      ("has_oov", [i for i, o in enumerate(is_oov) if o])):
        if not sel:
            res[name] = None
            continue
        m = gloss_metrics([refs[i] for i in sel], [preds[i] for i in sel],
                          vocabulary_size=voc.size)
        res[name] = {"wer": float(m["wer"]), "n": len(sel)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-view", default="full")
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
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=515)
    ap.add_argument("--configs", default="300:2,1828:2,3517:1",
                    help="逗号分隔的 max_tokens:min_frequency。"
                         "**词表总是用全量 train 标签构建**（不受 --train-limit 影响），"
                         "否则缩小 train 集会连带缩小词表，破坏单因素")
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p6-vocab-ablation.json")
    a = ap.parse_args()

    device = resolve_device("auto")
    print("device = {}".format(device))

    tr_labels = read_split("train")
    va_labels = read_split("validation")

    tr_root = ensure_link_dir("train")
    va_root = ensure_link_dir("validation")
    tr_recs = make_records(tr_labels, tr_root, a.train_limit)
    va_recs = make_records(va_labels, va_root, a.val_limit)
    print("train {} / dev {}".format(len(tr_recs), len(va_recs)))
    if not tr_recs or not va_recs:
        raise SystemExit("样本为空")

    nrm = FeatureNormalizer.fit(
        [np.load(tr_root / (r.sample_id + ".npy")) for r in tr_recs[:600]]
    )
    print("normalizer 来自 {} 条 train 样本".format(min(600, len(tr_recs))))

    configs = []
    for item in a.configs.split(","):
        mt, mf = item.split(":")
        configs.append((int(mt), int(mf)))

    results = []
    t_all = time.time()
    for max_tokens, min_freq in configs:
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        # 词表始终用**全量 train 标签**构建，不受 --train-limit 影响。
        # 若用受限子集建词表，缩小的是词表而非训练数据，就不是单因素了。
        voc, counts = build_ordered_vocabulary(
            (g for g in tr_labels.values()), min_frequency=min_freq,
            max_tokens=max_tokens
        )
        vset = set(voc.tokens)
        # 覆盖统计（dev 侧）
        dev_tok_cov = dev_tok_tot = 0
        dev_ok = 0
        for r in va_recs:
            toks = [t for t in r.label.split("/") if t.strip()]
            if not toks:
                continue
            dev_tok_tot += len(toks)
            dev_tok_cov += sum(1 for t in toks if t in vset)
            if all(t in vset for t in toks):
                dev_ok += 1
        cov = dev_tok_cov / max(dev_tok_tot, 1)
        ok_rate = dev_ok / max(len(va_recs), 1)

        print("")
        print("=" * 70)
        print("词表 max_tokens={} min_frequency={}  ->  {} 类".format(
            max_tokens, min_freq, voc.size))
        print("dev token 覆盖 {:.1%}   dev 样本完全覆盖 {:.1%}".format(cov, ok_rate))
        print("=" * 70)

        cfg = CTCConfig(input_size=368, vocabulary_size=voc.size,
                        hidden_size=a.hidden, num_layers=a.layers,
                        dropout=a.dropout, bidirectional=True,
                        projection_size=256, subsample_stride=1)
        model = CTCRecognizer(cfg).to(device)

        tr_ds = GlossSequenceDataset(tr_recs, tr_root, voc, nrm,
                                    feature_view=a.feature_view)
        va_ds = GlossSequenceDataset(va_recs, va_root, voc, nrm,
                                    feature_view=a.feature_view)

        tcfg = TrainingConfig(epochs=a.epochs, batch_size=a.batch,
                             learning_rate=a.lr, weight_decay=a.weight_decay,
                             seed=a.seed, device=str(device), amp=True,
                             early_stopping_patience=a.patience, beam_width=1,
                             min_epochs=1, sequence_length=48,
                             normalize_features=False)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                                weight_decay=a.weight_decay)
        use_amp = tcfg.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best = {"wer": float("inf")}
        best_state = None
        hist = []
        no_imp = 0
        t0 = time.time()
        for ep in range(1, tcfg.epochs + 1):
            model.train()
            ep_loss = 0.0
            seen = 0
            cur_lr = learning_rate_at(ep, tcfg.epochs, a.lr,
                                      warmup_epochs=a.warmup,
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
                   "vocab_util": float(m.get("vocabulary_utilization", 0.0)),
                   "empty_hyp": float(m["empty_hypothesis_rate"]),
                   "lr": cur_lr}
            hist.append(rec)
            print("  ep {:2d}  loss={:.4f}  WER={:.4f}  CER={:.4f}  blank={:.4f}  "
                  "util={:.3f}".format(ep, rec["train_loss"], rec["wer"], rec["cer"],
                                       rec["blank_ratio"], rec["vocab_util"]), flush=True)
            if rec["wer"] < best["wer"]:
                best = dict(rec); best["epoch"] = ep; no_imp = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_imp += 1
                if no_imp >= a.patience:
                    print("  early stop @ ep{}".format(ep))
                    break
        dt = (time.time() - t0) / 60

        # 分桶 WER（用最佳权重）
        if best_state is not None:
            model.load_state_dict(best_state)
        buckets = bucket_wer(model, va_ds, voc, device, a.batch, nrm)
        def _fmt(x):
            if x is None:
                return "n/a"
            return "WER={:.4f} (n={})".format(x["wer"], x["n"])
        print("  分桶: 词汇内样本 {}   含 OOV 样本 {}".format(
            _fmt(buckets["in_vocab"]), _fmt(buckets["has_oov"])))
        print("  -> best ep{}  WER={:.4f}  ({:.1f} 分钟)".format(
            best.get("epoch"), best["wer"], dt))

        results.append({
            "max_tokens": max_tokens, "min_frequency": min_freq,
            "vocab_size": int(voc.size),
            "dev_token_coverage": float(cov),
            "dev_sample_full_coverage": float(ok_rate),
            "best": best, "buckets": buckets, "history": hist, "minutes": dt,
        })

    # ---- 汇总 ----
    print("")
    print("=" * 78)
    print("P6 词表规模消融汇总（唯一变量 = 词表规模）")
    print("=" * 78)
    print("{:>8} {:>6} {:>10} {:>10} {:>9} {:>9} {:>10}".format(
        "max_tok", "minf", "词表", "token覆盖", "样本覆盖", "WER", "含OOV WER"))
    for r in results:
        b = r["buckets"]["has_oov"]
        print("{:>8} {:>6} {:>10} {:>10} {:>9} {:>9.4f} {:>10}".format(
            r["max_tokens"], r["min_frequency"], r["vocab_size"],
            "{:.1%}".format(r["dev_token_coverage"]),
            "{:.1%}".format(r["dev_sample_full_coverage"]),
            r["best"]["wer"],
            "-" if b is None else "%.4f" % b["wer"]))
    base = next((r for r in results if r["vocab_size"] == 301), results[0])
    print("")
    print("相对当前 cap301 基线：")
    for r in results:
        if r is base:
            continue
        print("  max_tokens={:<5} dWER={:+.4f}  token覆盖 {:+.1%}  样本覆盖 {:+.1%}".format(
            r["max_tokens"], r["best"]["wer"] - base["best"]["wer"],
            r["dev_token_coverage"] - base["dev_token_coverage"],
            r["dev_sample_full_coverage"] - base["dev_sample_full_coverage"]))

    best_r = min(results, key=lambda r: r["best"]["wer"])
    gain = base["best"]["wer"] - best_r["best"]["wer"]
    print("")
    print("最佳：max_tokens={} min_frequency={}  WER={:.4f}  ({:+.4f} vs cap301)".format(
        best_r["max_tokens"], best_r["min_frequency"], best_r["best"]["wer"], gain))
    if gain > 0.03:
        verdict = "词表裁剪是 WER 的主要瓶颈：解除裁剪后 WER 显著下降"
    elif gain > 0.01:
        verdict = "词表裁剪有中等影响：解除裁剪后 WER 明显下降"
    else:
        verdict = "词表裁剪影响有限：说明主因仍在特征判别力（AUC 0.68）"
    print("判定：{}".format(verdict))
    print("总耗时 {:.1f} 分钟".format((time.time() - t_all) / 60))

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "P6 vocabulary-size ablation (cap300 vs full)",
        "single_factor": "vocabulary size only; features/model/seed/schedule identical",
        "motivation": {
            "vocab_builder": "build_ordered_vocabulary(min_frequency, max_tokens) "
                             "applies TWO independent human-imposed cuts",
            "train_gloss_classes": 3841,
            "dev_token_coverage_cap300": 0.696,
            "dev_sample_full_coverage_cap300": 0.089,
            "implication": "91.1% of dev samples cannot be fully expressed by the "
                           "model regardless of training quality; WER has a hard "
                           "lower bound around 0.30 from vocabulary truncation",
        },
        "config": vars(a),
        "results": results,
        "best_config": {"max_tokens": best_r["max_tokens"],
                        "min_frequency": best_r["min_frequency"],
                        "wer": best_r["best"]["wer"]},
        "wer_gain_vs_cap301": gain,
        "verdict": verdict,
        "total_minutes": (time.time() - t_all) / 60,
        "test_split_read": False,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print("收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
