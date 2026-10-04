# -*- coding: utf-8 -*-
"""P25 · 修复后交付：可读的识别结果（真实 dev，off-by-one 已修）

## 目的

P23 修复了 `collate_samples` 的 off-by-one。本脚本在**真实 dev 515 条**上
重训配置 A（token / 词表 301 / 30 epoch），并导出：

1. 整体指标（WER / CER / 输出长度 / 输出种类数 / blank 率）
2. **逐样本对照表**（参考 vs 识别），供人工检视「是否可读」
3. 分桶统计：词汇内样本 vs 含 OOV 样本
4. 高频词混淆 Top-N

这是用户要的「可读的识别结果」的直接证据。

## 与 P18/P24 的差别

P18/P24 已用修好的 `collate_samples` 跑过，本脚本额外：
- 保存 checkpoint，供后续推理
- 导出逐样本对照（含严格/宽松两套口径）
- 统计**识别正确的样本里**平均命中几个词（区分「完全错」与「部分对」）

## 口径声明（必须并列给两个数）

- **宽松 WER**：ref 与 hyp 都经 `<unk>` 折叠（与既往工作可比）
- **严格 WER**：ref 保留真实 OOV token（真实水平）

只读 train/dev，**不触碰 test split**。
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

REPO = Path(__file__).resolve().parents[2]
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


def ensure_link_dir(split):
    root = REPO / ".vocab_link" / split
    root.mkdir(parents=True, exist_ok=True)
    for p in sorted((REPO / "artifacts/part3_features" / split).glob("*.landmark.npy")):
        sid = p.name[: -len(".landmark.npy")]
        dst = root / (sid + ".npy")
        if not dst.exists():
            try:
                os.symlink(p, dst)
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


@torch.no_grad()
def decode_all(model, ds, device, batch, voc, ref_by_id, ref_raw):
    model.eval()
    rows = []
    for samples in iterate_batches(ds, batch, shuffle=False, seed=42):
        b = to_torch_batch(collate_samples(samples), device)
        logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        p = np.exp(lp)
        blank_ratio = float((p[:, :, 0].argmax(axis=1) == 0).mean())
        dec, _, _ = decode_batch(lp, ol.cpu().numpy(), 1)
        for r, ids in enumerate(dec):
            sid = b["sample_ids"][r]
            hyp = voc.decode(list(ids))
            ref_loose = ref_by_id[sid]
            ref_strict = ref_raw[sid]
            d_loose = levenshtein(list(ref_loose), hyp)[0]
            d_strict = levenshtein(list(ref_strict), hyp)[0]
            rows.append({
                "sid": sid, "hyp": hyp,
                "ref_loose": ref_loose, "ref_strict": ref_strict,
                "edit_loose": d_loose, "edit_strict": d_strict,
                "n_ref": len(ref_strict), "n_hyp": len(hyp),
                "hits": max(len(ref_strict) - d_strict, 0),
                "blank_ratio_this_batch": blank_ratio,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--n-show", type=int, default=25)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p25-readable-output.json")
    a = ap.parse_args()

    device = resolve_device("auto")
    print("device = {}".format(device))

    tr_labels = read_split("train")
    va_labels = read_split("validation")
    tr_root = ensure_link_dir("train")
    va_root = ensure_link_dir("validation")
    tr_recs = make_records(tr_labels, tr_root, "train")
    va_recs = make_records(va_labels, va_root, "validation")
    print("train {} / dev {}".format(len(tr_recs), len(va_recs)))

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    voc, _ = build_ordered_vocabulary(
        (g for g in tr_labels.values()), min_frequency=a.min_freq,
        max_tokens=a.max_tokens)
    vset = set(voc.tokens)
    nrm = FeatureNormalizer.fit(
        [np.load(tr_root / (r.sample_id + ".npy")) for r in tr_recs[:600]])

    tr_ds = GlossSequenceDataset(tr_recs, tr_root, voc, nrm, feature_view="full")
    va_ds = GlossSequenceDataset(va_recs, va_root, voc, nrm, feature_view="full")

    # 严格 ref = CSV 原始 gloss；宽松 ref = 经 <unk> 折叠
    ref_raw = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()]
               for r in va_recs}
    ref_loose = {r.sample_id: [voc.tokens[i] for i in voc.encode(r.label)]
                 for r in va_recs}

    cfg = CTCConfig(input_size=368, vocabulary_size=voc.size, hidden_size=256,
                    num_layers=2, dropout=0.3, bidirectional=True,
                    projection_size=256, subsample_stride=1)
    model = CTCRecognizer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print("")
    print("训练（off-by-one 已修）")
    t0 = time.time()
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = seen = 0
        cur_lr = learning_rate_at(ep, a.epochs, a.lr, warmup_epochs=a.warmup,
                                  schedule="cosine", min_ratio=0.05)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        for samples in iterate_batches(tr_ds, a.batch, shuffle=True, seed=a.seed + ep):
            b = to_torch_batch(collate_samples(samples), device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
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
        rec = {"epoch": ep, "train_loss": tot / max(seen, 1)}
        if ep % 5 == 0 or ep == a.epochs:
            rows = decode_all(model, va_ds, device, a.batch, voc, ref_loose, ref_raw)
            wl = sum(r["edit_loose"] for r in rows) / max(sum(r["n_ref"] for r in rows), 1)
            ws = sum(r["edit_strict"] for r in rows) / max(sum(r["n_ref"] for r in rows), 1)
            nd = len({tuple(r["hyp"]) for r in rows})
            hl = float(np.mean([len(r["hyp"]) for r in rows]))
            rec.update({"wer_loose": round(wl, 4), "wer_strict": round(ws, 4),
                        "n_distinct": nd, "hyp_len_mean": round(hl, 2)})
            print("  ep {:2d} loss={:8.4f}  WER宽松={:.4f} 严格={:.4f}  "
                  "n_distinct={:4d}  len={:.2f}".format(
                      ep, rec["train_loss"], wl, ws, nd, hl), flush=True)
        hist.append(rec)
    dt = (time.time() - t0) / 60

    rows = decode_all(model, va_ds, device, a.batch, voc, ref_loose, ref_raw)
    N = sum(r["n_ref"] for r in rows)
    wer_loose = sum(r["edit_loose"] for r in rows) / max(N, 1)
    wer_strict = sum(r["edit_strict"] for r in rows) / max(N, 1)
    cer_like = wer_loose  # 词级下 CER 无意义，仅留字段
    seqs = {tuple(r["hyp"]) for r in rows}
    kinds = {t for r in rows for t in r["hyp"]}
    exact = sum(1 for r in rows if r["edit_strict"] == 0)
    partial = sum(1 for r in rows if 0 < r["hits"] < r["n_ref"])
    zero = sum(1 for r in rows if r["hits"] == 0)
    hit_tokens = sum(r["hits"] for r in rows)

    # 分桶
    in_v = [r for r in rows if all(t in vset for t in r["ref_strict"])]
    has_o = [r for r in rows if any(t not in vset for t in r["ref_strict"])]
    def bucket(rs):
        if not rs:
            return None
        n = sum(r["n_ref"] for r in rs)
        return {"n": len(rs), "wer_loose": round(sum(r["edit_loose"] for r in rs) / n, 4),
                "wer_strict": round(sum(r["edit_strict"] for r in rs) / n, 4),
                "token_recall": round(sum(r["hits"] for r in rs) / n, 4)}

    # 混淆
    conf = collections.Counter()
    for r in rows:
        for g in r["ref_strict"]:
            if g in r["hyp"]:
                conf[(g, g)] += 1
    wrong = collections.Counter()
    for r in rows:
        rh, rr = set(r["hyp"]), set(r["ref_strict"])
        for t in rr - rh:
            for h in rh:
                if h != t:
                    wrong[(t, h)] += 1

    print("")
    print("=" * 76)
    print("修复后 · 真实 dev {} 条 · 词表 {} 类".format(len(rows), voc.size))
    print("=" * 76)
    print("  WER 宽松（OOV 折叠）= {:.4f}".format(wer_loose))
    print("  WER 严格（保留 OOV） = {:.4f}".format(wer_strict))
    print("  不同输出序列种类     = {}  （修复前 6）".format(len(seqs)))
    print("  输出的 token 种类     = {}".format(len(kinds)))
    print("  输出长度均值         = {:.2f}  （参考 {:.2f}）".format(
        float(np.mean([len(r["hyp"]) for r in rows])), float(np.mean([r["n_ref"] for r in rows]))))
    print("  token 级召回         = {:.4f}".format(hit_tokens / max(N, 1)))
    print("  完全正确句 {}/{}   部分正确 {}   零命中 {}".format(
        exact, len(rows), partial, zero))
    print("")
    print("  分桶：")
    bv, bo = bucket(in_v), bucket(has_o)
    print("    词汇内样本   {}".format(bv))
    print("    含 OOV 样本  {}".format(bo))
    print("")
    print("  逐样本对照（前 {} 条，按参考长度排序取样）".format(a.n_show))
    show = sorted(rows, key=lambda r: -r["n_ref"])[: a.n_show]
    for r in show:
        print("    ref : {}".format("/".join(r["ref_strict"])))
        print("    hyp : {}".format("/".join(r["hyp"]) or "(空)"))
        print("           命中 {}/{}".format(r["hits"], r["n_ref"]))
    print("")
    print("  最常见混淆（ref -> 被误识为）Top10:")
    for (g, h), c in wrong.most_common(10):
        print("    {:<8s} -> {:<8s} {}".format(g, h, c))
    print("")
    print("  训练耗时 {:.1f} 分钟".format(dt))

    torch.save({"model_config": cfg.as_dict(), "state_dict": model.state_dict(),
                "vocabulary": list(voc.tokens),
                "vocabulary_config": voc.config.as_dict(),
                "note": "trained AFTER off-by-one fix in collate_samples"},
               REPO / "artifacts/checkpoints/ctc-fixed-cap300.pt")

    receipt = {
        "experiment": "P25 readable output after off-by-one fix",
        "bug_fixed": "collate_samples now sends token_id + 1 (CTC class space)",
        "all_data_real": True, "reads_test_split": False,
        "config": {"max_tokens": a.max_tokens, "min_frequency": a.min_freq,
                   "vocab_size": voc.size, "epochs": a.epochs, "seed": a.seed,
                   "batch": a.batch, "lr": a.lr},
        "metrics": {
            "wer_loose": round(wer_loose, 4), "wer_strict": round(wer_strict, 4),
            "n_distinct_outputs": len(seqs), "token_kinds": len(kinds),
            "hyp_len_mean": round(float(np.mean([len(r["hyp"]) for r in rows])), 2),
            "ref_len_mean": round(float(np.mean([r["n_ref"] for r in rows])), 2),
            "token_recall": round(hit_tokens / max(N, 1), 4),
            "n_exact": exact, "n_partial": partial, "n_zero": zero,
            "n_samples": len(rows),
        },
        "buckets": {"in_vocab": bv, "has_oov": bo},
        "top_confusions": [[list(k), v] for k, v in wrong.most_common(20)],
        "samples": [{k: v for k, v in r.items()} for r in show],
        "history": hist,
        "minutes": round(dt, 2),
        "comparison_before_fix": {"wer_loose": 0.9760, "n_distinct": 6, "hyp_len": 1.48},
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))
    print("checkpoint: artifacts/checkpoints/ctc-fixed-cap300.pt")


if __name__ == "__main__":
    main()
