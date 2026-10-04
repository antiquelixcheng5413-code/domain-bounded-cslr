# -*- coding: utf-8 -*-
"""P27 · 随机抽 5 个 dev 视频，实测识别效果（交付给用户直接看）

## 目的

用户要看「随机抽 5 个视频的识别效果」。
本脚本在**修复 off-by-one 之后**的模型上：
1. 用 train 4973 训练 30 epoch（词表 301，与 P25/P26 同配置）
2. 从 dev 514 里**随机抽 5 条**（seed 固定，可复现）
3. 逐条打印 参考 / 识别 / 命中情况
4. 附上全 dev 指标作对照，避免只看 5 条以偏概全

## 抽样规则（预先声明，防挑样本）

- `random.Random(20261004).sample(...)`，**先抽再排序输出**，不按结果挑选
- seed 写死在代码里，任何人重跑得到同一批样本
- 同时报告这 5 条的 WER 与全 dev WER 的对比

## 口径

- `ref`：CSV 原始 gloss（保留真实 OOV，不折叠）—— 真实水平
- `hyp`：模型识别结果
- `hits`：按顺序对齐后的正确词数（len(ref) - 编辑距离）

只读 train/dev，**不触碰 test split**。
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
from cslr.recognition.model import CTCConfig, CTCRecognizer
from cslr.recognition.gloss_sequence import build_ordered_vocabulary
from cslr.recognition.dataset import GlossSequenceDataset, FeatureNormalizer, collate_samples
from cslr.recognition.training import (
    ctc_loss, iterate_batches, learning_rate_at, resolve_device,
    to_torch_batch, decode_batch)
from p8_error_attribution import levenshtein

SAMPLE_SEED = 20261004
N_SHOW = 5


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
def decode_all(model, ds, device, batch, voc, ref_raw, ref_loose):
    model.eval()
    out = {}
    for samples in iterate_batches(ds, batch, shuffle=False, seed=42):
        b = to_torch_batch(collate_samples(samples), device)
        logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        dec, _, _ = decode_batch(lp, ol.cpu().numpy(), 1)
        for r, ids in enumerate(dec):
            sid = b["sample_ids"][r]
            hyp = voc.decode(list(ids))
            rs = ref_raw[sid]
            rl = ref_loose[sid]
            out[sid] = {
                "sid": sid, "ref": rs, "hyp": hyp,
                "ref_loose": rl,
                "edit_strict": levenshtein(list(rs), hyp)[0],
                "edit_loose": levenshtein(list(rl), hyp)[0],
                "n_ref": len(rs),
                "hits": len(rs) - levenshtein(list(rs), hyp)[0],
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p27-five-samples.json")
    a = ap.parse_args()

    device = resolve_device("auto")
    tr_labels = read_split("train")
    va_labels = read_split("validation")
    tr_root = ensure_link_dir("train")
    va_root = ensure_link_dir("validation")
    tr_recs = make_records(tr_labels, tr_root, "train")
    va_recs = make_records(va_labels, va_root, "validation")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    voc, _ = build_ordered_vocabulary((g for g in tr_labels.values()),
                                      min_frequency=2, max_tokens=300)
    nrm = FeatureNormalizer.fit(
        [np.load(tr_root / (r.sample_id + ".npy")) for r in tr_recs[:600]])
    tr_ds = GlossSequenceDataset(tr_recs, tr_root, voc, nrm, feature_view="full")
    va_ds = GlossSequenceDataset(va_recs, va_root, voc, nrm, feature_view="full")
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

    print("训练 {} 轮（修复后模型，词表 301）...".format(a.epochs))
    t0 = time.time()
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
        if ep % 10 == 0 or ep == a.epochs:
            print("  ep {:2d} loss={:.4f}".format(ep, tot / max(seen, 1)))
    print("训练完成 {:.1f} 分钟".format((time.time() - t0) / 60))

    res = decode_all(model, va_ds, device, a.batch, voc, ref_raw, ref_loose)
    N = sum(v["n_ref"] for v in res.values())
    wer_s = sum(v["edit_strict"] for v in res.values()) / N
    wer_l = sum(v["edit_loose"] for v in res.values()) / N

    # ---- 随机抽 5 条（先抽，不看结果）----
    pool = sorted(res.keys())
    picked = random.Random(SAMPLE_SEED).sample(pool, N_SHOW)
    picked.sort()

    print()
    print("=" * 78)
    print("随机抽 {} 条 dev 样本（seed={}，抽样与结果无关）".format(N_SHOW, SAMPLE_SEED))
    print("=" * 78)
    for i, sid in enumerate(picked, 1):
        v = res[sid]
        print()
        print("【样本 {}】{}".format(i, sid))
        print("  参考（人工标注）: {}".format(" / ".join(v["ref"])))
        print("  模型识别        : {}".format(" / ".join(v["hyp"]) or "(空)"))
        print("  命中 {}/{} 词   编辑距离 {}".format(
            v["hits"], v["n_ref"], v["edit_strict"]))
        # 标出命中的词
        hs = set(v["hyp"])
        mark = " ".join(("✓" if t in hs else "✗") + t for t in v["ref"])
        print("  逐词核对        : {}".format(mark))

    sub_n = sum(res[s]["n_ref"] for s in picked)
    sub_s = sum(res[s]["edit_strict"] for s in picked) / max(sub_n, 1)
    sub_l = sum(res[s]["edit_loose"] for s in picked) / max(sub_n, 1)
    print()
    print("=" * 78)
    print("这 {} 条 vs 全 dev 514 条".format(N_SHOW))
    print("=" * 78)
    print("  {:<12s} {:>10s} {:>10s} {:>12s}".format("", "严格WER", "宽松WER", "平均输出词数"))
    print("  {:<12s} {:>10.4f} {:>10.4f} {:>12.2f}".format(
        "这 5 条", sub_s, sub_l,
        sum(len(res[s]["hyp"]) for s in picked) / N_SHOW))
    print("  {:<12s} {:.4f}   {:.4f}   {:.2f}".format(
        "全 dev 514 条", wer_s, wer_l,
        sum(len(v["hyp"]) for v in res.values()) / len(res)))

    receipt = {
        "experiment": "P27 five random dev samples",
        "sample_seed": SAMPLE_SEED, "n_shown": N_SHOW,
        "sampling": "random.Random(20261004).sample(sorted(dev_ids), 5) -- "
                    "drawn before inspecting any result",
        "config": {"vocab": 301, "epochs": a.epochs, "train_seed": a.seed,
                   "batch": a.batch, "lr": a.lr},
        "five_samples": [{k: res[s][k] for k in ("sid", "ref", "hyp", "n_ref",
                                                 "hits", "edit_strict")}
                         for s in picked],
        "subset_metrics": {"wer_strict": round(sub_s, 4), "wer_loose": round(sub_l, 4)},
        "full_dev_metrics": {"wer_strict": round(wer_s, 4), "wer_loose": round(wer_l, 4),
                             "n": len(res)},
        "all_data_real": True, "reads_test_split": False,
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
