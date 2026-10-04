# -*- coding: utf-8 -*-
"""P30 · 容量控制实验：验证「泛化受限是否因模型容量过剩」

## 前提事实（全部实测）

```
P29 结论：
  flat 基线 train WER 0.0065 / dev WER 0.6938     -> 能拟合，泛化差
  skel_add（22.24M）train 0.0000 / dev 0.7311      -> 容量更大，dev 更差

P30 新测的 dev 词频结构（cap300 词表，dev 514 条 / 2842 token）：
  训练频次 100+ 的词占 dev token  48.1%   <- 近一半是高频词，模型应该能学会
  训练频次 11-100                23.1%
  训练频次 3-10                  16.0%
  训练频次 1-2                    6.7%
  训练完全未见（OOV）              6.2%   （按 token 计；按句计 91% 句子含 OOV）
  训练侧目标 token 里 <unk> 占 28.3%（P17 实测）
```

**关键推论**：dev 有近一半 token 是训练高频词，而这些词模型**必然见过很多次**。
若模型连这些都认不出，就不是「特征表达力不足」，
而是「容量过剩导致把训练集背下来、放弃了对共现结构的泛化」。

## 假设与判据

**H：dev WER 随模型容量减小而下降。**

单因素：只变模型容量与正则强度，其余全同
（词表 301、特征不变、batch 16、lr 1e-3、seed 42、30 epoch）。

| 配置 | hidden | layers | dropout | wd | 参数量 |
|---|---|---|---|---|---|
| `base` | 256 | 2 | 0.3 | 1e-4 | 2.88M |
| `small` | 128 | 2 | 0.3 | 1e-4 | ~0.9M |
| `tiny` | 64 | 1 | 0.5 | 1e-3 | ~0.2M |
| `heavy_reg` | 128 | 2 | 0.5 | 1e-3 | ~0.9M |

**判据（跑之前写死）**：
- 若 `tiny` 或 `small` 的 dev WER 比 `base` 低 **≥0.02** → 容量过剩确认
- 若三者 dev WER 差异 <0.01 → 容量不是瓶颈，需回到特征/数据层
- **train/dev gap 同步报告**：容量下降应让 gap 缩小，若 gap 不变则说明
  过拟合源不在容量

## 附带：按 dev 词频分桶报告 WER

这能直接回答「高频词到底认不认得」：
若 100+ 桶的 WER 也很高 → 模型连高频词都没学会，是训练问题
若 100+ 桶 WER 低但低频桶高 → 是长尾泛化问题

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

CONFIGS = {
    "base":       dict(hidden=256, layers=2, dropout=0.3, wd=1e-4),
    "small":      dict(hidden=128, layers=2, dropout=0.3, wd=1e-4),
    "tiny":       dict(hidden=64,  layers=1, dropout=0.5, wd=1e-3),
    "heavy_reg":  dict(hidden=128, layers=2, dropout=0.5, wd=1e-3),
}


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
def evaluate(model, ds, device, batch, voc, ref, subset=None, freq=None):
    """返回总体 WER + 按训练频次分桶的 WER + 逐句明细。"""
    model.eval()
    d = n = 0
    seqs = []
    lens = []
    per = []
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
            sid = b["sample_ids"][r]
            R = ref[sid]
            e = levenshtein(list(R), hyp)[0]
            d += e
            n += len(R)
            seqs.append(tuple(hyp))
            lens.append(len(hyp))
            per.append({"sid": sid, "ref": R, "hyp": hyp, "edit": e, "n_ref": len(R)})
    res = {"wer": round(d / max(n, 1), 4), "n_distinct": len(set(seqs)),
           "len_mean": round(float(np.mean(lens)), 3) if lens else 0.0}
    if freq is not None:
        buckets = collections.defaultdict(lambda: [0, 0])
        for row in per:
            for t in row["ref"]:
                f = freq.get(t, 0)
                b = ("OOV" if f == 0 else "1-2" if f < 3 else "3-10" if f < 11
                     else "11-100" if f < 101 else "100+")
                buckets[b][0] += 1
                buckets[b][1] += 0 if t in row["hyp"] else 1
        res["by_freq"] = {k: {"n_tokens": v[0], "wer": round(v[1] / max(v[0], 1), 4)}
                          for k, v in sorted(buckets.items())}
    return res, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-eval-n", type=int, default=1500)
    ap.add_argument("--configs", default="base,small,tiny,heavy_reg")
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p30-capacity.json")
    a = ap.parse_args()

    device = resolve_device("auto")
    tr_labels = read_split("train")
    va_labels = read_split("validation")
    tr_root = ensure_link_dir("train")
    va_root = ensure_link_dir("validation")
    tr_recs = make_records(tr_labels, tr_root, "train")
    va_recs = make_records(va_labels, va_root, "validation")

    voc, counts = build_ordered_vocabulary((g for g in tr_labels.values()),
                                           min_frequency=2, max_tokens=300)
    nrm = FeatureNormalizer.fit(
        [np.load(tr_root / (r.sample_id + ".npy")) for r in tr_recs[:600]])
    tr_ds = GlossSequenceDataset(tr_recs, tr_root, voc, nrm, feature_view="full")
    va_ds = GlossSequenceDataset(va_recs, va_root, voc, nrm, feature_view="full")
    # 折叠 ref（与训练 target 同源）
    ref_tr = {r.sample_id: [voc.tokens[i] for i in voc.encode(r.label)] for r in tr_recs}
    ref_va = {r.sample_id: [voc.tokens[i] for i in voc.encode(r.label)] for r in va_recs}
    tr_subset = set(random.Random(12345).sample(sorted(ref_tr.keys()),
                                               min(a.train_eval_n, len(ref_tr))))
    print("device = {}  train {} / dev {}  vocab {}".format(
        device, len(tr_recs), len(va_recs), voc.size))

    results = []
    for name in a.configs.split(","):
        spec = CONFIGS[name]
        print()
        print("#" * 74)
        print("# 配置 {}  {}".format(name, spec))
        print("#" * 74)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        cfg = CTCConfig(input_size=368, vocabulary_size=voc.size,
                        hidden_size=spec["hidden"], num_layers=spec["layers"],
                        dropout=spec["dropout"], bidirectional=True,
                        projection_size=256, subsample_stride=1)
        model = CTCRecognizer(cfg).to(device)
        nparam = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=spec["wd"])
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        print("  参数量 {:.3f}M".format(nparam / 1e6))
        hist = []
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
                tr, _ = evaluate(model, tr_ds, device, a.batch, voc, ref_tr, subset=tr_subset)
                dv, _ = evaluate(model, va_ds, device, a.batch, voc, ref_va)
                hist.append({"epoch": ep, "train_loss": round(tot / max(seen, 1), 4),
                             "train_wer": tr["wer"], "dev_wer": dv["wer"],
                             "gap": round(dv["wer"] - tr["wer"], 4),
                             "dev_n_distinct": dv["n_distinct"],
                             "dev_len": dv["len_mean"]})
                print("  ep {:3d} loss={:8.4f} trainWER={:.4f} devWER={:.4f} "
                      "gap={:.4f} ndist={:4d} len={:.2f}".format(
                          ep, tot / max(seen, 1), tr["wer"], dv["wer"],
                          dv["wer"] - tr["wer"], dv["n_distinct"], dv["len_mean"]), flush=True)
        dv, per = evaluate(model, va_ds, device, a.batch, voc, ref_va, freq=counts)
        tr, _ = evaluate(model, tr_ds, device, a.batch, voc, ref_tr, subset=tr_subset)
        dt = (time.time() - t0) / 60
        results.append({"config": name, **spec, "params_M": round(nparam / 1e6, 4),
                        "final_train_wer": tr["wer"], "final_dev_wer": dv["wer"],
                        "final_gap": round(dv["wer"] - tr["wer"], 4),
                        "dev_by_freq": dv["by_freq"], "history": hist,
                        "minutes": round(dt, 2), "samples": per[:5]})
        print("  => trainWER={:.4f} devWER={:.4f} gap={:.4f}".format(
            tr["wer"], dv["wer"], dv["wer"] - tr["wer"]))
        print("  按训练频次分桶的 dev WER:")
        for b, v in dv["by_freq"].items():
            print("     {:<7s} n={:5d}  WER={:.4f}".format(b, v["n_tokens"], v["wer"]))

    print()
    print("=" * 78)
    print("P30 汇总（判据：更小配置 devWER 比 base 低 >=0.02）")
    print("=" * 78)
    print("{:<11s} {:>9s} {:>10s} {:>9s} {:>8s} {:>9s}".format(
        "配置", "参数M", "trainWER", "devWER", "gap", "100+桶WER"))
    for r in results:
        f100 = r["dev_by_freq"].get("100+", {}).get("wer", float("nan"))
        print("{:<11s} {:>9.3f} {:>10.4f} {:>9.4f} {:>8.4f} {:>9.4f}".format(
            r["config"], r["params_M"], r["final_train_wer"], r["final_dev_wer"],
            r["final_gap"], f100))

    base = next((r for r in results if r["config"] == "base"), None)
    others = [r for r in results if r["config"] != "base"]
    if base and others:
        best = min(others, key=lambda r: r["final_dev_wer"])
        dd = best["final_dev_wer"] - base["final_dev_wer"]
        if dd <= -0.02:
            verdict = ("容量过剩确认：{} 的 devWER {:+.4f} 优于 base，"
                       "gap 从 {:.4f} 降到 {:.4f}".format(
                           best["config"], dd, base["final_gap"], best["final_gap"]))
        else:
            verdict = ("容量不是瓶颈：最好的非 base 配置 {:+.4f}（阈值 -0.02），"
                       "容量方向无效".format(dd))
        print()
        print("判决: " + verdict)
    else:
        verdict = "缺少 base 对照"

    receipt = {
        "experiment": "P30 capacity control - is overfitting caused by excess capacity?",
        "single_factor": "hidden/layers/dropout/weight_decay only; features, vocab, seed identical",
        "criterion": "smaller config dev_WER <= base - 0.02",
        "dev_freq_structure": {
            "100+": 0.481, "11-100": 0.231, "3-10": 0.160,
            "1-2": 0.067, "OOV": 0.062,
            "note": "share of dev tokens by training frequency; nearly half are high-frequency",
        },
        "reference_used": "folded tokens (same space as the training target)",
        "all_data_real": True, "reads_test_split": False,
        "results": results, "verdict": verdict,
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
