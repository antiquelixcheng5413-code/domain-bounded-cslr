# -*- coding: utf-8 -*-
"""P28 · 训练时长 vs 泛化：dev WER 是否还在降？（决定「训练不足」还是「特征到顶」）

## 为什么要做这个

P27 随机 5 条实测：句末标点 5/5 全对，中段内容词几乎全错。
同时 P25/P26 测到 **train WER 0.4036 vs dev 0.7019**。

关键疑点：**train 也没到完美（0.40）**。
这通常有两种解释，处置方式完全相反：

- **(a) 训练不足**：dev WER 会随轮数继续下降 → 加轮数/加容量就有收益
- **(b) 特征到顶**：dev WER 早早 plateau，train 继续降而 dev 不动
  → 加训练只会加剧过拟合，**必须换特征/多模态**

## 本脚本测什么（真实数据，单因素）

固定词表 301、模型、lr、batch、seed，**唯一变量是训练轮数**：
30 / 60 / 100 / 150 轮，每档都测 train 与 dev 的 WER。

**同时报告两个差值**：
- `gap = dev_WER - train_WER`  → 泛化差距
- `n_distinct_outputs`（dev）  → 输出多样性是否随轮次持续上升

## 判据（跑之前写死）

在最后一档（150 轮）：

- 若 `dev_WER` 相比 30 轮**仍下降 ≥ 0.03** → **训练不足**，
  下一步加轮数/加模型容量
- 若 `dev_WER` 在 60 轮后**变化 < 0.01 且 gap 持续扩大**
  → **特征到顶**，加训练只会过拟合，必须换特征（HaMeR 多模态路线）
- 若 dev 降但 train 降更多 → 过拟合为主

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
from cslr.recognition.model import CTCConfig, CTCRecognizer, BLANK_INDEX
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
def evaluate(model, ds, device, batch, voc, ref_loose, subset=None):
    model.eval()
    d = n = 0
    seqs = []
    lens = []
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
            lens.append(len(hyp))
            ref = ref_loose[b["sample_ids"][r]]
            d += levenshtein(list(ref), hyp)[0]
            n += len(ref)
    return {
        "wer": round(d / max(n, 1), 4),
        "n_distinct": len(set(seqs)),
        "len_mean": round(float(np.mean(lens)), 3) if lens else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--checkpoints", default="30,60,100,150")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-eval-n", type=int, default=1500,
                    help="train 侧全量 4973 解码较慢，固定抽这么多条做同口径对比")
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p28-training-length.json")
    a = ap.parse_args()

    marks = sorted(int(x) for x in a.checkpoints.split(","))
    marks = [m for m in marks if m <= a.max_epochs] or [a.max_epochs]

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
    ref_tr = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in tr_recs}
    ref_va = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in va_recs}
    # train 侧固定子集（seed 固定，可复现）
    tr_subset = set(random.Random(12345).sample(sorted(ref_tr.keys()),
                                                 min(a.train_eval_n, len(ref_tr))))

    cfg = CTCConfig(input_size=368, vocabulary_size=voc.size, hidden_size=a.hidden,
                    num_layers=a.layers, dropout=a.dropout, bidirectional=True,
                    projection_size=256, subsample_stride=1)
    model = CTCRecognizer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print("device = {}".format(device))
    print("训练轮数档位 = {}   （唯一变量）".format(marks))
    print("train 侧同口径子集 = {} 条".format(len(tr_subset)))
    print("")
    print("{:<8s} {:>10s} {:>10s} {:>8s} {:>10s} {:>10s} {:>8s}".format(
        "epoch", "train_ls", "trainWER", "devWER", "gap", "dev_ndist", "dev_len"))

    results = []
    t0 = time.time()
    for ep in range(1, a.max_epochs + 1):
        model.train()
        tot = seen = 0
        # cosine 调度按总轮数走，保证 30 档与 150 档不可比时也能看出趋势
        cur_lr = learning_rate_at(ep, a.max_epochs, a.lr, warmup_epochs=a.warmup,
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

        if ep in marks or ep % 10 == 0:
            tr = evaluate(model, tr_ds, device, a.batch, voc, ref_tr, subset=tr_subset)
            dv = evaluate(model, va_ds, device, a.batch, voc, ref_va)
            rec = {"epoch": ep, "train_loss": round(tot / max(seen, 1), 4),
                   "train_wer": tr["wer"], "dev_wer": dv["wer"],
                   "gap": round(dv["wer"] - tr["wer"], 4),
                   "dev_n_distinct": dv["n_distinct"], "dev_len": dv["len_mean"]}
            results.append(rec)
            star = " <<<" if ep in marks else ""
            print("{:<8d} {:>10.4f} {:>10.4f} {:>8.4f} {:>10.4f} {:>10d} {:>8.2f}{}".format(
                ep, rec["train_loss"], rec["train_wer"], rec["dev_wer"],
                rec["gap"], rec["dev_n_distinct"], rec["dev_len"], star), flush=True)

    dt = (time.time() - t0) / 60
    key = [r for r in results if r["epoch"] in marks]
    first, last = key[0], key[-1]
    d_dev = last["dev_wer"] - first["dev_wer"]
    d_train = last["train_wer"] - first["train_wer"]

    print("")
    print("=" * 74)
    print("判决（判据跑前已写死）")
    print("=" * 74)
    print("  {} 轮 -> {} 轮：dev WER {:+.4f}，train WER {:+.4f}，gap {:+.4f}".format(
        first["epoch"], last["epoch"], d_dev, d_train, last["gap"] - first["gap"]))
    if d_dev <= -0.03:
        verdict = ("训练不足：dev WER 仍在明显下降（{:+.4f}）-> "
                   "加轮数/加模型容量有收益".format(d_dev))
    elif abs(key[-2]["dev_wer"] - last["dev_wer"]) < 0.01 and last["gap"] > first["gap"] + 0.05:
        verdict = ("特征到顶：60 轮后 dev WER 已 plateau（变化 {:+.4f}），"
                   "而 gap 持续扩大（{:+.4f}）-> 加训练只加剧过拟合，"
                   "必须换特征（HaMeR 多模态）".format(
                       last["dev_wer"] - key[-2]["dev_wer"], last["gap"] - first["gap"]))
    else:
        verdict = "中间态，需看曲线形状判断"
    print("  " + verdict)
    print("  训练耗时 {:.1f} 分钟".format(dt))

    receipt = {
        "experiment": "P28 training length vs generalisation",
        "single_factor": "number of epochs only; vocab/model/lr/batch/seed identical",
        "criterion": {
            "undertrained": "dev_WER drops >= 0.03 from first to last mark",
            "feature_capped": "dev plateau (<0.01 change) while gap keeps widening"},
        "train_eval_subset": {"n": len(tr_subset), "seed": 12345},
        "all_data_real": True, "reads_test_split": False,
        "marks": marks,
        "history": results,
        "verdict": verdict,
        "minutes": round(dt, 2),
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
