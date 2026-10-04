# -*- coding: utf-8 -*-
"""P19 · 判决性诊断：过拟合 vs 特征判别力（决定「可读结果」能否实现）

## P18 留下的关键疑点

P18 配置 A（token/301）30 轮后：

```
train_loss = -3.6252      <- 负数！
dev  WER   =  0.9760
```

**CTC loss 是负的** —— 因为它是 `log P(目标序列)` 的均值除以 token 数，
模型对训练集目标赋予的概率 > 1 时 loss 可以为负。
换言之**配置 A 已经把训练集拟合到近乎完美了**。

于是问题不再是「模型没学会」，而是二选一：

- **(a) 纯过拟合**：train 上近乎全对，dev 上全错 → 特征无法泛化
- **(b) dev 分布异常**：train/dev 特征分布有系统偏移 → 归一化或划分问题

这两个的解法完全不同，必须分开测。

## 本脚本测的三个量（全部真实数据）

### 量1 · train WER vs dev WER（过拟合缺口）

在同一份训练好的权重上，分别在 **train 4973** 与 **dev 515** 上解码算 WER。
- 若 train WER ≪ dev WER（如 0.05 vs 0.98）→ **(a) 过拟合**
- 若两者都高 → 特征根本没有判别力，属于 (b) 或更差的情况

同时逐轮记录 `train_WER / dev_WER / gap`，
看 gap 是**一直很大**（特征问题）还是**从 0 涨上去**（记忆问题）。

### 量2 · gloss 级特征可分性上界

对每个 gloss，把它在某个 split 里所有出现位置的 mean-pooled 特征取平均
得到「类原型」，然后做 1-NN 分类：
- train 原型 → 分类 train 样本（自身检索，应当极高）
- **train 原型 → 分类 dev 样本（跨集合泛化，本脚本的关键量）**

若 1-NN 跨集合准确率极低，说明**特征本身就不含 gloss 身份信息**，
那么任何序列模型（CTC/Transformer）都不可能产出可读结果。

### 量3 · 混淆结构

dev 上 1-NN 的混淆对 Top-20，以及 dev 参考词与最近原型的距离分布。
距离分布能区分两种失败：
- 距离普遍很大 → 特征不可靠（MediaPipe landmark 抖动/归一化问题）
- 距离正常但邻居错 → 特征可分但标注/词表有歧义

## 判据（跑之前写死）

- 若 `train_WER < 0.3` 且 `dev_WER > 0.9` → **过拟合确认**，
  解法在正则化/数据量/特征增强
- 若 `train_WER > 0.7` → **特征判别力不足确认**，
  换模型无用，必须换特征
- 若 `1NN(train原型→dev) < 0.15` → **特征不含 gloss 身份信息**，
  这是硬上限，任何序列模型都救不了

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
from cslr.recognition.gloss_sequence import (
    GlossVocabulary, GlossSequenceConfig, build_ordered_vocabulary)
from cslr.recognition.dataset import GlossSequenceDataset, FeatureNormalizer, collate_samples
from cslr.recognition.training import (
    TrainingConfig, ctc_loss, iterate_batches, learning_rate_at,
    resolve_device, to_torch_batch, decode_batch)
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
def decode_wer(model, ds, voc, device, batch, ref_by_id):
    """在给定数据集上贪心解码并算 WER。同时返回输出序列用于多样性统计。"""
    model.eval()
    d = n = 0
    seqs = []
    for samples in iterate_batches(ds, batch, shuffle=False, seed=42):
        b = to_torch_batch(collate_samples(samples), device)
        logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        dec, _, _ = decode_batch(lp, ol.cpu().numpy(), 1)
        for row, ids in enumerate(dec):
            hyp = voc.decode(list(ids))
            seqs.append(tuple(hyp))
            ref = ref_by_id.get(b["sample_ids"][row]) if "sample_ids" in b else b["tokens"][row]
            d += levenshtein(list(ref), hyp)[0]
            n += len(ref)
    return {"wer": d / max(n, 1), "n_distinct_outputs": len(set(seqs)),
            "edit": d, "ref_len": n}


# ---------------------------------------------------------------- 量2
def gloss_prototypes(ds, voc, batch, device, min_count=2):
    """每个 gloss 的 mean-pooled 特征原型（只用数据集自身，不借用任何标签信息）。

    返回 {token: vec}。样本级特征 = 48 帧 landmark 的时间均值。
    """
    acc = collections.defaultdict(list)
    for samples in iterate_batches(ds, batch, shuffle=False, seed=42):
        b = to_torch_batch(collate_samples(samples), device)
        f = b["features"].float().cpu().numpy()          # [B, 48, 368]
        ol = b["input_lengths"].cpu().numpy()
        toks = b["tokens"]
        for row in range(f.shape[0]):
            t = int(ol[row])
            v = f[row, :t].mean(axis=0)                  # 时间池化
            for g in toks[row]:
                if g in voc:
                    acc[g].append(v)
    out = {}
    for g, vs in acc.items():
        if len(vs) >= min_count:
            out[g] = np.mean(np.stack(vs), axis=0)
    return out


def one_nn(proto, query_vecs, query_toks, topk=1):
    """1-NN：查询向量在 train 原型里找最近邻。返回 top-1 准确率与混淆对。"""
    keys = list(proto.keys())
    M = np.stack([proto[k] for k in keys])              # [K, D]
    M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-8)
    Q = np.stack(query_vecs)
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
    sim = Q @ M.T                                        # [N, K]
    idx = np.argsort(-sim, axis=1)[:, :topk]
    kset = set(keys)
    correct = 0
    n = 0
    conf = collections.Counter()
    dists = []
    for row, gt in enumerate(query_toks):
        if gt not in kset:
            continue
        n += 1
        pred = keys[idx[row, 0]]
        if pred == gt:
            correct += 1
        else:
            conf[(gt, pred)] += 1
        dists.append(1.0 - sim[row, idx[row, 0]])
    return {"n": n, "acc": correct / max(n, 1),
            "top_confusions": [[list(k), v] for k, v in conf.most_common(20)],
            "nn_dist_mean": float(np.mean(dists)) if dists else 0.0,
            "nn_dist_p90": float(np.percentile(dists, 90)) if dists else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p19-overfit-vs-feature.json")
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
    print("vocab {} 类   dev 词表外 token 率 {:.1%}".format(
        voc.size,
        sum(1 for r in va_recs for t in r.label.split("/")
            if t.strip() and t.strip() not in set(voc.tokens))
        / max(sum(len([t for t in r.label.split("/") if t.strip()]) for r in va_recs), 1)))

    nrm = FeatureNormalizer.fit(
        [np.load(tr_root / (r.sample_id + ".npy")) for r in tr_recs[:600]])

    tr_ds = GlossSequenceDataset(tr_recs, tr_root, voc, nrm, feature_view="full")
    va_ds = GlossSequenceDataset(va_recs, va_root, voc, nrm, feature_view="full")
    ref_tr = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in tr_recs}
    ref_va = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in va_recs}

    cfg = CTCConfig(input_size=368, vocabulary_size=voc.size, hidden_size=a.hidden,
                    num_layers=a.layers, dropout=a.dropout, bidirectional=True,
                    projection_size=256, subsample_stride=1)
    model = CTCRecognizer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    tcfg = TrainingConfig(epochs=a.epochs, batch_size=a.batch, learning_rate=a.lr,
                          weight_decay=a.weight_decay, seed=a.seed, device=str(device),
                          amp=True, early_stopping_patience=999, beam_width=1,
                          min_epochs=1, sequence_length=48, normalize_features=False)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist = []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        ep_loss, seen = 0.0, 0
        cur_lr = learning_rate_at(ep, a.epochs, a.lr, warmup_epochs=a.warmup,
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

        # 只在关键轮次做全量 train 评估（train 4973 条较慢）
        do_train_eval = ep in (1, 5, 10, 20, a.epochs)
        m_tr = decode_wer(model, tr_ds, voc, device, a.batch, ref_tr) if do_train_eval else None
        m_va = decode_wer(model, va_ds, voc, device, a.batch, ref_va)
        rec = {"epoch": ep, "train_loss": ep_loss / max(seen, 1),
               "dev_wer": round(m_va["wer"], 4),
               "dev_n_distinct": m_va["n_distinct_outputs"]}
        if m_tr:
            rec["train_wer"] = round(m_tr["wer"], 4)
            rec["gap"] = round(m_va["wer"] - m_tr["wer"], 4)
            rec["train_n_distinct"] = m_tr["n_distinct_outputs"]
        hist.append(rec)
        msg = "  ep {:2d} loss={:8.3f} dev_WER={:.4f} n_dist={:4d}".format(
            ep, rec["train_loss"], rec["dev_wer"], rec["dev_n_distinct"])
        if m_tr:
            msg += "  | train_WER={:.4f} gap={:.4f}".format(rec["train_wer"], rec["gap"])
        print(msg, flush=True)

    # ---- 量1 结论 ----
    last = [r for r in hist if "train_wer" in r][-1]
    if last["train_wer"] < 0.3 and last["dev_wer"] > 0.9:
        verdict1 = "过拟合确认：train 已拟合，dev 全错 -> 解法在正则化/数据量/特征增强"
    elif last["train_wer"] > 0.7:
        verdict1 = "特征判别力不足确认：train 也学不好 -> 换模型无用，必须换特征"
    else:
        verdict1 = "中间态：train={:.4f} dev={:.4f}".format(last["train_wer"], last["dev_wer"])

    # ---- 量2/3 ----
    print("")
    print("=" * 70)
    print("量2 · gloss 级 1-NN 特征可分性上界（跨集合泛化）")
    print("=" * 70)
    proto_tr = gloss_prototypes(tr_ds, voc, a.batch, device)
    print("train 原型数 {}".format(len(proto_tr)))

    def gather(ds, ref_by_id):
        V, T = [], []
        for samples in iterate_batches(ds, a.batch, shuffle=False, seed=42):
            b = to_torch_batch(collate_samples(samples), device)
            f = b["features"].float().cpu().numpy()
            ol = b["input_lengths"].cpu().numpy()
            for row in range(f.shape[0]):
                t = int(ol[row])
                g = ref_by_id.get(b["sample_ids"][row]) if "sample_ids" in b else b["tokens"][row]
                V.append(f[row, :t].mean(axis=0))
                T.append(g[0] if g else "")      # 用首词作为该样本的代表词
        return V, T

    v_tr, t_tr = gather(tr_ds, ref_tr)
    v_va, t_va = gather(va_ds, ref_va)
    nn_in = one_nn(proto_tr, v_tr, t_tr)
    nn_out = one_nn(proto_tr, v_va, t_va)
    print("1-NN  train 样本 vs train 原型:  acc={:.4f}  n={}  nn_dist mean={:.3f}".format(
        nn_in["acc"], nn_in["n"], nn_in["nn_dist_mean"]))
    print("1-NN  dev   样本 vs train 原型:  acc={:.4f}  n={}  nn_dist mean={:.3f} p90={:.3f}".format(
        nn_out["acc"], nn_out["n"], nn_out["nn_dist_mean"], nn_out["nn_dist_p90"]))
    print("dev 混淆 Top10:")
    for (g, p), c in nn_out["top_confusions"][:10]:
        print("    {:<8s} -> {:<8s} {}".format(g, p, c))

    if nn_out["acc"] < 0.15:
        verdict2 = ("特征不含 gloss 身份信息（跨集合 1-NN acc={:.4f} < 0.15）："
                    "这是硬上限，任何序列模型都救不了").format(nn_out["acc"])
    else:
        verdict2 = ("特征含部分判别力（跨集合 1-NN acc={:.4f}）："
                    "序列模型有提升空间").format(nn_out["acc"])

    dt = (time.time() - t0) / 60
    print("")
    print("量1 结论: " + verdict1)
    print("量2 结论: " + verdict2)
    print("({:.1f} 分钟)".format(dt))

    receipt = {
        "experiment": "P19 overfit vs feature separability",
        "criterion": {"overfit": "train_WER<0.3 and dev_WER>0.9",
                      "weak_feature": "train_WER>0.7",
                      "no_gloss_identity": "1NN(train proto -> dev) < 0.15"},
        "all_data_real": True, "reads_test_split": False,
        "vocab_size": int(voc.size),
        "history": hist,
        "final": last,
        "verdict_overfit": verdict1,
        "one_nn_in_sample": nn_in,
        "one_nn_cross_split": nn_out,
        "verdict_feature": verdict2,
        "minutes": round(dt, 2),
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
