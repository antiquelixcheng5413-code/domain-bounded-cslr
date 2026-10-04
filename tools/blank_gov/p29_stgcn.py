# -*- coding: utf-8 -*-
"""P29 · ST-GCN 骨架分支：把 landmark 拓扑补回来（单因素消融）

## 动机与文献依据

P28 已确诊「特征到顶」：train WER 卡在 0.3226 不降，dev WER 30→150 轮只动 0.0063。

**ref11 ST-GCN（Yan et al., AAAI 2018）Sec 1 明确批评了我们这种做法**：

> "Earlier methods of using skeletons for action recognition simply employ the joint
> coordinates at individual time steps to form feature vectors, and apply temporal
> analysis thereon. The capability of these methods is limited as they **do not
> explicitly exploit the spatial relationships among the joints**, which are crucial
> for understanding human actions."

**ref23 SignFormer-GCN（Arib et al., PLOS ONE 2025）**：
- 骨架 X ∈ R^{T×N×Cm}，邻接矩阵 A 按骨骼连通性定义
- STGCN block → STGCN-LSTM 分支，与 I3D/RGB 分支相加（式 7）

我们把 `[0:126]`（双手 21 关节 × 3 坐标）当 126 个无序通道喂给 LSTM，
**丢掉了手骨连接拓扑**。本实验补上这一信息。

## 单因素设计

**唯一变量：是否启用 skeleton 分支。** 其余全部固定：
词表 301、hidden 256、layers 2、dropout 0.3、batch 16、lr 1e-3、
seed 42、训练 60 epoch、特征文件不改、不重提特征。

| 配置 | 说明 |
|---|---|
| `flat` | 现有 BiLSTM（等同 P28 基线，作为对照） |
| `skel_add` | 双流相加（对应 ref23 式 7） |
| `skel_only` | 只用骨架分支（验证拓扑本身的信息量） |

## 判据（跑之前写死）

- `skel_add` 的 **train WER 显著低于 0.3226**（≤0.28）
  → 拓扑信息确实补上了，P28 的天花板是结构性的
- `skel_add` 的 dev WER 低于 flat 基线 ≥0.02
  → 有泛化收益
- 若两者都不满足 → 拓扑不是瓶颈，特征到顶另有原因

**必须同时报告 train/dev**，否则无法区分「过拟合」与「真收益」。

## 双侧自检（沿用 P20 铁律）

先跑「8 条样本过拟合 + argmax 解码正确」，
确认新模型没有把 off-by-one 之类的 bug 引入。

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

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.contracts import SampleRecord
from cslr.recognition.gloss_sequence import build_ordered_vocabulary
from cslr.recognition.dataset import GlossSequenceDataset, FeatureNormalizer, collate_samples
from cslr.recognition.training import (
    ctc_loss, iterate_batches, learning_rate_at, resolve_device,
    to_torch_batch, decode_batch)
from p8_error_attribution import levenshtein
from stgcn import DualStreamCTCRecognizer


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
    return {"wer": round(d / max(n, 1), 4), "n_distinct": len(set(seqs)),
            "len_mean": round(float(np.mean(lens)), 3) if lens else 0.0}


def sanity_check(voc, tr_root, nrm, device, a):
    """P20 铁律：新模型必须能完美拟合 8 条样本，且 argmax 解码正确。

    **参考必须用与训练目标同一套 token**（`voc.tokens[voc.encode(label)]`），
    不能用 CSV 原始 gloss —— 词表外词在训练目标是 `<unk>`，而 CSV 里还是原词，
    两者永远对不上，WER 会卡在 0.34 左右（实测误报过一次）。
    """
    print("=" * 74)
    print("自检：8 条样本过拟合（确认新模型无新 bug）")
    print("=" * 74)
    tr_labels = read_split("train")
    recs = make_records(tr_labels, tr_root, "train")
    cand = [r for r in recs if 3 <= len([t for t in r.label.split("/") if t.strip()]) <= 6][:8]
    ds = GlossSequenceDataset(cand, tr_root, voc, nrm, feature_view="full")
    # 与训练目标一致：折叠后的 token 序列
    ref = {r.sample_id: [voc.tokens[i] for i in voc.encode(r.label)] for r in cand}
    for use_skel in (False, True):
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        model = DualStreamCTCRecognizer(368, voc.size, dropout=0.0,
                                       use_skeleton=use_skel).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
        for ep in range(200):
            model.train()
            for s in iterate_batches(ds, 8, shuffle=False, seed=0):
                b = to_torch_batch(collate_samples(s), device)
                opt.zero_grad(set_to_none=True)
                lg = model(b["features"], b["input_lengths"])
                ol = model.output_lengths(b["input_lengths"])
                loss = ctc_loss(lg, b["input_lengths"], b["targets"],
                                b["target_lengths"], ol)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        m = evaluate(model, ds, device, 8, voc, ref)
        print("  use_skeleton={:<5s}  8条 WER={:.4f}  n_distinct={}  {}".format(
            str(use_skel), m["wer"], m["n_distinct"],
            "PASS" if m["wer"] < 0.05 else "FAIL"))
        if m["wer"] >= 0.05 and use_skel:
            print("  !! skeleton 分支未能拟合 8 条，实验终止（新分支可能有 bug）")
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--skel-channels", type=int, default=66)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-eval-n", type=int, default=1500)
    ap.add_argument("--configs", default="flat,skel_add,skel_only")
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--out",
                    default="artifacts/metrics/blank-gov/p29-stgcn.json")
    a = ap.parse_args()

    device = resolve_device("auto")
    tr_labels = read_split("train")
    va_labels = read_split("validation")
    tr_root = ensure_link_dir("train")
    va_root = ensure_link_dir("validation")
    tr_recs = make_records(tr_labels, tr_root, "train")
    va_recs = make_records(va_labels, va_root, "validation")
    print("device = {}   train {} / dev {}".format(device, len(tr_recs), len(va_recs)))

    voc, _ = build_ordered_vocabulary((g for g in tr_labels.values()),
                                      min_frequency=2, max_tokens=300)
    nrm = FeatureNormalizer.fit(
        [np.load(tr_root / (r.sample_id + ".npy")) for r in tr_recs[:600]])
    tr_ds = GlossSequenceDataset(tr_recs, tr_root, voc, nrm, feature_view="full")
    va_ds = GlossSequenceDataset(va_recs, va_root, voc, nrm, feature_view="full")
    ref_tr = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in tr_recs}
    ref_va = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in va_recs}
    # 折叠口径：与训练 target 同源（P28 用的是 CSV ref，会低估 train 拟合）
    ref_tr_f = {r.sample_id: [voc.tokens[i] for i in voc.encode(r.label)] for r in tr_recs}
    ref_va_f = {r.sample_id: [voc.tokens[i] for i in voc.encode(r.label)] for r in va_recs}
    tr_subset = set(random.Random(12345).sample(sorted(ref_tr.keys()),
                                               min(a.train_eval_n, len(ref_tr))))

    if not a.skip_sanity:
        if not sanity_check(voc, tr_root, nrm, device, a):
            raise SystemExit("sanity check 失败，终止")

    results = []
    for cfg_name in a.configs.split(","):
        use_skel = cfg_name != "flat"
        fusion = "add" if cfg_name == "skel_add" else "add"
        print()
        print("#" * 74)
        print("# 配置 {}  use_skeleton={}  fusion={}".format(cfg_name, use_skel, fusion))
        print("#" * 74)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        model = DualStreamCTCRecognizer(
            368, voc.size, hidden_size=a.hidden, num_layers=a.layers,
            dropout=a.dropout, skeleton_channels=a.skel_channels,
            use_skeleton=use_skel, fusion=fusion).to(device)
        nparam = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        print("  参数量 {:.2f}M".format(nparam / 1e6))
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
                tr = evaluate(model, tr_ds, device, a.batch, voc, ref_tr_f, subset=tr_subset)
                dv = evaluate(model, va_ds, device, a.batch, voc, ref_va)
                dv_strict = evaluate(model, va_ds, device, a.batch, voc, ref_va_f)
                hist.append({"epoch": ep, "train_loss": round(tot / max(seen, 1), 4),
                             "train_wer": tr["wer"], "dev_wer": dv["wer"],
                             "dev_wer_folded": dv_strict["wer"],
                             "dev_n_distinct": dv["n_distinct"],
                             "dev_len": dv["len_mean"]})
                print("  ep {:3d} loss={:8.4f} trainWER={:.4f} devWER={:.4f} "
                      "ndist={:4d} len={:.2f}".format(
                          ep, tot / max(seen, 1), tr["wer"], dv["wer"],
                          dv["n_distinct"], dv["len_mean"]), flush=True)
        dt = (time.time() - t0) / 60
        results.append({"config": cfg_name, "use_skeleton": use_skel,
                        "params_M": round(nparam / 1e6, 4),
                        "final": hist[-1], "history": hist, "minutes": round(dt, 2)})

    print()
    print("=" * 78)
    print("P29 汇总（判据：skel_add trainWER<=0.28 且 devWER 比 flat 低 >=0.02）")
    print("=" * 78)
    print("{:<11s} {:>9s} {:>10s} {:>9s} {:>9s}".format(
        "配置", "参数M", "trainWER", "devWER", "ndist"))
    for r in results:
        f = r["final"]
        print("{:<11s} {:>9.2f} {:>10.4f} {:>9.4f} {:>9d}".format(
            r["config"], r["params_M"], f["train_wer"], f["dev_wer"], f["dev_n_distinct"]))

    base = next((r for r in results if r["config"] == "flat"), None)
    skel = next((r for r in results if r["config"] == "skel_add"), None)
    if skel and base:
        d_train = skel["final"]["train_wer"] - base["final"]["train_wer"]
        d_dev = skel["final"]["dev_wer"] - base["final"]["dev_wer"]
        if skel["final"]["train_wer"] <= 0.28 and d_dev <= -0.02:
            verdict = ("ST-GCN 有效：trainWER {:+.4f} 明显下降且 devWER {:+.4f} 改善 -> "
                       "P28 的天花板是结构性的，landmark 拓扑信息确实有用".format(d_train, d_dev))
        elif skel["final"]["train_wer"] <= 0.28:
            verdict = ("拓扑补回后 train 拟合改善（{:+.4f}）但 dev 无收益（{:+.4f}）-> "
                       "过拟合，特征信息量仍不足".format(d_train, d_dev))
        else:
            verdict = ("ST-GCN 无效：trainWER {:+.4f}（未低于 0.28）-> "
                       "拓扑不是瓶颈".format(d_train))
        print()
        print("判决: " + verdict)
    else:
        verdict = "缺少 flat 或 skel_add 对照，无法判决"

    receipt = {
        "experiment": "P29 ST-GCN skeleton branch ablation",
        "motivation": "ref11 ST-GCN Sec 1 criticises flattening joints into vectors; "
                      "ref23 SignFormer-GCN fuses STGCN-LSTM with RGB stream",
        "single_factor": "skeleton branch on/off; features, vocab, hyperparams identical",
        "features_unchanged": True,
        "re_extract_features": False,
        "criterion": "skel_add train_WER<=0.28 and dev_WER <= flat-0.02",
        "sanity_check_run": not a.skip_sanity,
        "all_data_real": True, "reads_test_split": False,
        "results": results, "verdict": verdict,
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
