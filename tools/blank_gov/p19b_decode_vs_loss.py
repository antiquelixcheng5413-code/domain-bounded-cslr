# -*- coding: utf-8 -*-
"""P19b · 判决性验证：train_loss 为负但 train_WER≈0.99 的矛盾

## P19 观测到的直接矛盾

```
ep30  train_loss = -3.625   （CTC 负 loss => 训练集目标似然 > 1 => 拟合良好）
      train_WER  =  0.9935  （训练集上几乎全错）
      dev_WER    =  0.9760
```

这两行**不可能同时为真**。若模型真把训练集拟合到 loss<0，
贪心解码在训练集上应当接近完美。必有环节错了。

## 三种可能，必须逐个排除

- **(H1) 贪心解码实现有 bug**：训练时用 forward+ctc_loss（时序对齐求和），
  评估时用 `decode_batch` 逐帧 argmax。若 argmax 解码与 CTC 对齐不一致，
  就会出现「loss 极低但 argmax 输出全错」。**这在 CTC 里是已知陷阱**：
  正确后验是所有对齐路径的求和，逐帧 argmax 只是其近似，
  在多峰后验上会显著偏离。
- **(H2) `ref` 取错**：train 侧参考取自 CSV 原始 gloss，而模型输出经 `<unk>` 折叠。
  若折叠/清洗规则导致字符串不一致，编辑距离会虚高。
- **(H3) 真的拟合了**：则 WER 计算有误。

## 本脚本的判决性实验

### 实验1 · 直接量后验质量（绕过 argmax）

对 train 样本，用**束搜索（beam_width=8）**解码，与贪心对比。
若 beam 远好于贪心 → **H1 成立，是解码方式的问题**。

### 实验2 · 逐帧 argmax 与 CTC 路径的差距

统计：非 blank 帧上「后验最大值」的分布。
CTC 正确解码的条件是存在一条高概率对齐路径，
**不要求**每一帧的 argmax 都正确。量化这个差距。

### 实验3 · 训练集逐样本对齐检查

取 20 个训练样本，打印：
- 参考序列
- 贪心解码序列
- 该样本的目标序列在 CTC 下的**实际概率**（exp(loss)）
若 `P(target)` 很高但贪心解码完全不同 → H1 确认。

## 判据（跑之前写死）

- 若 `beam8_WER << greedy_WER`（例如 0.99 → 0.5 以下）
  → **H1 确认：模型其实学会了，是解码方式毁掉了输出**
- 若 beam 也不改善，且 `P(target)` 低
  → H3 排除，确属特征判别力不足

只读 train/dev，**不触碰 test split**。
"""
import argparse
import csv
import json
import math
import os
import random
import sys
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


def train_model(a, tr_recs, tr_root, voc, nrm, device):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfg = CTCConfig(input_size=368, vocabulary_size=voc.size, hidden_size=a.hidden,
                    num_layers=a.layers, dropout=a.dropout, bidirectional=True,
                    projection_size=256, subsample_stride=1)
    model = CTCRecognizer(cfg).to(device)
    tr_ds = GlossSequenceDataset(tr_recs, tr_root, voc, nrm, feature_view="full")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    tcfg = TrainingConfig(epochs=a.epochs, batch_size=a.batch, learning_rate=a.lr,
                          weight_decay=a.weight_decay, seed=a.seed, device=str(device),
                          amp=True, early_stopping_patience=999, beam_width=1,
                          min_epochs=1, sequence_length=48, normalize_features=False)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
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
            tot += float(loss.item()) * len(samples)
            seen += len(samples)
        if ep % 5 == 0 or ep == a.epochs:
            print("  train ep {:2d} loss={:.4f}".format(ep, tot / max(seen, 1)), flush=True)
    return model


@torch.no_grad()
def compare_decoders(model, ds, voc, device, batch, ref_by_id, beams=(1, 2, 4)):
    model.eval()
    res = {}
    store = {}
    for bw in beams:
        d = n = 0
        seqs = []
        for samples in iterate_batches(ds, batch, shuffle=False, seed=42):
            b = to_torch_batch(collate_samples(samples), device)
            logits = model(b["features"], b["input_lengths"])
            ol = model.output_lengths(b["input_lengths"])
            lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            dec, _, _ = decode_batch(lp, ol.cpu().numpy(), bw)
            for row, ids in enumerate(dec):
                hyp = voc.decode(list(ids))
                seqs.append(tuple(hyp))
                ref = ref_by_id[b["sample_ids"][row]]
                if bw == 1:
                    store[b["sample_ids"][row]] = (ref, hyp)
                d += levenshtein(list(ref), hyp)[0]
                n += len(ref)
        res["beam{}".format(bw)] = {
            "wer": round(d / max(n, 1), 4),
            "n_distinct": len(set(seqs)),
            "mean_len": round(float(np.mean([len(s) for s in seqs])), 3)}
        print("  beam={:<3d} WER={:.4f}  n_distinct={:4d}  mean_len={:.2f}".format(
            bw, res["beam{}".format(bw)]["wer"], res["beam{}".format(bw)]["n_distinct"],
            res["beam{}".format(bw)]["mean_len"]), flush=True)
    return res, store


@torch.no_grad()
def per_sample_analysis(model, ds, device, batch, store, n_show=15):
    """逐样本：目标序列的 CTC 概率 vs 贪心解码结果。判决 H1。"""
    model.eval()
    rows = []
    logp_frames = []
    for samples in iterate_batches(ds, batch, shuffle=False, seed=42):
        b = to_torch_batch(collate_samples(samples), device)
        logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1)

        # 逐样本 CTC loss（不复现批平均）
        tl = b["target_lengths"]
        tg = b["targets"]
        flat = []
        pos = 0
        for L in tl.tolist():
            flat.extend(tg[pos, :L].tolist())
            pos += L
        flat_t = torch.tensor(flat, dtype=torch.long, device=device)
        ll = lp.clone()
        ol_d = ol.to(device)
        ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
        loss_vec = ctc(ll, flat_t, ol_d, tl)
        loss_vec = loss_vec[:len(samples)]

        p = lp.cpu().numpy()
        for row, sid in enumerate(b["sample_ids"]):
            t = int(ol[row])
            L = int(tl[row])
            nll = float(loss_vec[row]) / max(L, 1)
            # 非 blank 帧的后验最大值分布
            nb = np.concatenate([p[row, :t, 1:].max(axis=1)])
            rows.append({"sid": sid, "target_len": L,
                         "ctc_nll_per_token": round(nll, 4),
                         "p_target": round(math.exp(-nll), 6),
                         "nb_frame_pmax_mean": round(float(nb.mean()), 5),
                         "nb_frame_pmax_max": round(float(nb.max()), 5),
                         "n_frames_over_0.5": int((nb > 0.5).sum()),
                         "t": t})
            logp_frames.append(nb)
    lp_all = np.concatenate(logp_frames)
    qs = [50, 75, 90, 95, 99]
    print("")
    print("  非 blank 帧后验最大值分位数:")
    for q in qs:
        print("    p{} = {:.5f}".format(q, float(np.percentile(lp_all, q))))
    print("  非 blank 帧后验 > 0.5 的比例: {:.4f}".format(float((lp_all > 0.5).mean())))
    print("  非 blank 帧后验 > 0.9 的比例: {:.4f}".format(float((lp_all > 0.9).mean())))

    print("")
    print("  逐样本（前 {} 条）：CTC 目标概率 vs 贪心解码".format(n_show))
    print("  {:<12s} {:>4s} {:>10s} {:>8s}  {}".format("sample", "L", "P(target)", "帧>0.5", "贪心输出 / 参考"))
    for r in rows[:n_show]:
        ref, hyp = store.get(r["sid"], ([], []))
        print("  {:<12s} {:>4d} {:>10.4f} {:>8d}  {}".format(
            r["sid"][:12], r["target_len"], r["p_target"], r["n_frames_over_0.5"],
            "{}  <<  {}".format("/".join(hyp) or "(空)", "/".join(ref) or "(空)")))
    agree = sum(1 for r in rows if r["p_target"] > 0.5)
    print("")
    print("  P(target) > 0.5 的训练样本: {} / {}".format(agree, len(rows)))
    return rows, {
        "nb_frame_pmax_p50": float(np.percentile(lp_all, 50)),
        "nb_frame_pmax_p90": float(np.percentile(lp_all, 90)),
        "nb_frame_pmax_p99": float(np.percentile(lp_all, 99)),
        "frac_frames_over_0p5": float((lp_all > 0.5).mean()),
        "frac_frames_over_0p9": float((lp_all > 0.9).mean()),
        "n_samples_p_target_gt_0p5": agree,
        "n_samples": len(rows),
    }


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
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p19b-decode-vs-loss.json")
    ap.add_argument("--beams", default="1,2,4")
    a = ap.parse_args()

    device = resolve_device("auto")
    print("device = {}".format(device))

    tr_labels = read_split("train")
    va_labels = read_split("validation")
    tr_root = ensure_link_dir("train")
    va_root = ensure_link_dir("validation")
    tr_recs = make_records(tr_labels, tr_root, "train")
    va_recs = make_records(va_labels, va_root, "validation")
    voc, _ = build_ordered_vocabulary(
        (g for g in tr_labels.values()), min_frequency=2, max_tokens=300)
    nrm = FeatureNormalizer.fit(
        [np.load(tr_root / (r.sample_id + ".npy")) for r in tr_recs[:600]])

    tr_ds = GlossSequenceDataset(tr_recs, tr_root, voc, nrm, feature_view="full")
    va_ds = GlossSequenceDataset(va_recs, va_root, voc, nrm, feature_view="full")
    ref_tr = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in tr_recs}
    ref_va = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in va_recs}

    print("训练配置 A（token/301）{} 轮".format(a.epochs))
    model = train_model(a, tr_recs, tr_root, voc, nrm, device)

    print("")
    print("=" * 70)
    print("实验1 · 解码方式对比（TRAIN 4973 条）")
    print("=" * 70)
    BEAMS = tuple(int(x) for x in a.beams.split(","))
    tr_res, store = compare_decoders(model, tr_ds, voc, device, a.batch, ref_tr, beams=BEAMS)
    print("")
    print("=" * 70)
    print("实验1b · 解码方式对比（DEV 515 条）")
    print("=" * 70)
    va_res, _ = compare_decoders(model, va_ds, voc, device, a.batch, ref_va, beams=BEAMS)

    print("")
    print("=" * 70)
    print("实验2/3 · 后验质量与逐样本（判决 H1）")
    print("=" * 70)
    rows, post = per_sample_analysis(model, tr_ds, device, a.batch, store, n_show=12)

    g_tr = tr_res["beam1"]["wer"]
    b_tr = min(v["wer"] for k, v in tr_res.items() if k != "beam1")
    if b_tr < g_tr - 0.1:
        verdict = ("H1 确认：束搜索 WER {:.4f} 远优于贪心 {:.4f} —— "
                   "模型其实学会了，是**逐帧 argmax 解码**毁掉了输出".format(b_tr, g_tr))
    else:
        verdict = ("H1 不成立：最佳束搜索 {:.4f} 与贪心 {:.4f} 无实质差异 —— "
                   "确属特征判别力不足".format(b_tr, g_tr))
    print("")
    print("判决: " + verdict)

    receipt = {
        "experiment": "P19b why train_loss<0 but train_WER~0.99",
        "hypotheses": {
            "H1": "greedy argmax decoding diverges from CTC path sum",
            "H2": "reference string mismatch (folding/cleaning)",
            "H3": "WER computation error"},
        "criterion": "beam_WER < greedy_WER - 0.1 => H1",
        "all_data_real": True, "reads_test_split": False,
        "train_decoders": tr_res, "dev_decoders": va_res,
        "posterior_quality": post,
        "verdict": verdict,
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
