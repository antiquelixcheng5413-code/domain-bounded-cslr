# -*- coding: utf-8 -*-
"""P26 · 修复后的 blank 率实测（回答「blank 率过高是否已解决」）

## 为什么必须重测

P0~P22 所有 blank 数字都测在 **off-by-one 模型**上，模型从未真正学过，
所以 97% 这个数字**不能用来判断修复后的情况**。

## 一个必须先讲清的口径问题

CTC 是「每帧输出一个类，blank 表示这一帧不产出符号」。
48 帧要输出平均 5.52 个词，**blank 天然必须占大多数**：

```
理论最优 blank 占比（严格单峰对齐）
  = 1 - 2*L/T 近似 = 1 - 2*5.52/48 = 0.7700
  （L 个词各占 1 帧 + 词间至少 L-1 个 blank = 2L-1 帧非 blank）

也就是说 **blank ≈ 77% 才是「健康」的**，97% 才是异常。
本项目 P0 实测 0.9698，比理论健康值高 20 个百分点。
```

所以「blank 率高」本身要分两种：
- **(a) 结构性高**：单峰对齐天然就要 ~77%，这不是病
- **(b) 病理性高**：非 blank 帧连不成有效的单调序列（peaky/坍缩）

判据应看 **peak_frames_per_token** 与 **非 blank 帧的连续性**，
而不是单看 blank_ratio。

## 本脚本测什么（全部真实数据，修复后的模型）

1. **blank_ratio**：与 P0 同口径，可直接对比
2. **理论健康值 0.77 对比**：给出偏离幅度
3. **peak_frames_per_token**：P0 实测 0.26（peaky 病理），现在如何
4. **非 blank 段结构**：多少个连续段、每段几帧
5. **这些数字在 dev 上随训练轮次的变化**：看是否还在改善
6. **train vs dev 的 blank**：判断是否过拟合

## 判据（跑之前写死）

- `blank_ratio` 若降到 ≤0.85 且 `peak_frames_per_token` ≥1
  → **blank 率问题已解决**（回到结构健康区）
- 若仍 >0.90 且 peak <0.5
  → **未解决**，blank 仍是病理

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
def blank_stats(model, ds, device, batch, voc, ref_by_id):
    """与 P0 同口径测 blank，另加 peak/段结构。"""
    model.eval()
    tot_steps = blank_steps = 0
    nonblank = tot_tokens = 0
    peak_frames = []
    n_runs_list = []
    run_len_list = []
    d = n = 0
    for samples in iterate_batches(ds, batch, shuffle=False, seed=42):
        b = to_torch_batch(collate_samples(samples), device)
        logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        dec, bl, st = decode_batch(lp, ol.cpu().numpy(), 1)
        tot_steps += st
        blank_steps += bl
        for r, ids in enumerate(dec):
            t = int(ol[r])
            am = lp[r, :t].argmax(axis=1)
            nb = am != BLANK_INDEX
            nonblank += int(nb.sum())
            # 连续非 blank 段
            runs = []
            cur = 0
            for v in nb:
                if v:
                    cur += 1
                elif cur:
                    runs.append(cur)
                    cur = 0
            if cur:
                runs.append(cur)
            n_runs_list.append(len(runs))
            run_len_list.extend(runs)
            hyp = voc.decode(list(ids))
            ref = ref_by_id[b["sample_ids"][r]]
            tot_tokens += len(hyp)
            peak_frames.append(int(nb.sum()) / max(len(hyp), 1))
            d += levenshtein(list(ref), hyp)[0]
            n += len(ref)
    rl = np.array(run_len_list) if run_len_list else np.array([0])
    return {
        "blank_ratio": round(blank_steps / max(tot_steps, 1), 4),
        "peak_frames_per_token": round(float(np.mean(peak_frames)), 4),
        "n_nonblank_runs_mean": round(float(np.mean(n_runs_list)), 4),
        "run_len_mean": round(float(rl.mean()), 4),
        "run_len_median": float(np.median(rl)),
        "run_len_p90": float(np.percentile(rl, 90)),
        "nonblank_ratio": round(nonblank / max(tot_steps, 1), 4),
        "mean_pred_len": round(tot_tokens / max(len(peak_frames), 1), 4),
        "wer": round(d / max(n, 1), 4),
    }


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
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p26-blank-after-fix.json")
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
                                      min_frequency=a.min_freq, max_tokens=a.max_tokens)
    nrm = FeatureNormalizer.fit(
        [np.load(tr_root / (r.sample_id + ".npy")) for r in tr_recs[:600]])
    tr_ds = GlossSequenceDataset(tr_recs, tr_root, voc, nrm, feature_view="full")
    va_ds = GlossSequenceDataset(va_recs, va_root, voc, nrm, feature_view="full")
    ref_tr = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in tr_recs}
    ref_va = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in va_recs}

    # 理论健康 blank 值：L 个词各 1 帧 + (L-1) 个词间 blank = 2L-1 非 blank
    T = 48
    L_mean = float(np.mean([len(v) for v in ref_va.values()]))
    nonblank_needed = max(2 * L_mean - 1, 1)
    healthy_blank = 1.0 - nonblank_needed / T

    print("=" * 76)
    print("blank 率口径说明")
    print("=" * 76)
    print("  T = {} 帧，平均词数 L = {:.2f}".format(T, L_mean))
    print("  单调对齐最少需要非 blank 帧 = 2L-1 = {:.2f}".format(nonblank_needed))
    print("  => 理论健康 blank 占比 = {:.4f}".format(healthy_blank))
    print("  => blank 高达 {:.4f} 属结构性必然，不是病".format(1.0))
    print("  P0（off-by-one 模型）实测 0.9698，比健康值高 {:.1f} 个百分点".format(
        (0.9698 - healthy_blank) * 100))

    cfg = CTCConfig(input_size=368, vocabulary_size=voc.size, hidden_size=256,
                    num_layers=2, dropout=0.3, bidirectional=True,
                    projection_size=256, subsample_stride=1)
    model = CTCRecognizer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

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

        if ep in (1, 2, 5, 10, 15, 20, 25, 30):
            dv = blank_stats(model, va_ds, device, a.batch, voc, ref_va)
            rec = {"epoch": ep, "train_loss": round(tot / max(seen, 1), 4), "dev": dv}
            hist.append(rec)
            print("")
            print("ep {:2d}  loss={:.4f}".format(ep, rec["train_loss"]))
            print("    blank_ratio            = {:.4f}   (理论健康 {:.4f})".format(
                dv["blank_ratio"], healthy_blank))
            print("    nonblank_ratio         = {:.4f}".format(dv["nonblank_ratio"]))
            print("    peak_frames_per_token  = {:.4f}   (P0 off-by-one 时 0.26)".format(
                dv["peak_frames_per_token"]))
            print("    非 blank 段数/样本      = {:.4f}".format(dv["n_nonblank_runs_mean"]))
            print("    段长 mean/median/p90   = {:.3f} / {:.1f} / {:.1f}".format(
                dv["run_len_mean"], dv["run_len_median"], dv["run_len_p90"]))
            print("    输出长度均值           = {:.4f}   WER={:.4f}".format(
                dv["mean_pred_len"], dv["wer"]))

    dt = (time.time() - t0) / 60
    last = hist[-1]["dev"]
    tr_last = None
    print("")
    print("=" * 76)
    print("train 集同口径复测（判断是否过拟合）")
    print("=" * 76)
    tr_last = blank_stats(model, tr_ds, device, a.batch, voc, ref_tr)
    print("  blank_ratio           = {:.4f}".format(tr_last["blank_ratio"]))
    print("  peak_frames_per_token = {:.4f}".format(tr_last["peak_frames_per_token"]))
    print("  WER                   = {:.4f}".format(tr_last["wer"]))

    br = last["blank_ratio"]
    pf = last["peak_frames_per_token"]
    if br <= 0.85 and pf >= 1.0:
        verdict = ("blank 率问题已解决：blank_ratio {:.4f} 进入结构健康区（理论 {:.4f}），"
                   "peak_frames_per_token {:.4f} 回到 >=1 的健康形态"
                   ).format(br, healthy_blank, pf)
    elif br > 0.90 or pf < 0.5:
        verdict = ("blank 率**未解决**：blank_ratio {:.4f}，peak_frames_per_token {:.4f}，"
                   "非 blank 帧仍不成有效单调序列".format(br, pf))
    else:
        verdict = ("部分改善但仍偏高：blank_ratio {:.4f}（健康 {:.4f}），peak {:.4f}"
                   ).format(br, healthy_blank, pf)
    print("")
    print("判决: " + verdict)

    receipt = {
        "experiment": "P26 blank ratio after off-by-one fix",
        "question": "is the high blank ratio problem solved?",
        "T_frames": T, "mean_ref_len": round(L_mean, 2),
        "min_nonblank_frames_single_peak": round(nonblank_needed, 2),
        "theoretical_healthy_blank_ratio": round(healthy_blank, 4),
        "note": "CTC needs blank to separate symbols; healthy blank is ~77%, not ~0%",
        "before_fix_off_by_one_model": {"blank_ratio": 0.9698,
                                        "peak_frames_per_token": 0.26,
                                        "wer": 0.9926, "source": "P0 diag_peaky"},
        "after_fix": {"dev": hist, "train": tr_last},
        "verdict": verdict,
        "all_data_real": True, "reads_test_split": False,
        "minutes": round(dt, 2),
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
