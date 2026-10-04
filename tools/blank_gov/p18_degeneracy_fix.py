# -*- coding: utf-8 -*-
"""P18 · 判决性实验：输出坍缩的修复（推翻 P16 结论）

## P17 诊断出的三个事实（全部真实数据实测）

```
1. 存盘 checkpoint = epoch 2（WER 最低的那一轮），n_distinct_outputs = 6
   -> P0/P13/P14/P15/P16 全部诊断的是一个只训了 2 epoch 的模型
2. 训练目标里 <unk> 占 28.3%（train 8002/28272），dev 占 26.0%
   -> 「输出坍缩到 <unk>/。/我/？」有明确的机制来源
3. CTC 对齐容量充足：T=48，需要 2L-1 均值 10.04，不可行样本 0/514
   -> 排除「时间分辨率不足导致吐不出多词」
```

## 机制解释（本实验要检验的假设）

cap300 词表下，**28.3% 的训练目标 token 全是同一个符号 `<unk>`**。
CTC 优化的是序列似然，「在不确定的位置吐 `<unk>`」是一个**低风险的局部最优**：
它能立刻拿到 28.3% 的位置正确率，而其余 71.7% 分散在 300 个类里、极难分辨。

特征判别力上限（macro AUC 0.68~0.71）低于区分 300 类所需 →
梯度永远朝 `<unk>` 走 → **输出坍缩是任务定义的自证闭环**，
不是「模型没学到区分」。

## 三个修复方向（都是解除机制，不是换模型）

| 配置 | 解除的机制 | 论文依据 |
|---|---|---|
| A 基线 token/cap300 | —— | 复现现状 |
| B token/全量词表 3517 | `<unk>` 从 28.3% 降到多少 | 工程（词表裁剪是人为的） |
| C **char 目标** | OOV 折叠几乎消失 + 目标长度 ~2x | gloss_sequence.py 既有设计 |

### 为什么 C（字符级）最可能有效

`GlossSequenceConfig.target_unit` 的 docstring 已写明设计意图：

> "CE-CSL token labels are extremely sparse (73% of tokens occur once or twice
>  on a small train split), while characters are ~4x denser and give CTC
>  longer targets to align against."

字符级同时做两件事：
1. **目标变长** → CTC 每个位置都有监督，不再有 28.3% 的一坨 `<unk>`
2. **`<unk>` 折叠几乎消失** → 常用汉字全在词表内，坍缩的燃料被抽掉

## 判据（跑之前写死，不许事后改）

**`n_distinct_outputs` 必须从 6 提升到 > 50**（P16 定的判据，沿用）。
同时报 WER / 输出长度 / 训练损失 / blank 率。

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

CSV_TEXT = None


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
        if not [t for t in g.split("/") if t.strip()]:
            continue
        out.append(SampleRecord(sample_id=sid, video=Path(sid + ".mp4"), label=g,
                                signer="x", session="x", split=split))
    out.sort(key=lambda r: r.sample_id)
    return out


def target_stats(records, voc):
    tot = unk = 0
    lens = []
    for r in records:
        ids = voc.encode(r.label)
        if not ids:
            continue
        lens.append(len(ids))
        tot += len(ids)
        unk += sum(1 for i in ids if i == 0)
    return {"n_tokens": tot, "n_unk": unk, "unk_rate": unk / max(tot, 1),
            "target_len_mean": float(np.mean(lens)) if lens else 0.0,
            "target_len_max": int(np.max(lens)) if lens else 0,
            "vocab_size": int(voc.size)}


@torch.no_grad()
def measure(model, ds, voc, device, batch, ref_by_id):
    model.eval()
    seqs, lens, kinds = [], [], set()
    d = n = 0
    for samples in iterate_batches(ds, batch, shuffle=False, seed=42):
        b = to_torch_batch(collate_samples(samples), device)
        logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        dec, _, _ = decode_batch(lp, ol.cpu().numpy(), 1)
        for row, ids in enumerate(dec):
            hyp = voc.decode(list(ids))
            seqs.append(tuple(hyp))
            lens.append(len(hyp))
            kinds.update(hyp)
            ref = ref_by_id.get(b["sample_ids"][row]) if "sample_ids" in b else b["tokens"][row]
            d += levenshtein(list(ref), hyp)[0]
            n += len(ref)
    cnt = collections.Counter(seqs)
    return {
        "n_samples": len(seqs),
        "n_distinct_outputs": len(cnt),
        "out_len_mean": float(np.mean(lens)) if lens else 0.0,
        "out_len_max": int(np.max(lens)) if lens else 0,
        "token_kinds": len(kinds),
        "wer": d / max(n, 1),
        "top3": [[list(k), v] for k, v in cnt.most_common(3)],
    }


def run_config(name, max_tokens, min_freq, target_unit, args, tr_recs, va_recs,
               tr_root, va_root, nrm, tr_labels):
    print("")
    print("#" * 74)
    print("# 配置 {} : max_tokens={} min_freq={} target_unit={}".format(
        name, max_tokens, min_freq, target_unit))
    print("#" * 74)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gcfg = GlossSequenceConfig(target_unit=target_unit)
    voc, _ = build_ordered_vocabulary(
        (g for g in tr_labels.values()), min_frequency=min_freq,
        max_tokens=max_tokens, config=gcfg)
    ts_tr = target_stats(tr_recs, voc)
    ts_va = target_stats(va_recs, voc)
    print("  词表 {} 类   train 目标长度 mean {:.2f} max {}   <unk> {:.1%}".format(
        voc.size, ts_tr["target_len_mean"], ts_tr["target_len_max"], ts_tr["unk_rate"]))
    print("  dev   目标长度 mean {:.2f}   <unk> {:.1%}".format(
        ts_va["target_len_mean"], ts_va["unk_rate"]))

    # T=48 的 CTC 可行性：2L-1 <= 48
    infeasible = sum(1 for r in va_recs
                     if len(voc.encode(r.label)) * 2 - 1 > args.seq_len)
    print("  dev 中 CTC 不可行样本（2L-1 > {}）: {} / {}".format(
        args.seq_len, infeasible, len(va_recs)))

    cfg = CTCConfig(input_size=368, vocabulary_size=voc.size,
                    hidden_size=args.hidden, num_layers=args.layers,
                    dropout=args.dropout, bidirectional=True,
                    projection_size=256, subsample_stride=1)
    device = resolve_device("auto")
    model = CTCRecognizer(cfg).to(device)

    tr_ds = GlossSequenceDataset(tr_recs, tr_root, voc, nrm, feature_view="full")
    va_ds = GlossSequenceDataset(va_recs, va_root, voc, nrm, feature_view="full")
    # 参考序列必须与 target_unit 同一层级。
    # 字符级配置下用词级参考算 WER 是错的（拿字符比词），曾导致 C 配置 WER 报成 1.0578。
    # 字符级报 CER，词级报 WER，两者不可直接比较。
    if target_unit == "char":
        ref_by_id = {r.sample_id: voc.units(r.label) for r in va_recs}
        metric_name = "CER"
    else:
        ref_by_id = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()]
                     for r in va_recs}
        metric_name = "WER"
    ref_len_mean = float(np.mean([len(v) for v in ref_by_id.values() if v]))

    tcfg = TrainingConfig(epochs=args.epochs, batch_size=args.batch,
                          learning_rate=args.lr, weight_decay=args.weight_decay,
                          seed=args.seed, device=str(device), amp=True,
                          early_stopping_patience=999,
                          beam_width=1, min_epochs=1,
                          sequence_length=args.seq_len, normalize_features=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist = []
    t0 = time.time()
    for ep in range(1, tcfg.epochs + 1):
        model.train()
        ep_loss, seen = 0.0, 0
        cur_lr = learning_rate_at(ep, tcfg.epochs, args.lr,
                                  warmup_epochs=args.warmup,
                                  schedule="cosine", min_ratio=0.05)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        for samples in iterate_batches(tr_ds, args.batch, shuffle=True,
                                       seed=args.seed + ep):
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
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                opt.step()
            ep_loss += float(loss.item()) * len(samples)
            seen += len(samples)

        m = measure(model, va_ds, voc, device, args.batch, ref_by_id)
        rec = {"epoch": ep, "train_loss": ep_loss / max(seen, 1),
               "n_distinct_outputs": m["n_distinct_outputs"],
               "token_kinds": m["token_kinds"],
               "out_len_mean": round(m["out_len_mean"], 3),
               metric_name.lower(): round(m["wer"], 4)}
        hist.append(rec)
        print("  ep {:2d}  loss={:8.4f}  n_distinct={:4d}  kinds={:4d}  "
              "len={:5.2f}  {}={:.4f}".format(
                  ep, rec["train_loss"], rec["n_distinct_outputs"],
                  rec["token_kinds"], rec["out_len_mean"], metric_name,
                  rec[metric_name.lower()]), flush=True)

    dt = (time.time() - t0) / 60
    final = hist[-1]
    verdict = ("PASS 坍缩已解除" if final["n_distinct_outputs"] > 50
               else "FAIL 坍缩未解除")
    print("  => {}  n_distinct 6 -> {}   {}={:.4f} (参考长度 mean {:.2f})   ({:.1f} 分钟)".format(
        verdict, final["n_distinct_outputs"], metric_name,
        final[metric_name.lower()], ref_len_mean, dt))

    return {
        "name": name, "max_tokens": max_tokens, "min_frequency": min_freq,
        "target_unit": target_unit, "vocab_size": int(voc.size),
        "metric": metric_name,
        "train_target": ts_tr, "dev_target": ts_va,
        "dev_ctc_infeasible": int(infeasible),
        "dev_ref_len_mean": round(ref_len_mean, 2),
        "history": hist, "final": final, "verdict": verdict,
        "minutes": round(dt, 2),
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
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default="A:300:2:token,B::1:token,C::1:char")
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p18-degeneracy-fix.json")
    a = ap.parse_args()

    print("device = {}".format(resolve_device("auto")))
    tr_labels = read_split("train")
    va_labels = read_split("validation")
    tr_root = ensure_link_dir("train")
    va_root = ensure_link_dir("validation")
    tr_recs = make_records(tr_labels, tr_root, "train")
    va_recs = make_records(va_labels, va_root, "validation")
    print("train {} / dev {}".format(len(tr_recs), len(va_recs)))

    nrm = FeatureNormalizer.fit(
        [np.load(tr_root / (r.sample_id + ".npy")) for r in tr_recs[:600]])
    print("normalizer 来自 {} 条真实 train 特征".format(min(600, len(tr_recs))))

    results = []
    for item in a.configs.split(","):
        nm, mt, mf, tu = item.split(":")
        results.append(run_config(
            nm, int(mt) if mt else None, int(mf), tu, a,
            tr_recs, va_recs, tr_root, va_root, nrm, tr_labels))

    print("")
    print("=" * 78)
    print("P18 汇总（判据：n_distinct_outputs > 50）")
    print("=" * 78)
    print("{:<4} {:>7} {:>7} {:>7} {:>9} {:>9} {:>7} {}".format(
        "配置", "词表", "unk率", "层级", "n_distinct", "kinds", "out_len", "判定"))
    for r in results:
        print("{:<4} {:>7} {:>7.1%} {:>7} {:>9} {:>9} {:>7.2f} {}".format(
            r["name"], r["vocab_size"], r["train_target"]["unk_rate"],
            r["target_unit"], r["final"]["n_distinct_outputs"],
            r["final"]["token_kinds"], r["final"]["out_len_mean"], r["verdict"]))
    print("\n注意：词级报 WER，字符级报 CER，两者不可直接比较。")
    print("CER 的下界受字符级无法跨词边界对齐影响，不是 gloss WER。")

    receipt = {
        "experiment": "P18 fix output degeneracy by removing the <unk> collapse loop",
        "hypothesis": "28.3% of training targets are the single symbol <unk>; "
                      "with weak features (macro AUC 0.68-0.71) CTC collapses onto it",
        "criterion": "n_distinct_outputs > 50 (pre-registered in P16)",
        "single_factor": "target unit / vocabulary size only; model, features, seed identical",
        "all_data_real": True,
        "reads_test_split": False,
        "configs": results,
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
