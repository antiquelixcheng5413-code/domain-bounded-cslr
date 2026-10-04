# -*- coding: utf-8 -*-
"""P20 · 管线自检：CTC 能否过拟合一个小批次（判决 loss 与 decode 是否自洽）

## 必须先排除的矛盾

P19b 实测（配置 A，30 轮，真实 train 4973）：

```
train_loss = -3.6252      负 loss => 训练集目标似然极高
train_WER  =  0.9935      训练集上几乎全错
beam2/4    =  1.0145/1.0131   束搜索更差 => 不是解码方式问题
```

**这两行不可能同时为真。**
若每 token 对数似然为 +3.6，则每 token 概率约 e^3.6≈37，
整句概率约 10^8.7 —— 模型在该样本上几乎是确定性的，
此时逐帧 argmax **必然**能还原目标序列，WER 不可能是 0.99。

所以在讨论「特征判别力」之前，必须先证明
**loss 计算与 decode 实现是自洽的**。这是整个实验链的地基。

## 判决性实验：8 条样本过拟合测试

取 8 条真实 train 样本，训练 400 轮（无正则、无 dropout、无 shuffle），
每 50 轮打印：

- **eval 模式**下的 CTC loss（排除 dropout 噪声）
- 逐样本 `P(target)`
- 贪心解码序列 vs 参考序列，逐条打印

这是 CTC 社区的标准 sanity check：一个正常实现的 CTC
**必须**能把 8 条样本拟合到 loss 接近 0 且解码完全正确。

## 判据（跑之前写死）

- 若 8 条被完美拟合（loss → 接近 0，解码逐条正确）
  → **管线自洽**。那么 4973 条上的矛盾只能解释为
  「模型在训练集上找到了高概率但 argmax 不可达的解」，
  需进一步查特征尺度/归一化。
- 若 8 条都拟合不了
  → **管线有 bug**（loss 归一化、blank 索引、target 构造或 decode 有误）。
  在修好之前，P0~P19b 所有基于 WER 的结论都不可信。

只读 train，**不触碰 dev 与 test**。
"""
import argparse
import csv
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
def eval_detail(model, ds, device, voc, ref_by_id, tag):
    """eval 模式下逐样本算 CTC loss / P(target) / 贪心解码。

    必须走仓库的 ``ctc_loss(..., reduction='none')``：
    它接受 padded ``[B, S]`` targets 并返回 per-sample loss。
    自己 flatten 再喂 ``nn.CTCLoss`` 在 torch 2.14 上会报
    "input_lengths must be of size batch_size"（实测）。
    """
    model.eval()
    rows = []
    for samples in iterate_batches(ds, 8, shuffle=False, seed=42):
        b = to_torch_batch(collate_samples(samples), device)
        logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        tl = b["target_lengths"]
        # 仓库封装：padded targets 直传，返回 per-sample loss
        lv = ctc_loss(logits, b["input_lengths"], b["targets"], tl, ol,
                      reduction="none")
        lp = torch.log_softmax(logits.float(), dim=-1)
        lp_np = lp.cpu().numpy()
        dec, _, _ = decode_batch(lp_np, ol.cpu().numpy(), 1)
        for r, ids in enumerate(dec):
            L = int(tl[r])
            nll = float(lv[r]) / max(L, 1)
            hyp = voc.decode(list(ids))
            ref = ref_by_id[b["sample_ids"][r]]
            rows.append({"sid": b["sample_ids"][r], "L": L,
                         "nll": nll, "p": math.exp(-nll),
                         "hyp": hyp, "ref": ref,
                         "edit": levenshtein(list(ref), hyp)[0]})
    tot_nll = sum(r["nll"] for r in rows) / max(len(rows), 1)
    tot_w = sum(r["edit"] for r in rows) / max(sum(len(r["ref"]) for r in rows), 1)
    perfect = sum(1 for r in rows if r["edit"] == 0)
    print("  [{}] eval loss/token={:.4f}  WER={:.4f}  完美解码 {}/{}  "
          "P(target) 中位={:.4f}".format(
              tag, tot_nll, tot_w, perfect, len(rows),
              float(np.median([r["p"] for r in rows]))))
    return rows, {"loss_per_token": round(tot_nll, 4), "wer": round(tot_w, 4),
                  "n_perfect": perfect, "n": len(rows),
                  "p_target_median": round(float(np.median([r["p"] for r in rows])), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p20-sanity-overfit.json")
    a = ap.parse_args()

    device = resolve_device("auto")
    print("device = {}".format(device))

    tr_labels = read_split("train")
    tr_root = ensure_link_dir("train")
    tr_recs = make_records(tr_labels, tr_root, "train")
    random.seed(a.seed)
    # 选目标长度适中的样本，避免一开始就不满足 CTC 可行性
    cand = [r for r in tr_recs
            if 3 <= len([t for t in r.label.split("/") if t.strip()]) <= 6]
    pick = cand[: a.n_samples]
    print("选取 {} 条真实 train 样本（目标长度 3~6）".format(len(pick)))
    for r in pick:
        print("  {}  {}".format(r.sample_id, r.label))

    voc, _ = build_ordered_vocabulary(
        (g for g in tr_labels.values()), min_frequency=2, max_tokens=300)
    nrm = FeatureNormalizer.fit([np.load(tr_root / (r.sample_id + ".npy")) for r in pick])
    ds = GlossSequenceDataset(pick, tr_root, voc, nrm, feature_view="full")
    ref = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in pick}

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfg = CTCConfig(input_size=368, vocabulary_size=voc.size, hidden_size=a.hidden,
                    num_layers=a.layers, dropout=a.dropout, bidirectional=True,
                    projection_size=256, subsample_stride=1)
    model = CTCRecognizer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)

    print("")
    print("=" * 74)
    print("过拟合测试：{} 条样本 / {} 轮 / dropout={} / wd=0".format(
        len(pick), a.epochs, a.dropout))
    print("=" * 74)
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = seen = 0
        for samples in iterate_batches(ds, 8, shuffle=False, seed=0):
            b = to_torch_batch(collate_samples(samples), device)
            opt.zero_grad(set_to_none=True)
            logits = model(b["features"], b["input_lengths"])
            ol = model.output_lengths(b["input_lengths"])
            loss = ctc_loss(logits, b["input_lengths"], b["targets"],
                            b["target_lengths"], ol)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()) * len(samples)
            seen += len(samples)
        if ep % 50 == 0 or ep == a.epochs:
            rows, st = eval_detail(model, ds, device, voc, ref, "ep{}".format(ep))
            hist.append({"epoch": ep, "train_loss": tot / max(seen, 1), **st})
            if ep == a.epochs:
                print("")
                print("  最终逐样本明细：")
                for r in rows:
                    print("    {}  L={}  P(target)={:.4f}  edit={}".format(
                        r["sid"][:12], r["L"], r["p"], r["edit"]))
                    print("      ref : {}".format("/".join(r["ref"])))
                    print("      hyp : {}".format("/".join(r["hyp"]) or "(空)"))

    last = hist[-1]
    ok = last["wer"] < 0.05 and last["n_perfect"] == last["n"]
    verdict = ("管线自洽：CTC 能完美拟合 8 条样本 -> loss/decode 实现正确，"
               "4973 条上的矛盾需另找解释" if ok else
               "管线有 bug：连 8 条样本都拟合不了 -> P0~P19b 的 WER 结论全部不可信")
    print("")
    print("=" * 74)
    print("判决: " + verdict)
    print("=" * 74)

    import json
    receipt = {
        "experiment": "P20 sanity check - can CTC overfit 8 real samples",
        "criterion": "WER<0.05 and all samples decoded perfectly => pipeline self-consistent",
        "all_data_real": True, "reads_dev": False, "reads_test_split": False,
        "n_samples": len(pick), "epochs": a.epochs,
        "lr": a.lr, "dropout": a.dropout, "weight_decay": 0.0,
        "history": hist, "final": last, "verdict": verdict,
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
