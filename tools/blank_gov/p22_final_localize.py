# -*- coding: utf-8 -*-
"""P22 · 最终定位：正确帧上的 argmax 类到底是什么（判决 WER 计算 vs 模型输出）

## P21 给出的硬事实（train-00007，真实样本）

```
ref              : 这/方面/帮助/给/。      (target_ids 应为 [5,32,26,1,?])
argmax=blank 帧   : 44 / 48
label 是 top1 帧  : 10 / 48
label 概率 max    : 0.999998
hyp (greedy)     : 你/不行/时间/<unk>     <- 与 ref 毫无重叠
```

**矛盾**：若真有 10 帧的 argmax 等于 ref 里的 5 个 label
（每个 label 至少一帧），贪心折叠后必然输出 ref 的子序列。
但 hyp 与 ref 零重叠。

**唯一自洽的解释**：`b["targets"][r, :L]` 里存的 id 与
`voc.decode(...)` / `ref` 用的字符串**不是同一套索引**。
即：**我在 P21 里统计「label 是 top1」时用的 tgt 索引是错的**，
而 `decode_batch` 输出的（经 `classes_to_token_ids` 减 1）是对的。

## 关键怀疑：off-by-one

`classes_to_token_ids` 做 `index - 1`（CTC 类 0 是 blank，
词表索引 i 对应类 i+1）。若某处把 **1-D target id**（已是词表索引）
当成了 **2-D 类索引**（需要 +1 才是类），比较就会整体错位一格。

本脚本做**同一份 logits 下的三方对拍**：
  (1) CTC 类空间 argmax -> -1 -> token 字符串   [仓库 decode 路径]
  (2) b['targets'] 里的 id 直接当词表索引         [P21 统计用的路径]
  (3) CSV 原始 gloss
若 (1)≈(3) 而 (2)≠(3)，则 **P21 的「label top1」统计是错的**，
真实情况是「模型输出完全无关的类」-> 指向特征/归一化问题。

## 同时打印原始数值
因为 P20dbg4 已看到第 0 帧 argmax=249，
而 train-00004 的 target_ids[0]=249 —— 说明 (1)(2) 本应一致。
需要看全部 48 帧的 argmax 序列才能判断。

只读 train，**不触碰 dev 与 test**。
"""
import csv
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
from cslr.recognition.model import CTCConfig, CTCRecognizer, BLANK_INDEX
from cslr.recognition.gloss_sequence import build_ordered_vocabulary
from cslr.recognition.dataset import GlossSequenceDataset, FeatureNormalizer, collate_samples
from cslr.recognition.training import (
    ctc_loss, iterate_batches, resolve_device, to_torch_batch, decode_batch)
from p8_error_attribution import levenshtein


def read_split(split):
    table = {"train": "train.csv", "validation": "dev.csv"}
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


def main():
    device = resolve_device("auto")
    tr_labels = read_split("train")
    tr_root = ensure_link_dir("train")
    recs = []
    for sid, g in tr_labels.items():
        if not (tr_root / (sid + ".npy")).exists():
            continue
        if not [t.strip() for t in g.split("/") if t.strip()]:
            continue
        recs.append(SampleRecord(sample_id=sid, video=Path(sid + ".mp4"), label=g,
                                 signer="x", session="x", split="train"))
    recs.sort(key=lambda r: r.sample_id)
    cand = [r for r in recs if 3 <= len([t for t in r.label.split("/") if t.strip()]) <= 6]
    pick = cand[:8]
    voc, _ = build_ordered_vocabulary((g for g in tr_labels.values()),
                                      min_frequency=2, max_tokens=300)
    print("voc.index_of('这') =", voc.index_of("这"),
          "  voc.index_of('你') =", voc.index_of("你"))
    print("voc.tokens[:12] =", voc.tokens[:12])
    nrm = FeatureNormalizer.fit([np.load(tr_root / (r.sample_id + ".npy")) for r in pick])
    ds = GlossSequenceDataset(pick, tr_root, voc, nrm, feature_view="full")
    ref = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in pick}

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    cfg = CTCConfig(input_size=368, vocabulary_size=voc.size, hidden_size=256,
                    num_layers=2, dropout=0.0, bidirectional=True,
                    projection_size=256, subsample_stride=1)
    model = CTCRecognizer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for ep in range(400):
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

    model.eval()
    with torch.no_grad():
        for s in iterate_batches(ds, 8, shuffle=False, seed=42):
            b = to_torch_batch(collate_samples(s), device)
            lg = model(b["features"], b["input_lengths"])
            ol = model.output_lengths(b["input_lengths"])
            p = torch.softmax(lg.float(), dim=-1).cpu().numpy()
            lp = torch.log_softmax(lg.float(), dim=-1).cpu().numpy()
            dec, _, _ = decode_batch(lp, ol.cpu().numpy(), 1)

            for r in range(4):
                sid = b["sample_ids"][r]
                t = int(ol[r])
                L = int(b["target_lengths"][r])
                tgt = b["targets"][r, :L].tolist()
                am = p[r, :t].argmax(axis=1)
                print("=" * 76)
                print("{}  T={} L={}".format(sid[:12], t, L))
                print("  CSV ref        : {}".format("/".join(ref[sid])))
                print("  target_ids     : {}   (词表索引, L={})".format(tgt, L))
                print("  target 字符串   : {}".format(
                    "/".join(voc.tokens[i] if 0 < i < voc.size else "?" for i in tgt)))
                print("  decode_batch   : {}".format(
                    "/".join(voc.decode(list(dec[r]))) or "(空)"))
                print("  edit           = {}".format(levenshtein(ref[sid], voc.decode(list(dec[r])))[0]))
                print()
                print("  全部 48 帧 argmax（CTC 类空间）:")
                print("   ", am.tolist())
                nz = [(i, int(c), voc.tokens[c - 1] if 0 < c < voc.size else "?")
                      for i, c in enumerate(am) if c != BLANK_INDEX]
                print("  非 blank 帧 (帧号, 类, token):")
                for i, c, tok in nz:
                    inT = "∈target" if (c - 1) in tgt else ""
                    print("     frame {:2d}  class {:3d}  token {:<8s} {}".format(i, c, tok, inT))
                print()
                print("  target 各 id 在整条序列上的最大概率:")
                for tid in tgt:
                    cls = tid + 1
                    mx = float(p[r, :t, cls].max())
                    where = int(p[r, :t, cls].argmax())
                    print("     id {:3d} -> class {:3d} ({:<8s}) max_prob={:.6f} @frame {}".format(
                        tid, cls, voc.tokens[tid] if 0 < tid < voc.size else "?", mx, where))
            break


if __name__ == "__main__":
    main()
