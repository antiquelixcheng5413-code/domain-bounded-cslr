# -*- coding: utf-8 -*-
"""定位 P20 矛盾：loss 极低但贪心解码全错。

已排除：仓库 ctc_loss 正确（与官方 1-D flatten 路径逐位一致）。

剩余怀疑点，按可能性排序：

- **(S1) decode_batch 收到的 log_probs 布局错误**
  它接收 `log_probs[row, :length]` 并在 axis=1 上做 argmax，
  若传入的是 [B,T,C] 则 row=B、length=T 恰好自洽 —— 但若上游传成 [T,B,C] 就全错。
  **必须核对 evaluate() 与 P20 里的实际布局。**
- **(S2) features 在 eval_detail 里被重复/错误归一化**
- **(S3) decode 用的 blank 索引与训练不一致**（都是 0，应无问题）

本脚本：用训练好的模型，同一份 logits，
  (a) 走仓库 decode_batch
  (b) 手工对 [B,T,C] 逐行 argmax 再折叠
若两者不一致 => S1 成立。
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


tr_labels = read_split("train")
tr_root = ensure_link_dir("train")
recs = []
for sid, g in tr_labels.items():
    if not (tr_root / (sid + ".npy")).exists():
        continue
    if not [t for t in g.split("/") if t.strip()]:
        continue
    recs.append(SampleRecord(sample_id=sid, video=Path(sid + ".mp4"), label=g,
                             signer="x", session="x", split="train"))
recs.sort(key=lambda r: r.sample_id)
cand = [r for r in recs if 3 <= len([t for t in r.label.split("/") if t.strip()]) <= 6]
pick = cand[:8]

voc, _ = build_ordered_vocabulary((g for g in tr_labels.values()),
                                  min_frequency=2, max_tokens=300)
nrm = FeatureNormalizer.fit([np.load(tr_root / (r.sample_id + ".npy")) for r in pick])
ds = GlossSequenceDataset(pick, tr_root, voc, nrm, feature_view="full")
ref = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in pick}
dev = resolve_device("auto")

random.seed(42); np.random.seed(42); torch.manual_seed(42)
cfg = CTCConfig(input_size=368, vocabulary_size=voc.size, hidden_size=256,
                num_layers=2, dropout=0.0, bidirectional=True,
                projection_size=256, subsample_stride=1)
model = CTCRecognizer(cfg).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

print("训练 8 条样本 400 轮（与 P20 相同超参）")
for ep in range(1, 401):
    model.train()
    for samples in iterate_batches(ds, 8, shuffle=False, seed=0):
        b = to_torch_batch(collate_samples(samples), dev)
        opt.zero_grad(set_to_none=True)
        logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        loss = ctc_loss(logits, b["input_lengths"], b["targets"],
                        b["target_lengths"], ol)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

model.eval()
print()
with torch.no_grad():
    for samples in iterate_batches(ds, 8, shuffle=False, seed=42):
        b = to_torch_batch(collate_samples(samples), dev)
        logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lv = ctc_loss(logits, b["input_lengths"], b["targets"],
                      b["target_lengths"], ol, reduction="none")
        lp = torch.log_softmax(logits.float(), dim=-1)   # [B,T,C]

        print("=" * 72)
        print("loss / P(target)  vs  解码")
        print("=" * 72)
        arr = lp.cpu().numpy()
        dec, blank_steps, total_steps = decode_batch(arr, ol.cpu().numpy(), 1)
        for r, ids in enumerate(dec):
            L = int(b["target_lengths"][r])
            nll = float(lv[r]) / max(L, 1)
            hyp = voc.decode(list(ids))
            r_ref = ref[b["sample_ids"][r]]
            t = int(ol[r])
            # 手工折叠 argmax
            am = arr[r, :t].argmax(axis=1)
            man = []
            prev = -1
            for c in am:
                if c != BLANK_INDEX and c != prev:
                    man.append(int(c))
                prev = c
            man_tok = voc.decode(man)
            print("  {}  L={}  loss/tok={:+.4f}  P={:.4f}".format(
                b["sample_ids"][r][:12], L, nll, np.exp(-nll)))
            print("     ref          : {}".format("/".join(r_ref)))
            print("     decode_batch : {}".format("/".join(hyp) or "(空)"))
            print("     手工 argmax   : {}".format("/".join(man_tok) or "(空)"))
            print("     target_ids   : {}".format(
                b["targets"][r, :L].tolist()))
            print("     手工解码 ids : {}".format(man))
            print("     blank 帧 {}/{}  非blank最大后验={:.4f}".format(
                blank_steps, total_steps, float(arr[r, :t, 1:].max())))
        break
