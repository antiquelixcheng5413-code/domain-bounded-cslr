# -*- coding: utf-8 -*-
"""终极定位：decode_batch 内部到底拿到了什么。

P20dbg2 事实：
  手工 arr[r, :t].argmax(axis=1) 折叠 -> `这/帮助/给/`（与 ref 完全一致）
  decode_batch(arr, ol, 1)          -> `你/不行/时间/<unk>`（全错）

greedy_decode 源码正确。唯一剩余解释：**decode_batch 传给 greedy_decode
的数组不是 arr[row, :length]**，或者 arr 的行顺序/内容与手工取的不一致。

本脚本 monkey-patch greedy_decode，把它实际收到的 shape / 前几个 argmax 打出来。
"""
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

import cslr.recognition.decode as D
from cslr.contracts import SampleRecord
from cslr.recognition.model import CTCConfig, CTCRecognizer, BLANK_INDEX
from cslr.recognition.gloss_sequence import build_ordered_vocabulary
from cslr.recognition.dataset import GlossSequenceDataset, FeatureNormalizer, collate_samples
from cslr.recognition.training import ctc_loss, iterate_batches, resolve_device, to_torch_batch

_orig = D.greedy_decode
CALLS = {"n": 0}


def traced(log_probs):
    CALLS["n"] += 1
    if CALLS["n"] <= 3:
        print("  >>> greedy_decode 第{}次收到 shape={}".format(CALLS["n"], log_probs.shape))
        print("      argmax(axis=1) 前 20 个: {}".format(
            log_probs.argmax(axis=1)[:20].tolist()))
        print("      每行最大值(前5行): {}".format(
            [round(float(log_probs[i].max()), 4) for i in range(min(5, log_probs.shape[0]))]))
    return _orig(log_probs)


D.greedy_decode = traced
import cslr.recognition.training as T
T.greedy_decode = traced


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
print("训练 400 轮 ...")
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
print("=" * 72)
print("decode_batch 内部追踪")
print("=" * 72)
with torch.no_grad():
    for samples in iterate_batches(ds, 8, shuffle=False, seed=42):
        b = to_torch_batch(collate_samples(samples), dev)
        logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1)
        arr = lp.cpu().numpy()
        print("lp.shape =", arr.shape, "  ol =", ol.cpu().numpy()[:8])
        ols = ol.cpu().numpy()
        print()
        print("对照：手工 arr[0, :ols[0]] 的 argmax 前 20 = {}".format(
            arr[0, :int(ols[0])].argmax(axis=1)[:20].tolist()))
        print()
        dec, _, _ = T.decode_batch(arr, ols, 1)
        print("decode_batch 第0条 =", voc.decode(list(dec[0])))
        print("手工折叠第0条      =", voc.decode(
            [int(c) - 1 for i, c in enumerate(arr[0, :int(ols[0])].argmax(axis=1))
             if c != BLANK_INDEX and (i == 0 or arr[0, :int(ols[0])].argmax(axis=1)[i - 1] != c)]))
        break
