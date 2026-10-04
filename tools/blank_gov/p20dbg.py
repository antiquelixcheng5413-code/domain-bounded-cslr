# -*- coding: utf-8 -*-
"""临时调试：打印 batch 各字段形状，定位 CTC loss 的 input_lengths 不匹配。"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.contracts import SampleRecord
from cslr.recognition.gloss_sequence import build_ordered_vocabulary
from cslr.recognition.dataset import GlossSequenceDataset, FeatureNormalizer, collate_samples
from cslr.recognition.training import iterate_batches, to_torch_batch, resolve_device


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
dev = resolve_device("auto")

for bi, s in enumerate(iterate_batches(ds, 8, shuffle=False, seed=42)):
    b = to_torch_batch(collate_samples(s), dev)
    n = len(s)
    print("batch {}: len(samples)={}".format(bi, n))
    for k in ("features", "input_lengths", "targets", "target_lengths"):
        v = b[k]
        print("   {:<16s} type={:<8s} shape={}".format(
            k, type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else "n/a"))
    print("   n(sample_ids)   =", len(b["sample_ids"]))
    print("   input_lengths   =", b["input_lengths"].tolist())
    print("   target_lengths  =", b["target_lengths"].tolist())
    print("   targets[0][:8]  =", b["targets"][0][:8].tolist())
    lp = torch.log_softmax(torch.randn(n, 48, voc.size + 1, device=dev), dim=-1)
    flat = []
    for r in range(n):
        L = int(b["target_lengths"][r])
        flat.extend(b["targets"][r, :L].tolist())
    ft = torch.tensor(flat, dtype=torch.long, device=dev)
    ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    try:
        lv = ctc(lp, ft, b["input_lengths"], b["target_lengths"])
        print("   CTC ok ->", tuple(lv.shape))
    except RuntimeError as e:
        print("   CTC FAIL:", e)
    if bi >= 1:
        break
