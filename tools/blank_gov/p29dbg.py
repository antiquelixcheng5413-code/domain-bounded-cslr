# -*- coding: utf-8 -*-
"""临时：查清 P29 自检为何两个配置 WER 完全相同（0.3421）。"""
import sys

sys.path.insert(0, "/home/su127/FYP/domain-bounded-cslr/src")
sys.path.insert(0, "/home/su127/FYP/domain-bounded-cslr/tools/blank_gov")
sys.argv = ["x"]

import csv
import random
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
import p29_stgcn as P
from stgcn import DualStreamCTCRecognizer

tr_labels = P.read_split("train")
tr_root = P.ensure_link_dir("train")
recs = P.make_records(tr_labels, tr_root, "train")
cand = [r for r in recs if 3 <= len([t for t in r.label.split("/") if t.strip()]) <= 6][:8]
voc, _ = __import__("cslr.recognition.gloss_sequence", fromlist=["x"]).build_ordered_vocabulary(
    (g for g in tr_labels.values()), min_frequency=2, max_tokens=300)
nrm = __import__("cslr.recognition.dataset", fromlist=["x"]).FeatureNormalizer.fit(
    [np.load(tr_root / (r.sample_id + ".npy")) for r in cand])

from cslr.recognition.dataset import GlossSequenceDataset, collate_samples
from cslr.recognition.training import iterate_batches, to_torch_batch, decode_batch, ctc_loss

ds = GlossSequenceDataset(cand, tr_root, voc, nrm, feature_view="full")

print("=" * 74)
print("关键：自检里 ref 用的是哪一种？")
print("=" * 74)
for r in cand[:3]:
    csv_ref = [t.strip() for t in r.label.split("/") if t.strip()]
    folded = [voc.tokens[i] for i in voc.encode(r.label)]
    print("  CSV ref : {}".format("/".join(csv_ref)))
    print("  折叠后  : {}".format("/".join(folded)))
    same = csv_ref == folded
    print("  一致?   {}".format(same))
    if not same:
        print("    ^ 不一致 -> 自检 WER 用 CSV ref 而训练用折叠 target，永远对不上")

print()
print("=" * 74)
print("P20 的做法（能过自检）用的是 folded ref，P29 误用了 CSV ref")
print("=" * 74)
folded_ref = {r.sample_id: [voc.tokens[i] for i in voc.encode(r.label)] for r in cand}
csv_ref = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in cand}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for tag, ref in (("CSV ref（错的）", csv_ref), ("folded ref（对的）", folded_ref)):
    for use_skel in (False, True):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        m = DualStreamCTCRecognizer(368, voc.size, dropout=0.0, use_skeleton=use_skel).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
        for ep in range(200):
            m.train()
            for s in iterate_batches(ds, 8, shuffle=False, seed=0):
                b = to_torch_batch(collate_samples(s), device)
                opt.zero_grad(set_to_none=True)
                lg = m(b["features"], b["input_lengths"])
                ol = m.output_lengths(b["input_lengths"])
                loss = ctc_loss(lg, b["input_lengths"], b["targets"], b["target_lengths"], ol)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
        r1 = P.evaluate(m, ds, device, 8, voc, ref)
        print("  {:<16s} use_skel={:<5s} WER={:.4f} n_distinct={}".format(
            tag, str(use_skel), r1["wer"], r1["n_distinct"]))
