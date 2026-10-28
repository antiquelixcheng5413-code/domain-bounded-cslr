#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P176: reproduce P150 official eval path on p150-rgbfix-best.pt.
Uses p78 RGBSeqDataset (artifacts/official_rgb frames) + _dev_eval.
Decisive test: model problem vs front-end problem.
"""
import sys, os
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (REPO/"tools"/"blank_gov", REPO/"external"/"TFNet", REPO/"src", REPO/"app"/"backend"):
    sys.path.insert(0, str(p))
os.environ.setdefault("CSLR_TFNET_DEVICE", "cuda")

import torch
import p78_train_official_tfnet as P78
from p78_train_official_tfnet import (RGBSeqDataset, build_transforms, collate,
                                      _dev_eval, Net)
from torch.utils.data import DataLoader

P78.OFFICIAL_PAD = True
P78.RGB_FIX = True
P78.BLANK_BIAS = 0.0

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DPM = P78.DPM
CSV_DIR = P78.CSV_DIR

print("device =", DEV, flush=True)
labels_dv = P78.read_csv_split("dev.csv")
word2idx, wordSetNum, idx2word = DPM.Word2Id(
    str(CSV_DIR/"train.csv"), str(CSV_DIR/"dev.csv"), str(CSV_DIR/"test.csv"), "CE-CSL")
print("wordSetNum =", wordSetNum, flush=True)

dv_tf = build_transforms(hflip=False, img_size=160, y_shift=0,
                         temporal_rescale=0.0)[1]
dv_ds = RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
print("dev items =", len(dv_ds), flush=True)

mdl = Net.moduleNet(1024, wordSetNum+1, "VAC", DEV, "CE-CSL", True)
ckpt_path = REPO/"artifacts/checkpoints/p150-rgbfix-best.pt"
ckpt = torch.load(str(ckpt_path), map_location=DEV, weights_only=False)
sd = ckpt.get("model_state", ckpt)
mdl.load_state_dict(sd, strict=False)
mdl = mdl.to(DEV).eval()
print("model loaded from", ckpt_path, flush=True)

dv_loader = DataLoader(dv_ds, batch_size=2, shuffle=False, collate_fn=collate,
                       num_workers=0)
res = _dev_eval(mdl, dv_loader, DEV, idx2word, labels_dv)
print("==== OFFICIAL-PATH dev WER (P150 config) ====")
print("WER =", res)
