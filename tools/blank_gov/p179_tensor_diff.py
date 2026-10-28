#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P179: tensor-level diff, front-end _load_frames vs official RGBSeqDataset.

Same sid, same img_size, same weights -> compare:
  * shape (frame count)
  * per-pixel stats
  * decoded tokens / WER
"""
import os, sys, glob, csv
from pathlib import Path
import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (REPO / "tools" / "blank_gov", REPO / "external" / "TFNet",
          REPO / "src", REPO / "app" / "backend"):
    sys.path.insert(0, str(p))
os.environ.setdefault("CSLR_TFNET_DEVICE", "cuda")

import torch
import p78_train_official_tfnet as P78
import tfnet_service as TS

P78.RGB_FIX = True
P78.OFFICIAL_PAD = True

SID = sys.argv[1] if len(sys.argv) > 1 else "dev-00001"

labels = P78.read_csv_split("dev.csv")
w2i, n, i2w = P78.DPM.Word2Id(
    str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
    str(P78.CSV_DIR / "test.csv"), "CE-CSL")
tf = P78.build_transforms(hflip=False, img_size=160, y_shift=0,
                          temporal_rescale=0.0)[1]
ds = P78.RGBSeqDataset("dev", labels, w2i, tf, False)
pos = {it["sid"]: i for i, it in enumerate(ds.items)}
item = ds[pos[SID]]
off = item["video"] if isinstance(item, dict) else item
if not torch.is_tensor(off):
    off = torch.as_tensor(off)
print("official tensor:", tuple(off.shape), "min", float(off.min()),
      "max", float(off.max()), "mean", float(off.mean()))

svc = TS.create_vac_service()
svc.OFFICIAL_PAD = True
svc._P = P78                      # 需要 build_transforms / _conv_len
svc._idx2word = i2w
mp4 = glob.glob(str(REPO / f"data/raw/CE-CSL/video/dev/*/{SID}.mp4"))[0]
fe = svc._load_frames(Path(mp4))
fe = torch.as_tensor(fe)
print("frontend tensor:", tuple(fe.shape), "min", float(fe.min()),
      "max", float(fe.max()), "mean", float(fe.mean()))

T = min(off.shape[0], fe.shape[0])
d = (off[:T] - fe[:T]).abs()
print(f"frame count: official={off.shape[0]} frontend={fe.shape[0]}  (T used={T})")
print(f"pixel MAE over first {T} frames = {float(d.mean()):.3f}  max={float(d.max()):.1f}")
print("per-frame MAE (first 10):",
      [round(float(d[i].mean()), 2) for i in range(min(10, T))])
print("per-frame MAE (last 5):",
      [round(float(d[i].mean()), 2) for i in range(max(0, T - 5), T)])
