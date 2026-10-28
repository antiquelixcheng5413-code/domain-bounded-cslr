#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P177: pixel-level comparison, front-end mp4 decode vs official_rgb jpg frames.

Answers two questions:
  1. Is VA.ToTensor output in [0,255] (no /255)?  -> stats
  2. Does mp4 decode need cvtColor(BGR2RGB) to match official_rgb frames?
"""
import sys, os, glob
from pathlib import Path
import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (REPO / "tools" / "blank_gov", REPO / "external" / "TFNet",
          REPO / "src", REPO / "app" / "backend"):
    sys.path.insert(0, str(p))

import torch
import cv2
import videoAugmentation as VA

SID = sys.argv[1] if len(sys.argv) > 1 else "dev-00001"
TR = "dev"

mp4s = sorted(glob.glob(str(REPO / f"data/raw/CE-CSL/video/dev/*/{SID}.mp4")))
mp4 = Path(mp4s[0])
print("mp4 =", mp4, "exists =", mp4.exists())

jpg_dirs = sorted(glob.glob(str(REPO / f"artifacts/official_rgb/dev/*/{SID}")))
jpg_dir = Path(jpg_dirs[0])
jpgs = sorted(glob.glob(str(jpg_dir / "*.jpg")))
print("jpg dir =", jpg_dir, "n =", len(jpgs))

# ---- A) official jpg path, RGB_FIX=True (no cvtColor) ----
A = [cv2.imread(f) for f in jpgs[:8]]
A = [cv2.resize(x, (256, 256)) for x in A]
A_fix = np.array([cv2.resize(cv2.imread(f), (256, 256)) for f in jpgs[:8]])
A_nor = np.array([cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), (256, 256))
                  for f in jpgs[:8]])

# ---- B) mp4 decode ----
cap = cv2.VideoCapture(str(mp4))
B_raw = []
while len(B_raw) < 8:
    ok, fr = cap.read()
    if not ok:
        break
    B_raw.append(cv2.resize(fr, (256, 256)))
cap.release()
B = np.array(B_raw)
B_rgb = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in B_raw])
print("mp4 frames read =", len(B))

def cmp(name, x, y):
    n = min(len(x), len(y))
    d = np.abs(x[:n].astype(np.int16) - y[:n].astype(np.int16))
    print(f"{name}: MAE={d.mean():.3f} max={d.max()}")

cmp("mp4_BGR  vs jpg_nocvt(RGB_FIX=True)", B, A_fix)
cmp("mp4_RGB  vs jpg_nocvt(RGB_FIX=True)", B_rgb, A_fix)
cmp("mp4_BGR  vs jpg_cvt  (RGB_FIX=False)", B, A_nor)
cmp("mp4_RGB  vs jpg_cvt  (RGB_FIX=False)", B_rgb, A_nor)
cmp("jpg_nocvt vs jpg_cvt", A_fix, A_nor)

# ---- C) ToTensor value range ----
import p78_train_official_tfnet as P78
P78.RGB_FIX = True
tf = P78.build_transforms(hflip=False, img_size=160, y_shift=0,
                          temporal_rescale=0.0)[1]
t = tf([cv2.imread(f) for f in jpgs[:4]])
print("after build_transforms[1] (CenterCropShift+ToTensor): dtype", t.dtype,
      "min", float(t.min()), "max", float(t.max()), "mean", float(t.mean()))
scale = t.float() / 127.5 - 1.0
print("after /127.5-1: min", float(scale.min()), "max", float(scale.max()),
      "mean", float(scale.mean()))
