#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P180: verify the oversized-video guard in tfnet_service._load_frames."""
import sys, glob, time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (REPO / "tools" / "blank_gov", REPO / "external" / "TFNet",
          REPO / "src", REPO / "app" / "backend"):
    sys.path.insert(0, str(p))

import tfnet_service as TS
import p78_train_official_tfnet as P78

svc = TS.VacTrainedService.__new__(TS.VacTrainedService)
svc.CHANNEL_FIX = True
svc.PIXEL_SCALE = "pm1"
svc.img_size = 160
svc._P = P78

vs = sorted(glob.glob(str(REPO / "data/raw/CE-CSL/video/dev/*/*.mp4")))
for idx in (int(sys.argv[1]) if len(sys.argv) > 1 else 402,):
    v = vs[idx]
    mb = Path(v).stat().st_size / 1e6
    print(f"idx={idx} {Path(v).name} {mb:.1f} MB", flush=True)
    t = time.time()
    s = svc._load_frames(Path(v))
    print(f"  -> frames={s.shape} min={s.min():.3f} max={s.max():.3f} "
          f"in {time.time()-t:.1f}s", flush=True)
