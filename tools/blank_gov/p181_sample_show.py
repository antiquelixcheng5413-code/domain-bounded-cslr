#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P181: show 5 representative dev outputs (ref vs hyp), and re-run live.

Picks 1 best / 2 median / 2 worst from the cached P178 token dump, then
re-runs those 5 videos through the live service to confirm reproducibility.
"""
import os, sys, csv, glob, json
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (REPO / "tools" / "blank_gov", REPO / "src", REPO / "app" / "backend"):
    sys.path.insert(0, str(p))
os.environ.setdefault("CSLR_TFNET_DEVICE", "cuda")

import tfnet_service as TS
from official_wer import wer_official

CACHE = REPO / "tools/blank_gov/p178_tokens_vac.json"
store = json.loads(CACHE.read_text(encoding="utf-8"))

ref_map = {}
with open(REPO / "data/raw/CE-CSL/label/dev.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ref_map[row["Number"]] = row["Gloss"]

videos = sorted(glob.glob(str(REPO / "data/raw/CE-CSL/video/dev/*/*.mp4")))
vmap = {Path(v).stem: v for v in videos}

scored = []
for sid, hyp in store.items():
    if sid not in ref_map or sid not in vmap:
        continue
    ref = ref_map[sid]
    w = wer_official([ref], [hyp])[0]
    scored.append((w, sid, ref, hyp))
scored.sort(key=lambda x: x[0])

n = len(scored)
picks = [scored[0], scored[n // 3], scored[n // 2], scored[-2], scored[-1]]

print(f"dev total = {n}\n")
svc = TS.create_vac_service()

for w, sid, ref, hyp in picks:
    print("=" * 70)
    print(f"{sid}   cached WER = {w:.1f}%")
    live = svc.predict_video(Path(vmap[sid]))
    lh = live.get("gloss_tokens") or []
    lw = wer_official([ref], [lh])[0]
    print(f"  REF : {ref}")
    print(f"  HYP : {'/'.join(hyp) if hyp else '(空)'}")
    print(f"  HYP(live, conf={live.get('confidence', 0):.3f}): {'/'.join(lh) if lh else '(空)'}")
    print(f"  live WER = {lw:.1f}%   {'一致' if lh == hyp else '与缓存不同'}")
