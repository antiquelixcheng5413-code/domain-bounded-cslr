#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P178: verify front-end service on dev (resumable, chunked).

Runs videos [START, END) through a service, merges per-video tokens into a
JSON file, then reports corpus WER over everything accumulated so far.

Env:
    P178_SVC   = vac | tfnet        (default vac)
    P178_PAD   = 1 | 0              (default: service default)
    P178_START = int                (default 0)
    P178_END   = int                (default 50)
"""
import os, sys, csv, glob, json, time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (REPO / "tools" / "blank_gov", REPO / "src", REPO / "app" / "backend"):
    sys.path.insert(0, str(p))
os.environ.setdefault("CSLR_TFNET_DEVICE", "cuda")

import tfnet_service as TS
from official_wer import wer_official

SVC = os.getenv("P178_SVC", "vac").lower()
START = int(os.getenv("P178_START", "0"))
END = int(os.getenv("P178_END", "50"))
PAD = os.getenv("P178_PAD", "")

OUT = REPO / f"tools/blank_gov/p178_tokens_{SVC}.json"

svc = TS.create_vac_service() if SVC == "vac" else TS.create_tfnet_service()
if PAD in ("0", "1"):
    svc.OFFICIAL_PAD = (PAD == "1")

print(f"service={type(svc).__name__} scale={svc.PIXEL_SCALE} "
      f"chfix={svc.CHANNEL_FIX} pad={svc.OFFICIAL_PAD} "
      f"range=[{START},{END})", flush=True)

ref_map = {}
with open(REPO / "data/raw/CE-CSL/label/dev.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ref_map[row["Number"]] = row["Gloss"]

videos = sorted(glob.glob(str(REPO / "data/raw/CE-CSL/video/dev/*/*.mp4")))
chunk = videos[START:END]

store = {}
if OUT.exists():
    store = json.loads(OUT.read_text(encoding="utf-8"))

t0 = time.time()
for i, v in enumerate(chunk):
    num = Path(v).stem
    if num in store:
        continue
    try:
        store[num] = svc.predict_video(Path(v)).get("gloss_tokens") or []
    except Exception as exc:
        print(f"  [ERR] {num}: {type(exc).__name__}: {exc}", flush=True)
        store[num] = []
    if (i + 1) % 5 == 0:
        OUT.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
        print(f"  ..{i+1}/{len(chunk)} saved={len(store)} {time.time()-t0:.0f}s", flush=True)
OUT.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")

# ---- report over everything accumulated (aligned to sorted video order) ----
sids = [Path(v).stem for v in videos]
have = [s for s in sids if s in store]
hyps = [store[s] for s in have]
refs = [ref_map.get(s, "") for s in have]

n = len(have)
pure = sum(1 for h in hyps if h == ["。"])
empty = sum(1 for h in hyps if not h)
res = wer_official(refs, hyps)
print("=" * 60)
print(f"SVC={SVC} pad={svc.OFFICIAL_PAD} scale={svc.PIXEL_SCALE} chfix={svc.CHANNEL_FIX}")
print(f"accumulated  = {n}/{len(sids)} dev videos")
print(f"corpus WER   = {res[0]:.2f}%")
print(f"pure '。'    = {pure}/{n} = {100.0*pure/n:.1f}%")
print(f"empty        = {empty}/{n}")
print(f"distinct hyp = {len(set(tuple(h) for h in hyps))}")
print(f"chunk took   = {time.time()-t0:.1f}s")
