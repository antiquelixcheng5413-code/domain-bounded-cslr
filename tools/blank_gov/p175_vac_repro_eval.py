#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P175: reproduce VAC(P150) dev WER on official path (OFFICIAL_PAD=True),
and quantify pure single-period (。) collapse + sample-level mean WER.

Robust: per-video try/except; incremental JSON checkpoint every 50 videos.
"""
from __future__ import annotations
import os, sys, csv, glob, json, time
from pathlib import Path

ROOT = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (ROOT / "tools" / "blank_gov", ROOT / "src", ROOT / "app" / "backend"):
    sys.path.insert(0, str(p))
os.environ.setdefault("CSLR_TFNET_DEVICE", "cuda")

import torch
import tfnet_service as TS
from official_wer import wer_official, diagnostics

DEVICE = os.getenv("CSLR_TFNET_DEVICE", "cuda")
VID_DIR = ROOT / "data/raw/CE-CSL/video/dev"
CSV = ROOT / "data/raw/CE-CSL/label/dev.csv"
CKPT_JSON = ROOT / "tools/blank_gov/p175_result.json"

ref_map = {}
with open(CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ref_map[row["Number"]] = row["Gloss"]

videos = sorted(glob.glob(str(VID_DIR / "**" / "*.mp4"), recursive=True))
LIMIT = int(os.getenv("P175_LIMIT", "0")) or len(videos)


def run(pad, limit, svc):
    TS.VacTrainedService.OFFICIAL_PAD = pad
    hyps, refs = [], []
    pure_period = 0
    empty = 0
    n = min(limit, len(videos))
    per_sent_wer = []
    for i, v in enumerate(videos[:n], 1):
        num = Path(v).stem
        try:
            res = svc.predict_video(Path(v))
            toks = res.get("gloss_tokens") or []
        except Exception as exc:
            toks = []
            print(f"  [ERR] {num}: {type(exc).__name__}: {exc}", flush=True)
        hyps.append(toks)
        refs.append(ref_map.get(num, ""))
        if toks == ["。"]:
            pure_period += 1
        if not toks:
            empty += 1
        # sample-level WER for this one sentence
        try:
            w = wer_official([refs[-1]], [toks])[0]
            per_sent_wer.append(w)
        except Exception:
            pass
        if i % 50 == 0 or i == n:
            wer, e, tot = wer_official(refs, hyps)
            snap = {"pad": pad, "done": i, "n": n, "wer": wer,
                    "pure_period": pure_period, "empty": empty,
                    "sample_mean_wer": (sum(per_sent_wer)/len(per_sent_wer) if per_sent_wer else None),
                    "diag": diagnostics(refs, hyps)}
            CKPT_JSON.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{pad}] {i}/{n} corpus_wer={wer:.2f}% pure={pure_period}({pure_period/i*100:.1f}%) empty={empty} smean={snap['sample_mean_wer']}", flush=True)
    wer, e, tot = wer_official(refs, hyps)
    diag = diagnostics(refs, hyps)
    smean = sum(per_sent_wer)/len(per_sent_wer) if per_sent_wer else None
    print(f"\n=== OFFICIAL_PAD={pad} (n={n}) ===")
    print(f"  corpus WER         : {wer:.2f}%  (ed={e}, ref_tokens={tot})")
    print(f"  纯单句号 [。]      : {pure_period}/{n} = {pure_period/n*100:.1f}%")
    print(f"  空输出             : {empty}/{n} = {empty/n*100:.1f}%")
    print(f"  有内容样本         : {n-pure_period-empty}/{n} = {(n-pure_period-empty)/n*100:.1f}%")
    print(f"  逐句平均 WER       : {smean:.2f}%  (多数样本的真实水准)")
    print(f"  diagnostics        : {json.dumps(diag, ensure_ascii=False)}")
    return {"pad": pad, "n": n, "wer": wer, "pure_period": pure_period,
            "empty": empty, "content": n-pure_period-empty,
            "sample_mean_wer": smean, "diag": diag}


if __name__ == "__main__":
    print(f"device={DEVICE} total_dev={len(videos)} limit={LIMIT}", flush=True)
    svc = TS.VacTrainedService(device=DEVICE)
    svc._ensure_loaded()
    print("model loaded.", flush=True)
    res = [run(True, LIMIT, svc)]
    print("\n==== SUMMARY ====")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    CKPT_JSON.write_text(json.dumps({"final": res}, ensure_ascii=False, indent=2), encoding="utf-8")
