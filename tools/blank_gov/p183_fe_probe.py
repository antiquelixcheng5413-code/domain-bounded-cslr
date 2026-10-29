#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P183-frontend-probe: 直接调 VacTrainedService.predict_video 验证 BLANK_BIAS 接入。
跳过缓存，实时解码 N 条 dev，输出 corpus WER + 单句号率。
"""
import sys, os, glob, json
from pathlib import Path
REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (str(REPO/"app"/"backend"), str(REPO/"external"/"TFNet"), str(REPO/"src")):
    sys.path.insert(0, p)
os.environ.setdefault("CSLR_TFNET_DEVICE", "cuda")

import p78_train_official_tfnet as P
import tfnet_service as TS

N = int(os.environ.get("P183FE_N", "20"))
svc = TS.create_vac_service()
print("VacTrainedService.BLANK_BIAS =", svc.BLANK_BIAS, flush=True)

labels = P.read_csv_split("dev.csv")
vids = sorted(glob.glob(str(REPO / "data/raw/CE-CSL/video/dev/*/*.mp4")))
if N > 0:
    vids = vids[:N]

refs, hyps = [], []
sp = 0
for i, vp in enumerate(vids):
    sid = Path(vp).stem
    ref = labels[sid][1]
    try:
        r = svc.predict_video(Path(vp))
    except Exception as e:
        print("ERR", sid, e, flush=True)
        continue
    hyp = r["gloss_tokens"]
    refs.append(ref); hyps.append(hyp)
    if hyp == ["。"]:
        sp += 1
    print(f"[{i:2d}] {sid} ref={ref.split('/')} hyp={hyp} conf={r.get('confidence')}", flush=True)

from official_wer import evaluate
o = evaluate(refs, hyps)
print(f"\n==== VAC front-end (bias={svc.BLANK_BIAS}) N={len(refs)} ====")
print(f"corpus WER = {o['WER_official']:.2f}%  (ref_tokens={o['ref_tokens']})")
print(f"pure '。' = {sp}/{len(refs)} = {sp/len(refs)*100:.1f}%")
