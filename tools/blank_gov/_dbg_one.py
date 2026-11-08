#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单样本对拍：文件里的 beam_decode vs 干净的参考实现。"""
import sys
sys.path.insert(0, "/home/su127/FYP/domain-bounded-cslr/tools/blank_gov")
import numpy as np
import torch
import p78_train_official_tfnet as P
from p189_beam_eval import beam_decode as bd_file

DUMP = "/home/su127/FYP/domain-bounded-cslr/artifacts/metrics/blank-gov/p188_logp_official.pt"


def bd_ref(logp, length, beam=10, topk=32, blank=0):
    lp = np.asarray(logp[:length], dtype=np.float32)
    pos = np.argpartition(-lp, topk - 1, axis=1)[:, :topk]
    beams = {(): (0.0, -np.inf)}
    for t in range(lp.shape[0]):
        new = {}

        def add(pref, pb_new, pnb_new):
            cur = new.get(pref)
            if cur is None:
                new[pref] = [pb_new, pnb_new]
            else:
                cur[0] = np.logaddexp(cur[0], pb_new)
                cur[1] = np.logaddexp(cur[1], pnb_new)

        lb = float(lp[t, blank])
        for pref, (pb, pnb) in beams.items():
            tot = float(np.logaddexp(pb, pnb))
            add(pref, tot + lb, -np.inf)
            last = pref[-1] if pref else -1
            for c in pos[t].tolist():
                if c == blank:
                    continue
                v = float(lp[t, c])
                if c == last:
                    add(pref, -np.inf, pnb + v)
                    add(pref + (c,), -np.inf, pb + v)
                else:
                    add(pref + (c,), -np.inf, tot + v)
        scored = sorted(new.items(), key=lambda kv: -float(np.logaddexp(kv[1][0], kv[1][1])))
        if t < 4 or t > lp.shape[0] - 3:
            print("[REF] t=%02d p_blank=%.5f best_len=%d tot=%.3f n_new=%d"
                  % (t, np.exp(lb), len(scored[0][0]),
                     float(np.logaddexp(scored[0][1][0], scored[0][1][1])),
                     len(scored)), flush=True)
        beams = {k: (v[0], v[1]) for k, v in scored[:beam]}
    best = max(beams.items(), key=lambda kv: float(np.logaddexp(kv[1][0], kv[1][1])))
    return list(best[0])


blob = torch.load(DUMP, map_location="cpu", weights_only=False)["items"]
sid = sorted(blob)[0]
rec = blob[sid]
lp = rec["logp"]
L = int(P._conv_len(rec["dl"]))
print("sid", sid, "logp", tuple(lp.shape), lp.dtype, "dl", rec["dl"], "L", L)

x = lp[:L].float().numpy()
am = x.argmax(1)
print("argmax 0-ratio %.3f" % float((am == 0).mean()), "mean p_blank %.4f" % float(np.exp(x[:, 0]).mean()))
print("sum exp per frame (first5)", np.exp(x[:5]).sum(1))

out_file = bd_file(lp, L, list(range(1 << 20)), beam=1, topk=32)
print("FILE beam1 len", len(out_file))
out_ref = bd_ref(x, L, beam=1, topk=32)
print("REF  beam1 len", len(out_ref))
