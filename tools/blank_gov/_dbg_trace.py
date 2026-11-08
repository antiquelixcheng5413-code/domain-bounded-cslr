import sys
from pathlib import Path
REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
import torch, numpy as np
import p78_train_official_tfnet as P

blob = torch.load(str(REPO/"artifacts/metrics/blank-gov/p188_logp_official.pt"),
                  map_location="cpu", weights_only=False)["items"]
sid = sorted(blob)[0]
lp = blob[sid]["logp"][:P._conv_len(blob[sid]["dl"])].numpy().astype(np.float32)
blank = 0
topk = 32
beam = 10
idx = np.argpartition(-lp, topk, axis=1)[:, :topk]
beams = {(): (0.0, -np.inf)}

for t in range(lp.shape[0]):
    cand = lp[t, idx[t]]
    ks = idx[t].tolist()
    new = {}

    def put(pref, pb, pnb):
        cur = new.get(pref)
        if cur is None:
            new[pref] = [pb, pnb]
        else:
            cur[0] = np.logaddexp(cur[0], pb)
            cur[1] = np.logaddexp(cur[1], pnb)

    for pref, (pb, pnb) in beams.items():
        tot = np.logaddexp(pb, pnb)
        put(pref, tot + lp[t, blank], -np.inf)
        last = pref[-1] if pref else -1
        for c, v in zip(ks, cand):
            if c == blank:
                continue
            if c == last:
                put(pref, -np.inf, pnb + v)
                put(pref + (c,), -np.inf, pb + v)
            else:
                put(pref + (c,), -np.inf, tot + v)
    scored = sorted(new.items(), key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1]))
    beams = {k: (v[0], v[1]) for k, v in scored[:beam]}
    top = scored[0]
    print("t=%d p_blank=%.3f  best_len=%d best=%.3f  empty=%.3f  nprefix=%d"
          % (t, float(np.exp(lp[t, blank])), len(top[0]),
             float(np.logaddexp(top[1][0], top[1][1])),
             float(np.logaddexp(*beams.get((), [-np.inf, -np.inf])) if () in beams else -99),
             len(new)))
