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
print("shape", lp.shape, "row sum exp", np.exp(lp[0]).sum())
print("frame0 top5", np.sort(lp[0])[-5:], "blank", lp[0,0])
arg = lp.argmax(1)
print("blank frames", int((arg==0).sum()), "/", len(arg))
amb = np.exp(lp).max(1)
print("mean max prob", amb.mean(), "min", amb.min())
