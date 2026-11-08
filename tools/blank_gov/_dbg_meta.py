import torch

for t in ("official", "p184", "p184b"):
    d = torch.load("/home/su127/FYP/domain-bounded-cslr/artifacts/metrics/blank-gov/p188_logp_%s.pt" % t,
                   map_location="cpu", weights_only=False)
    print(t, d["img"], d["ckpt"], len(d["items"]))
