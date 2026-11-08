import torch
import numpy as np

P = "/home/su127/FYP/domain-bounded-cslr/"
d = torch.load(P + "artifacts/metrics/blank-gov/p188_logp_p184.pt",
               map_location="cpu", weights_only=False)
print("p188 dump type:", type(d))
if isinstance(d, dict):
    print("keys:", list(d.keys())[:12])
    for k, v in list(d.items())[:6]:
        shp = getattr(v, "shape", None)
        print("  ", k, type(v).__name__, shp if shp else
              (len(v) if hasattr(v, "__len__") else v))
elif isinstance(d, (list, tuple)):
    print("len:", len(d))
    for i, v in enumerate(d[:3]):
        print("  ", i, type(v).__name__, getattr(v, "shape", None))
else:
    print("shape:", getattr(d, "shape", None))

x = np.load(P + "data/processed/ce-csl-landmark48/dev-00001.npy")
print("landmark shape:", x.shape, x.dtype, "finite:", np.isfinite(x).all())
