import torch

P = "/home/su127/FYP/domain-bounded-cslr/"
d = torch.load(P + "artifacts/metrics/blank-gov/p188_logp_p184.pt",
               map_location="cpu", weights_only=False)
items = d["items"]
print("items len:", len(items))
k0 = list(items.keys())[0]
print("first key:", k0, "type:", type(items[k0]))
v = items[k0]
if isinstance(v, dict):
    print("item keys:", list(v.keys()))
    for kk, vv in v.items():
        print("  ", kk, type(vv).__name__, getattr(vv, "shape", None))
else:
    print("value type:", type(v), getattr(v, "shape", None))
