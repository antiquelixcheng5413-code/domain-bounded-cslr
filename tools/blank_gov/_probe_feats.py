import numpy as np
from pathlib import Path
from collections import Counter

P = Path("/home/su127/FYP/domain-bounded-cslr/data/processed")
for name in ("ce-csl-qwenvl48", "ce-csl-landmark96", "ce-csl-landmark48"):
    d = P / name
    fs = list(d.rglob("*.npy"))
    if not fs:
        print(name, "EMPTY")
        continue
    pref = Counter()
    for f in fs:
        s = f.stem.split("-")[0]
        pref[s] += 1
    f = fs[0]
    x = np.load(f)
    print(name, "n=", len(fs), "split=", dict(pref),
          "shape=", x.shape, "dtype=", x.dtype,
          "finite=", bool(np.isfinite(x).all()),
          "absmax=", float(np.abs(x).max()))
