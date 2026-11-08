import sys
from pathlib import Path
import numpy as np

R = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (R / "tools/blank_gov", R / "external/TFNet", R / "src"):
    sys.path.insert(0, str(p))
import p78_train_official_tfnet as P78  # noqa
import DataProcessMoudle as DPM  # noqa

tr = P78.read_csv_split("train.csv")
w2i, n, i2w = P78.DPM.Word2Id(
    str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
    str(P78.CSV_DIR / "test.csv"), "CE-CSL")
c = np.zeros(n + 1)
for sid, (p, seq) in tr.items():
    for t in DPM.PreWords(str(seq).split("/")):
        g = w2i.get(t)
        if g is not None:
            c[g] += 1
c = c[1:]
print("classes:", len(c), "total tokens:", int(c.sum()))
for q in (1, 2, 3, 5, 10, 20, 50, 100):
    k = int((c <= q).sum())
    print("  freq<=%3d: %4d classes (%5.1f%%)" % (q, k, 100.0 * k / len(c)))
print("  freq==0:", int((c == 0).sum()),
      " median:", float(np.median(c[c > 0])), " max:", int(c.max()))
print("  top-10 token share: %.1f%%" % (100 * np.sort(c)[-10:].sum() / c.sum()))
print("  top-1%% classes token share: %.1f%%" %
      (100 * np.sort(c)[-max(1, len(c) // 100):].sum() / c.sum()))
