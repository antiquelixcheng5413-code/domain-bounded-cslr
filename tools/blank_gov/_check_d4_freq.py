import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78  # noqa
import official_wer as OW  # noqa
import numpy as np  # noqa

tr = P78.read_csv_split("train.csv")
word2idx, wordSetNum, idx2word = P78.DPM.Word2Id(
    str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
    str(P78.CSV_DIR / "test.csv"), "CE-CSL")
print("wordSetNum:", wordSetNum, "word2idx type:", type(word2idx).__name__,
      "len:", len(word2idx))

cnt = np.zeros(wordSetNum + 1, dtype=np.float64)
miss = 0
for _sid, (_p, seq) in tr.items():
    for t in str(seq).split("/"):
        if not t:
            continue
        gi = word2idx.get(t)
        if gi is not None and 0 <= gi < len(cnt):
            cnt[gi] += 1.0
        else:
            miss += 1
print("train samples:", len(tr), "nonzero classes:", int((cnt > 0).sum()),
      "median freq:", float(np.median(cnt[cnt > 0])),
      "min:", float(cnt[cnt > 0].min()), "max:", float(cnt.max()),
      "missing tokens:", miss)

# 低频桶占比（dev token 中 train 频次 <=10）
dv = P78.read_csv_split("dev.csv")
low = tot = 0
for _sid, (_p, seq) in dv.items():
    for t in OW.split_glued([x for x in str(seq).split("/") if x]):
        gi = word2idx.get(t)
        f = cnt[gi] if (gi is not None and 0 <= gi < len(cnt)) else 0
        tot += 1
        low += (f <= 10)
print("dev tokens:", tot, "low-freq(<=10):", low,
      "ratio:", round(low / max(tot, 1), 4))
