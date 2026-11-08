import sys
from collections import Counter
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))
import p78_train_official_tfnet as P78  # noqa

tr = P78.read_csv_split("train.csv")
word2idx, wordSetNum, idx2word = P78.DPM.Word2Id(
    str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
    str(P78.CSV_DIR / "test.csv"), "CE-CSL")

miss = Counter()
for _sid, (_p, seq) in tr.items():
    for t in str(seq).split("/"):
        if t and word2idx.get(t) is None:
            miss[t] += 1
print("distinct missing:", len(miss), "total occurrences:", sum(miss.values()))
print("top20:", miss.most_common(20))
print("sample sid/seq:", list(tr.items())[0][0], list(tr.items())[0][1][1][:80])
