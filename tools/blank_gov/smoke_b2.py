"""P107 B2 冒烟测试：验证 y 传 1d 类标的修复是否成立，并外推全量耗时。

用法（WSL 内）：
  venv/bin/python tools/blank_gov/smoke_b2.py <N_videos>
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/su127/FYP/domain-bounded-cslr")
FEATS = ROOT / "artifacts/frame_feats/train"
CSV = ROOT / "data/raw/CE-CSL/label/train.csv"

n_vid = int(sys.argv[1]) if len(sys.argv) > 1 else 200

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
X, y, grp = [], [], []
t_load = time.time()
used = 0
for r in rows:
    if used >= n_vid:
        break
    f = FEATS / (r["Number"] + ".npy")
    if not f.exists():
        continue
    a = np.load(f).astype(np.float32)
    g = r["Gloss"].split("/")
    T_ = a.shape[0]
    for i, w in enumerate(g):
        s = int(i * T_ / len(g))
        e = max(s + 1, int((i + 1) * T_ / len(g)))
        seg = a[s:e]
        X.append(np.concatenate([seg.mean(0), seg.std(0)]))
        y.append(w)
        grp.append(r["Number"])
    used += 1

X = np.stack(X)
y_arr = np.asarray(y)
print("加载 %.1fs  视频=%d  片段=%d  维度=%d  distinct类=%d"
      % (time.time() - t_load, used, len(X), X.shape[1], len(set(y))))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

gv = sorted(set(grp))
half = set(gv[: len(gv) // 2])
tr = np.array([g in half for g in grp])
te = ~tr

lb = LabelBinarizer().fit(y)
Y = lb.transform(y)
print("one-hot Y shape:", Y.shape)

t = time.time()
clf = LogisticRegression(max_iter=300, n_jobs=-1)
clf.fit(X[tr], y_arr[tr])          # 修复点：1d 类标
t_fit = time.time() - t
print("✅ fit 成功，耗时 %.1fs" % t_fit)

t = time.time()
P = clf.predict_proba(X[te])
auc = roc_auc_score(Y[te], P, average="macro", multi_class="ovr")
print("✅ AUC 计算成功 %.1fs  macroAUC=%.4f" % (time.time() - t, auc))

print()
print("=== 外推全量 4973 视频 ===")
scale = 4973 / used
print("fit  %.0fs → 约 %.1f 分钟" % (t_fit, t_fit * scale / 60))
