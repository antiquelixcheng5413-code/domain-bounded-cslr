"""P107 B2 诊断：macroAUC 在 3841 类规模下是否可计算？若不可，量化替代品。

产出：
1. 类频次 vs 片段数分布
2. train/test 切分后，测试侧每类样本数的分布（判断 AUC 可算性）
3. Cohen's d（零号自检，不依赖分类器）
4. 不同频次阈值下，可评估类的数量与覆盖的片段数
"""
import csv
import collections
from pathlib import Path

import numpy as np

ROOT = Path("/home/su127/FYP/domain-bounded-cslr")
FEATS = ROOT / "artifacts/frame_feats/train"
CSV = ROOT / "data/raw/CE-CSL/label/train.csv"

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
X, y, grp = [], [], []
for r in rows:
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

X = np.stack(X)
y_arr = np.asarray(y)
print("片段总数 = %d   维度 = %d   distinct 类 = %d" % (len(X), X.shape[1], len(set(y))))

# train/test 按视频对半切（与原脚本一致）
gv = sorted(set(grp))
half = set(gv[: len(gv) // 2])
tr = np.array([g in half for g in grp])
te = ~tr

cnt_all = collections.Counter(y_arr)
cnt_te = collections.Counter(y_arr[te])
cnt_tr = collections.Counter(y_arr[tr])

print()
print("=== 测试侧每类样本数分布（决定 AUC 能否定义）===")
dist = collections.Counter(cnt_te[c] for c in cnt_all)
tot_c = len(cnt_all)
run = 0
for k in sorted(dist):
    run += dist[k]
    print("  测试侧只有 %2d 个样本的类: %5d 个类 (占 %.1f%%)" % (k, dist[k], 100 * dist[k] / tot_c))
print("⇒ 最小类!  测试样本数为 0 的类: %d 个" % (tot_c - len(cnt_te),))
ok = sum(1 for c in cnt_te.values() if c >= 2)
print("⇒ 测试侧 ≥2 个样本的类: %d / %d  ⇒ 其余无法参与 macroAUC" % (ok, tot_c))

print()
print("=== 不同频次阈值下可评估类的规模 ===")
print("%8s %10s %10s %12s" % ("阈值", "类数", "片段数", "占全片段"))
for th in [1, 5, 10, 20, 30, 50, 100]:
    keep = set(c for c, v in cnt_all.items() if v >= th)
    n = sum(v for c, v in cnt_all.items() if c in keep)
    print("%8d %10d %10d %11.1f%%" % (th, len(keep), n, 100 * n / len(X)))

# ---------- Cohen's d（零号自检，不依赖分类器）----------
print()
print("=== Cohen's d 零号自检 ===")
gmap = collections.defaultdict(list)
for xx, yy in zip(X, y_arr):
    gmap[yy].append(xx)
# 只保留样本数 >=2 的类，才能算同词内部距离
gmap = {k: v for k, v in gmap.items() if len(v) >= 2}
ks = list(gmap)
print("可用类数(≥2 样本) = %d" % len(ks))
rs = np.random.RandomState(0)
same, diff = [], []
for _ in range(20000):
    i, j = rs.choice(len(ks), 2, replace=False)
    va = gmap[ks[i]]
    same.append(np.linalg.norm(va[0] - va[1]))
    diff.append(np.linalg.norm(va[0] - gmap[ks[j]][0]))
d = (np.mean(diff) - np.mean(same)) / (np.std(same + diff) + 1e-9)
print("Cohen's d = %+.4f   (P36 参照: upper pose+face = +0.3041, full368 = +0.0410)"
      % d)
