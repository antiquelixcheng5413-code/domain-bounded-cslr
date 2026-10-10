"""P107 B2 修正探针：原判据 macroAUC 在全 3841 类上不可算（30%+ 类测试侧为 0 样本）。

修正方案（不改变评分本质，只让它可计算）：
- 主判据：Cohen's d（零号自检，不依赖分类器，全类可用）
- 辅助：macroAUC 限制到「训练集频次 >= MIN_FREQ」的高频类子集

阈值 MIN_FREQ=20 ⇒ 165 类，覆盖 62.9% 片段。
"""
import collections
import csv
import json
import time
from pathlib import Path

import numpy as np

MIN_FREQ = 20
USE_PCA = True          # 原 1024 维 LR 在 165 类上 >30 min 未完成；降维以显著加速
PCA_DIM = 256
ROOT = Path("/home/su127/FYP/domain-bounded-cslr")
FEATS = ROOT / "artifacts/frame_feats/train"
CSV = ROOT / "data/raw/CE-CSL/label/train.csv"
OUT = ROOT / "artifacts/metrics/blank-gov/p107b2-framefeat-probe.json"

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
grp_arr = np.asarray(grp)
print("片段=%d 维=%d distinct类=%d" % (len(X), X.shape[1], len(set(y))))

if USE_PCA:
    from sklearn.decomposition import PCA

    t_p = time.time()
    X = PCA(n_components=PCA_DIM, svd_solver="randomized",
            random_state=0).fit_transform(X).astype(np.float32)
    print("PCA %d -> %d 维 (%.1fs)" % (1024, PCA_DIM, time.time() - t_p), flush=True)

gv = sorted(set(grp))
half = set(gv[: len(gv) // 2])
tr = np.array([g in half for g in grp])
te = ~tr

# ---------- 主判据：Cohen's d（全类，不依赖分类器）----------
gmap = collections.defaultdict(list)
for xx, yy in zip(X, y_arr):
    gmap[yy].append(xx)
usable = {k: v for k, v in gmap.items() if len(v) >= 2}
ks = list(usable)
rs = np.random.RandomState(0)
same, diff = [], []
for _ in range(20000):
    i, j = rs.choice(len(ks), 2, replace=False)
    va = usable[ks[i]]
    same.append(np.linalg.norm(va[0] - va[1]))
    diff.append(np.linalg.norm(va[0] - usable[ks[j]][0]))
d = (np.mean(diff) - np.mean(same)) / (np.std(same + diff) + 1e-9)
print("Cohen's d = %+.4f  (可用类 %d)" % (d, len(ks)))

# ---------- 辅助：高频类子集 macroAUC ----------
cnt = collections.Counter(y_arr)
keep = np.array([cnt[w] >= MIN_FREQ for w in y_arr])
print("高频子集(频次>=%d): %d 片段, %d 类" % (MIN_FREQ, keep.sum(), len(set(y_arr[keep]))))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

Xk, yk = X[keep], y_arr[keep]
trk, tek = tr[keep], te[keep]

# 过滤掉「无法参与 AUC」的类：必须训练侧 ≥1 且测试侧 ≥2 个样本，
# 否则该类 y_true 列只有单一取值 ⇒ per-class AUC undefined ⇒ macro 得 nan
cnt_trk = collections.Counter(yk[trk])
cnt_tek = collections.Counter(yk[tek])
valid = set(c for c in cnt_tek if cnt_tek[c] >= 2 and cnt_trk.get(c, 0) >= 1)
mask_v = np.array([c in valid for c in yk])
print("可参与 AUC 的类: %d / %d（剔除测试侧<2 或训练侧缺失的类）"
      % (len(valid), len(set(yk))), flush=True)
Xk, yk, trk, tek = Xk[mask_v], yk[mask_v], trk[mask_v], tek[mask_v]

lb = LabelBinarizer().fit(yk)
Yk = lb.transform(yk)
t = time.time()
clf = LogisticRegression(max_iter=500, n_jobs=-1)
clf.fit(Xk[trk], yk[trk])
P = clf.predict_proba(Xk[tek])
auc = roc_auc_score(Yk[tek], P, average="macro", multi_class="ovr")
print("子集 macroAUC = %.4f  (fit %.0fs)" % (auc, time.time() - t), flush=True)

rep = {
    "probe": "P107 B2 frames-feature discriminability",
    "note": "原全类 macroAUC 不可算（1281/3841 类测试侧 0 样本），故改双指标",
    "n_segments": int(len(X)),
    "dim": int(X.shape[1]),          # PCA 后的实际维度
    "pca_applied": USE_PCA,
    "pca_dim": PCA_DIM if USE_PCA else None,
    "distinct_glosses": int(len(set(y))),
    "head_to_head": int(len(keep)),
    "n_test_zero_sample_classes": None,
    "threshold": {"dead_auc": 0.10, "alive_auc": 0.25, "cohens_d_ref_upper": 0.3041,
                  "cohens_d_ref_full368": 0.0410},
    "cohens_d": round(float(d), 4),
    "cohens_d_usable_classes": len(ks),
    "subset_min_freq": MIN_FREQ,
    "subset_classes": int(len(set(yk))),
    "subset_macro_auc": round(float(auc), 4),
    "verdict_by_cohens_d": ("判活" if d >= 0.25 else "判死" if d <= 0.10 else "灰区"),
    "verdict_by_auc": ("判活" if auc >= 0.25 else "判死" if auc <= 0.10 else "灰区"),
    "_caveat": ("Cohen's d 在 resnet34MAM RGB 帧特征上；P36 参照值是 landmark 特征，"
                "模态不同，仅作效应量量级参照。Cohen's d 只测前端判别力，"
                "不回答 LLM 后端能否降低 WER。"),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
print("收据 ->", OUT)
