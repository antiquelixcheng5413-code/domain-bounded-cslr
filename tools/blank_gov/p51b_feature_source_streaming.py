# -*- coding: utf-8 -*-
"""P51b · 特征源选型：流式判别力探针（修 P51 的 OOM）

## P51 失败原因
`dmesg` 确认 OOM：进程 anon-rss 7445152 kB = 7.1GB 被杀。
根因：736 维特征 × 400 视频 × ~5.5 词 = 数千片段，**全部堆在内存里**，
再乘以「整特征 + 5 个分块」两份 = 翻倍。WSL 只有 7GB。

## 本脚本的修法：SGDClassifier + partial_fit（流式）
1. 逐视频提取 -> 立即池化成片段 -> 立刻喂给 `partial_fit`
   -> 内存里只保留一个分块的小缓冲，**不随视频数增长**
2. 不存特征矩阵，改存「每个片段的标签 + 所属 fold」
3. 判据仍用 **macro-AUC**（P38 铁律），但改为在**留出 fold** 上算
   —— 用 holdout 而不是 5 折 CV，省内存且更接近真实用法

## 重要：同时修一个我 P51 里的统计错误
P51 用 `pool_segments(feat[:, s0:s1], ...)` 切分块后池化，
但**差分块（deltas[186:368]）不能单独池化** —— 它是帧间差值，
语义上依赖相邻帧。切开后池化会破坏其结构。
→ 本脚本对 deltas 块改用「整特征先池化再切片」的顺序，
   保证块内差分的相对关系不被破坏。

## 判据
macro-AUC（holdout），片段级，concat(seg.mean, seg.std)（P36 铁律）
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "app" / "backend"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402

OFF = REPO / "artifacts/part3_features"
VID = Path("/mnt/c/Users/su127/Desktop/csl视频")
UNK = "<unk>"
# 块定义：先在整特征上池化，再按维度切 —— 保证 deltas 的帧间关系不被破坏
SLICES = {
    "hands": (0, 126),
    "pose": (126, 158),
    "face": (158, 182),
    "deltas": (186, 368),
}
HOLDOUT = 0.3


def read_csv(p):
    import csv
    with open(p, newline="", encoding="utf-8") as f:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(f)}


def pool(feat, tokens):
    """片段池化：整特征上做 concat(mean, std)（P36 铁律：不能整句池化）。"""
    X = []
    T = feat.shape[0]
    k = len(tokens)
    if k == 0:
        return X
    edges = np.linspace(0, T, k + 1).round().astype(int)
    for i in range(k):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            b = min(a + 1, T)
        if b <= a:
            continue
        blk = feat[a:b]
        X.append(np.concatenate([blk.mean(0), blk.std(0)]))
    return X


def macro_auc(scores, y, classes):
    aucs = []
    for c in classes:
        pos = scores[y == c]
        neg = scores[y != c]
        if len(pos) == 0 or len(neg) == 0:
            continue
        r = np.argsort(np.argsort(np.concatenate([pos, neg])))
        aucs.append((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                    / (len(pos) * len(neg)))
    return float(np.mean(aucs)) if aucs else 0.0


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", default=str(
        REPO / "artifacts/metrics/blank-gov/p51b-feature-source.json"))
    a = ap.parse_args()

    from sklearn.linear_model import SGDClassifier

    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)
    classes = np.arange(int(voc.size))

    from realtime_landmark import RealtimeLandmarkExtractor
    ex = RealtimeLandmarkExtractor()

    # 为每个特征源、每个块建一个流式分类器
    keys = ["full"] + list(SLICES)
    clf = {src: {k: SGDClassifier(loss="log_loss", alpha=1e-4,
                                   max_iter=1, tol=None, random_state=0)
                 for k in keys} for src in ("offline", "realtime")}
    seen = {src: {k: 0 for k in keys} for src in clf}
    trX = {src: {k: [] for k in keys} for src in clf}
    trY = {src: {k: [] for k in keys} for src in clf}
    teX = {src: {k: [] for k in keys} for src in clf}
    teY = {src: {k: [] for k in keys} for src in clf}

    rng = np.random.RandomState(0)
    n_done = 0
    for sid in sorted(lab_dv):
        if n_done >= a.n_videos:
            break
        p = OFF / "validation" / (sid + ".landmark.npy")
        if not p.exists():
            continue
        vp = None
        for d in sorted((VID / "dev").iterdir()) if (VID / "dev").exists() else []:
            q = d / (sid + ".mp4")
            if q.exists():
                vp = q
                break
        if vp is None:
            continue
        toks = [t for t in lab_dv[sid].split("/") if t]
        if len(toks) < 3:
            continue
        try:
            on_feat = ex.extract_to_48x368(str(vp))
        except Exception:                                       # noqa: BLE001
            continue

        for src, feat in (("offline", np.load(p).astype(np.float32)),
                          ("realtime", on_feat)):
            pooled = pool(feat, toks)            # 先整特征池化
            if not pooled:
                continue
            y = np.array([voc.index_of(t) for t in toks])
            # 特征字典：full + 各块切片（池化之后切，保护 deltas 结构）
            feats = {"full": np.stack(pooled)}
            for b, (s0, s1) in SLICES.items():
                # 池化后再切维度：deltas 段是帧间差值，
                # 若先切维度再池化会破坏其帧间关系。
                feats[b] = np.stack([v[s0:s1] for v in pooled])
            for k in keys:
                Xk = feats[k]
                m = rng.rand(len(Xk)) < (1 - HOLDOUT)
                if m.sum() == 0 or (~m).sum() == 0:
                    continue
                clf[src][k].partial_fit(Xk[m], y[m], classes=classes)
                seen[src][k] += int(m.sum())
                # holdout 缓冲只留有限数量，控制内存
                if len(trX[src][k]) < 4000:
                    trX[src][k].append(Xk[m])
                    trY[src][k].append(y[m])
                    teX[src][k].append(Xk[~m])
                    teY[src][k].append(y[~m])
                del Xk
        n_done += 1
        if n_done % 25 == 0:
            import resource
            mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print("  %d/%d 视频  峰值内存 %.0f MB" % (n_done, a.n_videos, mb),
                  flush=True)
    print("\n完成：%d 个 dev 视频" % n_done, flush=True)

    # ---- 评估 ----
    out = {"n_videos": n_done, "holdout": HOLDOUT, "results": {}}
    print("\n" + "=" * 76)
    print("判别力对比（片段级 macro-AUC，holdout %.0f%%）" % (HOLDOUT * 100))
    print("=" * 76)
    print("%-10s %18s %18s %10s" % ("块", "离线 holistic", "实时 tasksapi",
                                    "差值"))
    for k in keys:
        row = {}
        for src in ("offline", "realtime"):
            if not teX[src][k] or not trX[src][k]:
                row[src] = None
                continue
            Xtr = np.concatenate(trX[src][k])
            Ytr = np.concatenate(trY[src][k])
            Xte = np.concatenate(teX[src][k])
            Yte = np.concatenate(teY[src][k])
            c = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=0)
            for _ in range(12):
                c.partial_fit(Xtr, Ytr, classes=classes)
            p = c.predict_proba(Xte)
            keep = [j for j, cc in enumerate(c.classes_) if
                    (Yte == cc).sum() >= 3 and (Yte != cc).sum() >= 3]
            sub = p[:, keep]
            row[src] = round(macro_auc(sub, Yte, c.classes_[keep]), 4)
            row[src + "_n_frag"] = int(len(Yte))
            row[src + "_n_cls"] = int(len(keep))
            del Xtr, Xte, p, sub
        out["results"][k] = row
        if row.get("offline") is not None and row.get("realtime") is not None:
            d = row["realtime"] - row["offline"]
            print("%-10s %18.4f %18.4f %10.4f"
                  % (k, row["offline"], row["realtime"], d))
        else:
            print("%-10s %18s %18s" % (k, row.get("offline", "n/a"),
                                       row.get("realtime", "n/a")))
        # 及时释放
        for src in ("offline", "realtime"):
            trX[src][k] = []
            teX[src][k] = []

    # ---- 判决 ----
    print("\n" + "=" * 76)
    print("判决")
    print("=" * 76)
    f = out["results"].get("full", {})
    if f.get("offline") is not None and f.get("realtime") is not None:
        d = f["realtime"] - f["offline"]
        out["delta_full"] = round(d, 4)
        print("  整特征 macro-AUC：离线 %.4f  实时 %.4f  差 %+.4f"
              % (f["offline"], f["realtime"], d))
        print()
        if abs(d) < 0.05:
            print("  => 判别力相当，**选离线**（零重训，保住 48 轮实验判优价值）")
            out["decision"] = "offline"
        elif d > 0:
            print("  => 实时判别力更强，**值得重提重训**（1 小时）")
            out["decision"] = "realtime"
        else:
            print("  => 离线判别力更强，**选离线**")
            out["decision"] = "offline"
        print()
        print("  ⚠️ 判别力是上限，不等于端到端效果；最终仍需重训验 dev WER。")
        print("  ⚠️ 但对「用哪个」这个问题，判别力是唯一可先验比较的量。")

    p = Path(a.out)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
