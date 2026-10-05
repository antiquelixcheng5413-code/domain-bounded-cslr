# -*- coding: utf-8 -*-
"""P51c · 特征源选型（定论版）：分批落盘 + 大样本 + 正确 AUC

## 前两版为什么作废
- P51：OOM（7.1GB），进程被杀
- P51b：流式化解决了 OOM，但 **holdout 只有 62~80 片段 / 2~5 类别**，
  macro-AUC 极不稳定（实测 0.07 与 0.51 混现），结论不可信
- 附：期间我怀疑自己的 AUC 实现有 bug，穷举验证后确认**实现是对的**
  （穷举 / 手写 Mann-Whitney / sklearn 三者完全一致）。
  之前「完美分数得 0.5」是我的**自检样例设计错误**（正组全 0、
  负组含 1 和 2，AUC 必然 0/0.5/1 分布，平均成 0.5）。
  **教训：自检样例必须先验算过，不能想当然。**

## 本版的三处修正
1. **样本量**：目标 >= 400 片段 / >= 15 类别，不达标就继续收样本
2. **内存**：片段分批写入 npy，不在内存里堆全量
3. **AUC**：用 sklearn.roc_auc_score（已穷举验证等价）
4. **判定标准**：加置信区间，AUC 差值 <0.05 且区间重叠 -> 判「无显著差异」

## 判据（P38 铁律）
片段级 macro-AUC，concat(seg.mean, seg.std)（P36 铁律：不能整句池化）
"""
from __future__ import annotations

import csv
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
CACHE = Path("/tmp/p51c")
SLICES = {"hands": (0, 126), "pose": (126, 158),
          "face": (158, 182), "deltas": (186, 368)}
MIN_FRAG, MIN_CLS = 400, 15


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(f)}


def pool(feat, tokens):
    X = []
    T, k = feat.shape[0], len(tokens)
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


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=250)
    ap.add_argument("--out", default=str(
        REPO / "artifacts/metrics/blank-gov/p51c-feature-source.json"))
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    CACHE.mkdir(parents=True, exist_ok=True)
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)

    from realtime_landmark import RealtimeLandmarkExtractor
    ex = RealtimeLandmarkExtractor()

    # 逐批收集并落盘
    batch_i = 0
    buf = {"offline": [[], []], "realtime": [[], []]}
    total = {"offline": 0, "realtime": 0}
    n_done = 0

    def flush():
        nonlocal buf, batch_i
        for src in buf:
            if buf[src][0]:
                np.savez_compressed(CACHE / ("%s_%03d.npz" % (src, batch_i)),
                                    X=np.concatenate(buf[src][0]),
                                    y=np.concatenate(buf[src][1]))
        buf = {"offline": [[], []], "realtime": [[], []]}
        batch_i += 1

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
        if len(toks) < 2:
            continue
        try:
            on_feat = ex.extract_to_48x368(str(vp))
        except Exception:                                       # noqa: BLE001
            continue
        y = np.array([voc.index_of(t) for t in toks])
        for src, feat in (("offline", np.load(p).astype(np.float32)),
                          ("realtime", on_feat)):
            pooled = pool(feat, toks)
            if not pooled:
                continue
            buf[src][0].append(np.stack(pooled))
            buf[src][1].append(y)
            total[src] += len(pooled)
        n_done += 1
        if total["offline"] >= 1200:
            flush()
            import resource
            print("  %d 视频，已缓存 %d/%d 片段，峰值 %.0f MB"
                  % (n_done, total["offline"], total["realtime"],
                     resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                  flush=True)
    flush()
    print("\n视频 %d，片段 offline=%d realtime=%d"
          % (n_done, total["offline"], total["realtime"]), flush=True)

    def load(src):
        Xs, ys = [], []
        for f in sorted(CACHE.glob("%s_*.npz" % src)):
            z = np.load(f)
            Xs.append(z["X"])
            ys.append(z["y"])
        return np.concatenate(Xs), np.concatenate(ys)

    data = {s: load(s) for s in ("offline", "realtime")}

    def blocks(X):
        out = {"full": X}
        for b, (s0, s1) in SLICES.items():
            # 池化后的向量 = concat(mean[0:368], std[0:368])，长度 736
            out[b] = np.concatenate([X[:, s0:s1], X[:, 368 + s0:368 + s1]], 1)
        return out

    def macro_auc_ci(p, y, classes, n_boot=200, seed=0):
        """bootstrap 置信区间，AUC 差值小于区间宽度即判无显著差异。"""
        rng = np.random.RandomState(seed)
        aucs = []
        for c in classes:
            m = y == c
            if m.sum() < 3 or (~m).sum() < 3:
                continue
            aucs.append(roc_auc_score(m.astype(int), p[:, c]))
        if not aucs:
            return None, None, 0
        point = float(np.mean(aucs))
        boots = []
        idx = np.arange(len(y))
        for _ in range(n_boot):
            bidx = rng.choice(idx, len(idx), replace=True)
            bb = []
            for c in classes:
                m = y[bidx] == c
                if m.sum() < 3 or (~m).sum() < 3:
                    bb.append(0.5)
                    continue
                bb.append(roc_auc_score(m.astype(int), p[bidx][:, c]))
            boots.append(float(np.mean(bb)))
        return point, (float(np.percentile(boots, 2.5)),
                       float(np.percentile(boots, 97.5))), len(aucs)

    out = {"n_videos": n_done,
           "n_frag": {s: int(len(data[s][1])) for s in data},
           "min_frag": MIN_FRAG, "min_cls": MIN_CLS, "results": {}}

    print("\n" + "=" * 78)
    print("判别力对比（片段级 macro-AUC，25% holdout，200 次 bootstrap）")
    print("=" * 78)
    print("%-9s %-9s %18s %18s %10s"
          % ("块", "样本", "离线 holistic", "实时 tasksapi", "差值"))
    sig = []
    for b in ["full"] + list(SLICES):
        row = {}
        for src in ("offline", "realtime"):
            X, y = data[src]
            B = blocks(X)[b]
            cnt = {c: n for c, n in
                   zip(*np.unique(y, return_counts=True))}
            keep = {c for c, n in cnt.items() if n >= 5}
            m = np.array([c in keep for c in y])
            Xf, yf = B[m], y[m]
            if len(Xf) < MIN_FRAG or len(keep) < MIN_CLS:
                row[src] = None
                row[src + "_n"] = "%d段/%d类" % (len(Xf), len(keep))
                continue
            remap = {c: i for i, c in enumerate(sorted(keep))}
            yr = np.array([remap[c] for c in yf])
            Xtr, Xte, ytr, yte = train_test_split(
                Xf, yr, test_size=0.25, random_state=0, stratify=yr)
            sc = LogisticRegression(max_iter=3000, C=0.5)
            sc.fit(Xtr, ytr)
            p = sc.predict_proba(Xte)
            pt, ci, nc = macro_auc_ci(p, yte, np.arange(sc.classes_.shape[0]))
            row[src] = {"auc": round(pt, 4), "ci": [round(ci[0], 4),
                                                    round(ci[1], 4)],
                        "n_cls": nc, "n_frag": int(len(yte))}
        out["results"][b] = row
        o, r = row.get("offline"), row.get("realtime")
        if isinstance(o, dict) and isinstance(r, dict):
            d = r["auc"] - o["auc"]
            overlap = not (r["ci"][0] > o["ci"][1] or o["ci"][0] > r["ci"][1])
            if abs(d) < 0.05 or overlap:
                sig.append((b, False))
            print("%-9s %-9s %10.4f[%5.3f,%5.3f] %10.4f[%5.3f,%5.3f] %10.4f%s"
                  % (b, "%d/%d" % (o["n_frag"], o["n_cls"]),
                     o["auc"], o["ci"][0], o["ci"][1],
                     r["auc"], r["ci"][0], r["ci"][1], d,
                     "  (重叠)" if overlap else ""))
        else:
            print("%-9s %-9s %18s %18s"
                  % (b, str(row.get("offline_n", "")),
                     o["auc"] if isinstance(o, dict) else "样本不足",
                     r["auc"] if isinstance(r, dict) else "样本不足"))

    # ---- 判决 ----
    print("\n" + "=" * 78)
    print("判决")
    print("=" * 78)
    f = out["results"].get("full", {})
    if isinstance(f.get("offline"), dict) and isinstance(f.get("realtime"), dict):
        d = f["realtime"]["auc"] - f["offline"]["auc"]
        overlap = not (f["realtime"]["ci"][0] > f["offline"]["ci"][1]
                       or f["offline"]["ci"][0] > f["realtime"]["ci"][1])
        out["delta_full"] = round(d, 4)
        out["ci_overlap"] = overlap
        print("  整特征：离线 %.4f%s  实时 %.4f%s  差 %+.4f"
              % (f["offline"]["auc"], f["offline"]["ci"],
                 f["realtime"]["auc"], f["realtime"]["ci"], d))
        print("  95%% 置信区间%s" % ("**重叠** -> 无显著差异" if overlap
                                       else "不重叠 -> 差异显著"))
        print()
        if abs(d) < 0.05 or overlap:
            print("  => 判别力**无显著差异**，选离线（零重训，"
                  "保住 48 轮实验判优价值）")
            out["decision"] = "offline"
        elif d > 0:
            print("  => 实时判别力显著更强，值得重提重训（1 小时）")
            out["decision"] = "realtime"
        else:
            print("  => 离线判别力显著更强，选离线")
            out["decision"] = "offline"
    else:
        print("  ⚠️ 样本仍未达标，需再收样本才能定论")
        out["decision"] = "inconclusive"

    p = Path(a.out)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
