# -*- coding: utf-8 -*-
"""P51 · 选特征源：不是「哪个过拟合好」，而是「哪个判别力强」

## 用户指令
「用效果好的那个」

## 但「效果好」有三种可能，不能混为一谈
1. **过拟合能力**：模型能不能记住训练集
   -> 离线赢（exact 96.7% vs 3.3%），但这**不代表泛化**，只代表坐标系匹配
2. **判别力**：特征本身能不能区分不同 gloss
   -> 必须用**探针 macro-AUC** 测（P38 铁律：acc 在不均衡任务上失真）
3. **端到端 dev 表现**：最终指标
   -> 需重训才能测（1 小时）

## 本脚本做「二选一」的决定性实验
分别用两种特征，测**片段级判别力**（与模型架构无关，纯比特征）：

### 探针设计（严格遵守 P36/P38 铁律）
- **片段池化**而非整句池化（P36 铁律：整句池化让句内 gloss 共享向量，任务不可解）
- 特征 = concat(seg.mean, seg.std)
- 判据 = **macro-AUC**（P38 铁律：类别不均衡，acc 会失真）
- 每块单独测：hands / pose / face / deltas
- 5 折 CV，报告均值±标准差

### 为什么这个实验能决定选谁
判别力是特征的**上限**。如果离线特征判别力 >= 实时特征，
那选离线（零重训）没有损失；如果实时特征明显更强，
那就值得花 1 小时重提重训（可能有额外收益）。

## 同时验证一件关键事
旧特征（holistic）用了 478 点 face + 33 pose，新特征只取 8+8。
探针能直接量化「少取这么多点，损失了多少判别力」。
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
BLOCKS = {"hands[0:126]": (0, 126), "pose[126:158]": (126, 158),
          "face[158:182]": (158, 182), "deltas[186:368]": (186, 368)}


def read_csv(p):
    import csv
    with open(p, newline="", encoding="utf-8") as f:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(f)}


def pool_segments(feat, tokens, voc):
    """片段池化：按 gloss 数等分时间轴，每段 concat(mean, std)。

    P36 铁律：绝不能整句池化（句内 gloss 共享向量，任务不可解）。
    """
    X, y = [], []
    T = feat.shape[0]
    k = len(tokens)
    if k == 0:
        return X, y
    edges = np.linspace(0, T, k + 1).round().astype(int)
    for i, t in enumerate(tokens):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            b = min(a + 1, T)
        if b <= a:
            continue
        blk = feat[a:b]
        X.append(np.concatenate([blk.mean(0), blk.std(0)]))
        y.append(voc.index_of(t))
    return X, y


def macro_auc(scores, y, classes):
    """P38 铁律：类别不均衡下只用 macro-AUC 作主判据。"""
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


def probe(X, y, n_splits=5, seed=0):
    """逻辑回归探针，5 折 CV，返回 macro-AUC 均值/标准差与 acc。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    cnt = collections.Counter(y.tolist())
    keep = {c for c, n in cnt.items() if n >= 5}
    mask = np.array([c in keep for c in y])
    X, y = X[mask], y[mask]
    remap = {c: i for i, c in enumerate(sorted(keep))}
    y = np.array([remap[c] for c in y])
    if len(remap) < 2 or len(X) < 20:
        return None
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    aucs, accs = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=seed)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0,
                                 multi_class="multinomial")
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        pred = p.argmax(1)
        accs.append(float((pred == y[te]).mean()))
        aucs.append(macro_auc(p, y[te], np.arange(len(remap))))
    return {"macro_auc": round(float(np.mean(aucs)), 4),
            "macro_auc_std": round(float(np.std(aucs)), 4),
            "acc": round(float(np.mean(accs)), 4),
            "n_frag": int(len(X)), "n_cls": int(len(remap)),
            "chance": round(1.0 / len(remap), 4)}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=400)
    ap.add_argument("--out", default=str(
        REPO / "artifacts/metrics/blank-gov/p51-feature-source-decision.json"))
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression  # noqa: F401
    try:
        import sklearn  # noqa: F401
        print("sklearn %s 可用" % sklearn.__version__)
    except ImportError:
        print("❌ sklearn 未安装")
        return

    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)

    from realtime_landmark import RealtimeLandmarkExtractor
    ex = RealtimeLandmarkExtractor()

    # 收集成对样本（同一批视频，两种特征）
    data = {"offline": {"X": [], "y": []},
            "realtime": {"X": [], "y": []}}
    per_block = {src: {b: {"X": [], "y": []} for b in BLOCKS}
                 for src in ("offline", "realtime")}
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
        except Exception as exc:                                # noqa: BLE001
            print("  提取失败 %s: %s" % (sid, str(exc)[:50]), flush=True)
            continue
        off_feat = np.load(p).astype(np.float32)
        for src, feat in (("offline", off_feat), ("realtime", on_feat)):
            X, y = pool_segments(feat, toks, voc)
            data[src]["X"].extend(X)
            data[src]["y"].extend(y)
            for b, (s0, s1) in BLOCKS.items():
                Xb, yb = pool_segments(feat[:, s0:s1], toks, voc)
                per_block[src][b]["X"].extend(Xb)
                per_block[src][b]["y"].extend(yb)
        n_done += 1
        if n_done % 50 == 0:
            print("  已处理 %d/%d 视频" % (n_done, a.n_videos), flush=True)

    print("\n处理完成：%d 个 dev 视频" % n_done, flush=True)
    out = {"n_videos": n_done, "full368": {}, "per_block": {}}

    print("\n" + "=" * 72)
    print("整特征（368 维）判别力对比")
    print("=" * 72)
    print("%-14s %22s %22s %10s" % ("特征源", "macro-AUC", "acc", "片段/类"))
    for src in ("offline", "realtime"):
        r = probe(data[src]["X"], data[src]["y"])
        out["full368"][src] = r
        if r:
            print("%-14s %10.4f ±%-8.4f %10.4f %6d/%d"
                  % ("离线 holistic", r["macro_auc"], r["macro_auc_std"],
                     r["acc"], r["n_frag"], r["n_cls"]))
            print("%-14s %10.4f ±%-8.4f %10.4f %6d/%d"
                  % ("实时 tasksapi", r["macro_auc"], r["macro_auc_std"],
                     r["acc"], r["n_frag"], r["n_cls"]))
            print("%-14s %10.4f（随机 %.4f）" % ("差值",
                                               r["macro_auc"], r["chance"]))

    print("\n" + "=" * 72)
    print("分块判别力（看清「少取 face/pose 点」损失多少）")
    print("=" * 72)
    print("%-18s %14s %14s %10s" % ("block", "离线 AUC", "实时 AUC", "差值"))
    for b in BLOCKS:
        ro = probe(per_block["offline"][b]["X"], per_block["offline"][b]["y"])
        rn = probe(per_block["realtime"][b]["X"], per_block["realtime"][b]["y"])
        out["per_block"][b] = {"offline": ro, "realtime": rn}
        if ro and rn:
            print("%-18s %14.4f %14.4f %10.4f"
                  % (b, ro["macro_auc"], rn["macro_auc"],
                     rn["macro_auc"] - ro["macro_auc"]))

    # ---- 判决 ----
    print("\n" + "=" * 72)
    print("判决")
    print("=" * 72)
    o = out["full368"].get("offline")
    r = out["full368"].get("realtime")
    if o and r:
        d = r["macro_auc"] - o["macro_auc"]
        out["delta_realtime_minus_offline"] = round(d, 4)
        if d < 0.05:
            print("  两种特征判别力差异 <0.05（不显著）")
            print("  => 选**离线**（零重训，保留 48 轮实验判优价值）")
            out["decision"] = "offline（判别力相当，零成本）"
        elif d > 0.05:
            print("  实时特征判别力明显更强（+%0.4f）" % d)
            print("  => 值得花 1 小时重提重训（可能有额外收益）")
            out["decision"] = "realtime（判别力更强，值得重训）"
        else:
            print("  离线特征判别力明显更强（%+.4f）" % d)
            print("  => 选**离线**，且若要走方案 A 需注意新特征更差")
            out["decision"] = "offline（判别力更强）"
        print()
        print("  ⚠️ 提醒：判别力 ≠ 端到端效果。判别力是上限，")
        print("     最终仍需重训验证 dev WER 才能定论。")

    p = Path(a.out)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
