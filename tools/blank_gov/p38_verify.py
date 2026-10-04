# -*- coding: utf-8 -*-
"""P38 · 验证 P37 的两个发现是否真实（跨子集稳定性 + 真实分类转化）

## P37 给了两个反直觉发现

1. `upper`（pose+face）d=+0.3125，不是维度假象
   （同维度 hands 随机 56 维基线仅 +0.0129，差 24 倍）
2. **`body_deltas` z-score 后 d=+0.4796 —— 全场最高，比 upper 还高**
   而它此前从未被单独研究过（`[312:368]`，仅 56 维）

## 但这两个数字都还只是「相似度统计量」，有两处必须先验证

### 验证 1：跨子集稳定性
Cohen's d 是在全部 1865 个片段上一次算出来的。
若它在不同的视频子集上剧烈波动，说明是采样噪声而非稳定信号。
→ **按视频分 5 折，每折独立算 d，看变异系数。**

### 验证 2：d 能否转化为真实分类提升
P36 已暴露一个教训：**d 高不等于分类好**
（upper 的 d=0.30 对应 heldout acc 仅 0.0222，远低于基线 0.1833）。
所以 d=0.4796 若不能转化为分类提升，就不值得投 CTC 训练。
→ **同一套片段池化 + 视频级 80/20 heldout，对比 5 个视图的 acc / AUC。**

AUC 在类别极不均衡（P30: 11-100 桶占 24%）下比 acc 更可信，
所以同时报 `macro-AUC` 与 `balanced acc`。

## 判决

- 验证 1 变异系数 < 0.3 → 信号稳定，继续
- 验证 2 macro-AUC > full368 基线 2 倍 → 值得投入 CTC 训练
- 否则判为「统计量好看但无法转化」，与 P34 的 face128 同样结局
"""
import argparse
import collections
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

BLOCKS = {
    "hands": (0, 126),
    "pose": (126, 158),
    "face": (158, 182),
    "masks": (182, 186),
    "hand_deltas": (186, 312),
    "body_deltas": (312, 368),
}


def read_csv(p):
    rows = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["Number"]] = r["Gloss"]
    return rows


def l2n(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def zs(X):
    return (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)


def cohen_d(V, labels, n_pairs=6000, seed=11):
    Vr = l2n(V)
    rng = random.Random(seed)
    pos, neg = [], []
    for _ in range(n_pairs):
        i, j = rng.randrange(len(V)), rng.randrange(len(V))
        if i == j:
            continue
        s = float(Vr[i] @ Vr[j])
        (pos if labels[i] == labels[j] else neg).append(s)
    if not pos or not neg:
        return float("nan")
    return (float(np.mean(pos) - np.mean(neg))) / float(np.std(pos + neg))


def macro_auc(scores, labels_pos, labels_all):
    """二分类式的 macro-AUC：正类=目标片段，负类=其余，按分数排序算秩。

    避免 sklearn 依赖；对不均衡稳健。
    """
    pos = np.asarray(scores)[labels_pos]
    neg = np.asarray(scores)[~labels_pos]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 并列取平均秩
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p38-verify-upper-deltas.json")
    a = ap.parse_args()

    from cslr.recognition.gloss_sequence import build_ordered_vocabulary

    t0 = time.time()
    lab = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    voc, _ = build_ordered_vocabulary(lab.values(), min_frequency=2, max_tokens=300)
    feat = REPO / "artifacts/part3_features/train"
    avail = sorted(p.name[:-len(".landmark.npy")] for p in feat.glob("*.landmark.npy"))
    have_rgb = {p.name[:-len(".rgb.npy")] for p in feat.glob("*.rgb.npy")}
    avail = [s for s in avail if s in have_rgb]
    picked = random.Random(a.seed).sample(avail, min(a.n_videos, len(avail)))

    # ---- 片段池化（P36 铁律）----
    rows = []
    for sid in picked:
        toks = [t.strip() for t in lab[sid].split("/") if t.strip()]
        toks = [t for t in toks if t in voc]
        if len(toks) < 2:
            continue
        p = feat / (sid + ".landmark.npy")
        if not p.exists():
            continue
        L = np.load(p).astype(np.float32)
        if L.shape[0] < len(toks) * 2:
            continue
        bn = np.linspace(0, L.shape[0], len(toks) + 1).astype(int)
        for i, t in enumerate(toks):
            if bn[i + 1] - bn[i] < 2:
                continue
            rows.append((sid, t, L[bn[i]:bn[i + 1]]))
    labels_all = [r[1] for r in rows]
    sids_all = np.array([r[0] for r in rows])
    print("片段 {} 条，类别 {}".format(len(rows), len(set(labels_all))))

    def pv(a):
        return np.concatenate([a.mean(axis=0), a.std(axis=0)])

    V = {}
    for name, (s, e) in BLOCKS.items():
        V[name] = np.stack([pv(r[2][:, s:e]) for r in rows])
    V["upper"] = np.concatenate([V["pose"], V["face"]], axis=1)
    V["full368"] = np.stack([pv(r[2]) for r in rows])
    # 最佳候选：上半身 + 两个 delta 块，分块标准化后等权 L2 拼接
    V["cand_bn"] = np.concatenate(
        [l2n(zs(V["upper"])), l2n(zs(V["hand_deltas"])), l2n(zs(V["body_deltas"]))],
        axis=1)
    V["deltas_bn"] = np.concatenate(
        [l2n(zs(V["hand_deltas"])), l2n(zs(V["body_deltas"]))], axis=1)

    CAND = ["full368", "upper", "body_deltas", "deltas_bn", "cand_bn", "hands"]

    # ================= 验证 1：5 折跨子集稳定性 =================
    print()
    print("=" * 88)
    print("验证 1 · 5 折跨子集稳定性（按视频切分，每折独立算 d）")
    print("=" * 88)
    uniq = sorted(set(sids_all.tolist()))
    shuffled = uniq[:]
    random.Random(23).shuffle(shuffled)
    folds = [set(shuffled[i::5]) for i in range(5)]
    print("  {:<12s} {:>8s} {:>10s} {:>10s} {:>8s}".format(
        "视图", "d全量", "d均值", "d标准差", "变异系数"))
    stab = {}
    for name in CAND:
        ds = []
        for f in folds:
            m = np.array([s in f for s in sids_all])
            if m.sum() < 30 or (~m).sum() < 30:
                continue
            ds.append(cohen_d(V[name][m], [l for l, k in zip(labels_all, m) if k]))
        d_full = cohen_d(V[name], labels_all)
        mu = float(np.mean(ds))
        sd = float(np.std(ds))
        cv = sd / abs(mu) if abs(mu) > 1e-9 else float("inf")
        stab[name] = {"d_full": round(d_full, 4), "d_folds": [round(x, 4) for x in ds],
                      "d_mean": round(mu, 4), "d_std": round(sd, 4), "cv": round(cv, 4)}
        print("  {:<12s} {:>8.4f} {:>10.4f} {:>10.4f} {:>8.3f}".format(
            name, d_full, mu, sd, cv))
    print()
    print("  >>> 变异系数 < 0.3 视为稳定；CV 越小越可信")

    # ================= 验证 2：d 能否转化为真实分类提升 =================
    print()
    print("=" * 88)
    print("验证 2 · 视频级 80/20 heldout 分类（d 高不等于分类好，P36 已证实）")
    print("=" * 88)
    TEST = set(shuffled[:max(int(len(uniq) * 0.2), 1)])
    te_mask = np.array([s in TEST for s in sids_all])
    tr_i, te_i = np.where(~te_mask)[0], np.where(te_mask)[0]
    y = np.array(labels_all)
    print("  train 片段 {}  test 片段 {}".format(len(tr_i), len(te_i)))

    cls_res = {}
    for name in CAND:
        X = l2n(V[name])
        # 只保留训练侧出现 >=2 次的类（原型至少要 2 个样本才有意义）
        cnt = collections.Counter(y[tr_i].tolist())
        keep = {c for c, n in cnt.items() if n >= 2}
        tri = np.array([i for i in tr_i if y[i] in keep])
        tei = np.array([i for i in te_i if y[i] in keep])
        if len(tei) < 20 or len(tri) < 20:
            print("  {:<12s} 样本不足".format(name))
            continue
        keys = sorted(keep)
        P = {c: X[tri[y[tri] == c]].mean(axis=0) for c in keys}
        M = l2n(np.stack([P[c] for c in keys]))
        sim = X[tei] @ M.T
        pred = [keys[i] for i in sim.argmax(axis=1)]
        gold = y[tei]
        acc = float(np.mean([p == g for p, g in zip(pred, gold)]))
        maj = collections.Counter(gold.tolist()).most_common(1)[0][1] / len(gold)
        bal = float(np.mean([
            np.mean([p == g for p, g in zip(pred, gold) if g == c])
            for c in keys if (gold == c).sum() >= 1]))
        # macro-AUC：正类 = 测试侧出现 >=3 次的类
        freq = collections.Counter(gold.tolist())
        aucs = []
        for c in keys:
            m = freq.get(c, 0)
            if m < 3:
                continue
            pos = np.array([g == c for g in gold])
            aucs.append(macro_auc(sim[:, keys.index(c)], pos, gold))
        mauc = float(np.mean(aucs)) if aucs else float("nan")
        cls_res[name] = {
            "heldout_acc": round(acc, 4), "balanced_acc": round(bal, 4),
            "macro_auc": round(mauc, 4), "majority_baseline": round(maj, 4),
            "n_classes": len(keys), "n_test": int(len(tei))}
        print("  {:<12s} acc={:.4f}  balAcc={:.4f}  macroAUC={:.4f}  基线={:.4f}  类 {}".format(
            name, acc, bal, mauc, maj, len(keys)))

    # ================= 判决 =================
    print()
    print("=" * 88)
    print("判决")
    print("=" * 88)
    base_auc = cls_res.get("full368", {}).get("macro_auc", float("nan"))
    best = max(cls_res.items(), key=lambda kv: kv[1]["macro_auc"])
    bname, bres = best
    cv_worst = max((v["cv"] for v in stab.values() if np.isfinite(v["cv"])),
                   default=float("inf"))
    print("  最稳的视图 CV = {:.3f}（<0.3 视为稳定）".format(min(
        (v["cv"] for v in stab.values() if np.isfinite(v["cv"])), default=float("inf"))))
    print("  最佳分类视图 = {}  macroAUC={:.4f}（full368 基线 {:.4f}，比值 {:.2f}x）".format(
        bname, bres["macro_auc"], base_auc,
        bres["macro_auc"] / base_auc if base_auc else float("inf")))
    ok = bres["macro_auc"] > base_auc * 2 if base_auc else False
    verdict = (
        "验证通过：{} 的 macroAUC {:.4f} 是 full368 的 {:.2f} 倍 -> 值得投 CTC 训练验证".format(
            bname, bres["macro_auc"],
            bres["macro_auc"] / base_auc if base_auc else float("inf"))
        if ok else
        "验证不通过：统计量（d）好看但分类无提升（{} macroAUC {:.4f} vs 基线 {:.4f}）"
        "-> 与 P34 face128 同样结局，不投 CTC".format(bname, bres["macro_auc"], base_auc))
    print("  {}".format(verdict))

    out = {
        "experiment": "P38 verify P37 findings: 5-fold stability + classification transfer",
        "n_videos": len(picked), "n_segments": len(rows),
        "n_classes": len(set(labels_all)), "seed": a.seed,
        "verify1_stability": stab,
        "verify2_classification": cls_res,
        "verdict": verdict,
        "minutes": round((time.time() - t0) / 60, 2),
        "notes": "Cohen's d 不涉及训练/测试切分，故无过拟合；"
                 "但 5 折按视频切分验证采样稳定性。"
                 "P36 教训：d 高不等于分类好，故两问分别验证。",
    }
    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n收据 -> {}".format(p))


if __name__ == "__main__":
    main()
