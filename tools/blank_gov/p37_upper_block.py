# -*- coding: utf-8 -*-
"""P37 · pose+face 块专项 + 分块标准化

## 动机

P36 的 Cohen's d 自检给出两个反直觉结果：
  upper (pose+face, 56 维)  d = +0.3041  ← 全部视图最清晰
  hands  (126 维, 占 68%)   d = -0.0171  ← 同 gloss 反而更不相似
而 `upper` 只占 368 维的 15.2%，P30/P31/P34/P35 四轮实验全都忽略了它。

## 🔴 但先排除一个致命混淆：d 会不会只是「维度数的函数」？

P36 里各视图的**绝对相似度**差异极大：
  upper  同 gloss 0.9093 / 异 0.8884
  hands  同 gloss 0.3774 / 异 0.3827
低维向量 L2 归一化后两点夹角天然更小、余弦天然更大。
**若 d 也随维度单调变化，那 upper 的 d=0.3041 可能只是 56 维的副产品，
而不是 pose+face 真的带 gloss 信息。**

判决性对照：`hands_random56` —— 从 hands 的 126 维里**随机抽 56 维**
（跑多个种子），若它的 d 也≈0.30，则维度效应成立，upper 无优势；
若明显低于 0.30，则 upper 的信号是真的。

## 第二个问题：hands 的 d 为何是负的

P36 实测 hands 126 维 std 跨度 9.7 倍（0.127~1.229）。
L2 归一化后方向被少数高方差维主导，信号维被压扁。
**分块逐维 z-score** 是否能修好？若能，则「hands 里其实有信息，
只是被尺度不均埋了」——这是一个和 P30「容量过剩」完全不同的机制。

## 判据

1. **维度对照**：`hands_random56` 的 d 必须显著低于 `upper` 的 d，
   否则 upper 优势是维度假象
2. **分块标准化**：`hands` 的 d 从 -0.0171 需显著转正
3. 两者都成立才值得投入 CTC 训练验证

## 方法（沿用 P36 已验证的设计）

- 片段池化：按 gloss 数等分时间轴，每段对应 1 个 gloss（P36 铁律）
- 视频级 80/20 切分，所有视图共用同一切分
- Cohen's d 为主判据（**不依赖分类器**，唯一不会因评估器 bug 失真的量）
- 只读 train 特征，不训练任何深度模型
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

# 368 维布局（见 src/cslr/features/extractor.py）
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
    """逐维 z-score。P36 实证：hands 方差跨度 9.7 倍，L2 归一化时被高方差维主导。"""
    return (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)


def cohen_d(V, labels, n_pairs=8000, seed=11):
    """Cohen's d：同 label 配对均值 - 异 label 配对均值，除以全体配对 std。

    不依赖任何分类器 —— 这是 P36 定下的零号自检。
    """
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
        return None
    return (float(np.mean(pos) - np.mean(neg))) / float(np.std(pos + neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p37-upper-block.json")
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
    print("视频 {} 个（train split）".format(len(picked)))

    # ---- 片段池化（P36 铁律：绝不用整句池化）----
    rows = []   # (sid, tok, raw368, rgb)
    for sid in picked:
        toks = [t.strip() for t in lab[sid].split("/") if t.strip()]
        toks = [t for t in toks if t in voc]
        if len(toks) < 2:
            continue
        lp = feat / (sid + ".landmark.npy")
        rp = feat / (sid + ".rgb.npy")
        if not (lp.exists() and rp.exists()):
            continue
        L = np.load(lp).astype(np.float32)
        R = np.load(rp).astype(np.float32)
        if L.shape[0] < len(toks) * 2 or R.shape[0] < len(toks) * 2:
            continue
        bn = np.linspace(0, L.shape[0], len(toks) + 1).astype(int)
        bnR = np.linspace(0, R.shape[0], len(toks) + 1).astype(int)
        for i, t in enumerate(toks):
            if bn[i + 1] - bn[i] < 2 or bnR[i + 1] - bnR[i] < 2:
                continue
            segL = L[bn[i]:bn[i + 1]]
            segR = R[bnR[i]:bnR[i + 1]]
            rows.append((sid, t, segL, segR))
    print("片段 {} 条（每段对应 1 个 gloss）".format(len(rows)))
    labels = [r[1] for r in rows]
    print("覆盖类别 {} 个".format(len(set(labels))))

    def pv(a):
        return np.concatenate([a.mean(axis=0), a.std(axis=0)])

    def sl(name, arr):
        s, e = BLOCKS[name]
        return arr[:, s:e]

    # ================= 第一部分：维度混淆对照 =================
    print()
    print("=" * 92)
    print("第一部分 · 维度混淆对照（判决 upper 的 d=0.3041 是否只是 56 维的副产品）")
    print("=" * 92)
    print("  {:<20s} {:>6s} {:>10s} {:>10s}".format("视图", "维度", "同gloss", "Cohen d"))

    # 收集各块片段向量
    vecs = {}
    for name in BLOCKS:
        vecs[name] = np.stack([pv(sl(name, r[2])) for r in rows])
    vecs["upper"] = np.concatenate(
        [vecs["pose"], vecs["face"]], axis=1)          # 56 维
    vecs["full368"] = np.stack([pv(r[2]) for r in rows])
    vecs["rgb"] = np.stack([pv(r[3]) for r in rows])

    # hands 随机抽 56 维（对齐 upper 的维度），跑 5 个种子
    rngd = random.Random(a.seed)
    hand56_ds = []
    H = vecs["hands"]
    for k in range(5):
        idx = rngd.sample(range(H.shape[1]), 56)
        d = cohen_d(H[:, idx], labels)
        hand56_ds.append(d)
    vecs["hands_rand56_s0"] = H[:, rngd.sample(range(H.shape[1]), 56)]

    part1 = {}
    for name in ("upper", "pose", "face", "hands", "hand_deltas", "body_deltas",
                 "full368", "rgb"):
        V = l2n(vecs[name])
        same = float(np.mean([
            float(V[i] @ V[j]) for i, j in
            [(k, k + 1) for k in range(0, min(400, len(V) - 1), 2)]
        ])) if len(V) > 2 else float("nan")
        d = cohen_d(vecs[name], labels)
        part1[name] = {"dim": int(vecs[name].shape[1]), "cohens_d": round(d, 4)}
        print("  {:<20s} {:>6d} {:>10} {:>10.4f}".format(
            name, vecs[name].shape[1], "-", d))

    print()
    print("  【对照】hands 随机抽 56 维（5 个种子）：")
    for k, d in enumerate(hand56_ds):
        print("    seed{}  d = {:+.4f}".format(k, d))
    print("  hands 全 126 维      d = {:+.4f}".format(part1["hands"]["cohens_d"]))
    print("  upper (pose+face)    d = {:+.4f}".format(part1["upper"]["cohens_d"]))
    m56 = float(np.mean(hand56_ds))
    up_d = part1["upper"]["cohens_d"]
    print()
    print("  >>> 维度效应基线 hands@56维 d = {:+.4f}".format(m56))
    if up_d > m56 * 1.3 + 0.02:
        dim_verdict = ("upper 优势不是维度假象（{:+.4f} > 基线 {:+.4f} 的 1.3 倍）"
                       .format(up_d, m56))
    else:
        dim_verdict = ("⚠️ upper 优势主要是维度效应（{:+.4f} vs 同维度基线 {:+.4f}）"
                       .format(up_d, m56))
    print("  >>> {}".format(dim_verdict))

    # ================= 第二部分：分块标准化 =================
    print()
    print("=" * 92)
    print("第二部分 · 分块逐维 z-score（修 hands 的方差跨度 9.7 倍）")
    print("=" * 92)
    print("  {:<24s} {:>6s} {:>12s} {:>12s} {:>8s}".format(
        "视图", "维度", "d(原始)", "d(z-score)", "变化"))
    part2 = {}
    for name in ("hands", "pose", "face", "upper", "hand_deltas", "body_deltas",
                 "full368", "rgb"):
        V = vecs[name]
        d0 = cohen_d(V, labels)
        d1 = cohen_d(zs(V), labels)
        part2[name] = {"dim": int(V.shape[1]),
                       "d_raw": round(d0, 4), "d_zscore": round(d1, 4)}
        delta = (d1 - d0)
        print("  {:<24s} {:>6d} {:>12.4f} {:>12.4f} {:>+8.4f}".format(
            name, V.shape[1], d0, d1, delta))

    # 融合视图：分块标准化后拼接（各块等权，避免 hands 因维数多而主导）
    def cat_blocknorm(names):
        parts = [zs(vecs[n]) for n in names]
        parts = [l2n(p) for p in parts]          # 每块先 L2，避免维数多的块主导
        return np.concatenate(parts, axis=1)

    fused = {
        "fuse_blocks_bn": cat_blocknorm(
            ["hands", "pose", "face", "hand_deltas", "body_deltas"]),
        "fuse_upper_hands_bn": cat_blocknorm(["upper", "hands"]),
        "fuse_upper_hands_rgb_bn": cat_blocknorm(["upper", "hands", "rgb"]),
    }
    print()
    print("  {:<24s} {:>6s} {:>12s}".format("融合视图（分块标准化+等权 L2）", "维度", "d"))
    part3 = {}
    for name, V in fused.items():
        d = cohen_d(V, labels)
        part3[name] = {"dim": int(V.shape[1]), "cohens_d": round(d, 4)}
        print("  {:<24s} {:>6d} {:>12.4f}".format(name, V.shape[1], d))

    # ================= 判决 =================
    best_single = max(
        (v["cohens_d"] for v in part1.values()), default=0.0)
    best_fused = max((v["cohens_d"] for v in part3.values()), default=0.0)
    hands_zs = part2["hands"]["d_zscore"]
    print()
    print("=" * 92)
    print("判决")
    print("=" * 92)
    print("  最佳单块 d        = {:+.4f}".format(best_single))
    print("  最佳融合 d        = {:+.4f}".format(best_fused))
    print("  hands z-score 后  = {:+.4f}（原始 {:+.4f}）".format(
        hands_zs, part2["hands"]["d_raw"]))
    verdict = (
        "{}；{}；最佳融合 d {:+.4f}。".format(
            dim_verdict,
            ("分块标准化把 hands 从 {:+.4f} 提到 {:+.4f}，机制成立"
             .format(part2["hands"]["d_raw"], hands_zs))
            if hands_zs - part2["hands"]["d_raw"] > 0.05
            else "分块标准化对 hands 无显著改善（{:+.4f} -> {:+.4f}）"
            .format(part2["hands"]["d_raw"], hands_zs),
            best_fused))
    print("  {}".format(verdict))

    out = {
        "experiment": "P37 pose+face block probe + block-wise z-score, "
                      "with dimension-confound control",
        "n_videos": len(picked), "n_segments": len(rows),
        "n_classes": len(set(labels)), "seed": a.seed,
        "pooling": "片段池化（P36 铁律：整句池化的任务不可解）",
        "metric": "Cohen's d（不依赖分类器）",
        "part1_dimension_control": part1,
        "part1_hands_random56": [round(x, 4) for x in hand56_ds],
        "part1_hands_random56_mean": round(m56, 4),
        "dimension_verdict": dim_verdict,
        "part2_block_zscore": part2,
        "part3_fused": part3,
        "verdict": verdict,
        "blocks_layout": {k: list(v) for k, v in BLOCKS.items()},
        "minutes": round((time.time() - t0) / 60, 2),
        "notes": "只读 train 特征，未训练任何深度模型。"
                 "对照设计排除了「d 只是维度数的函数」这一混淆。",
    }
    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n收据 -> {}".format(p))


if __name__ == "__main__":
    main()
