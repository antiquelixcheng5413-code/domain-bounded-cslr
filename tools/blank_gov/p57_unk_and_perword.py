"""P57：三个用户提问的量化回答

Q1 为什么用 <unk>            -> 词表构建逻辑 + OOV 来源拆解
Q2 能不能去掉 <unk>          -> 不同 (min_freq, max_tokens) 组合的天花板实测
Q3 能否改成「每个词单独二分类」-> per-gloss 二分类可分性实测（片段池化）
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

def _find_repo() -> Path:
    """向上找含 src/cslr 的目录。

    这个脚本在 tools/blank_gov/ 下运行，所以仓库根是 parents[2]；
    但历史上我在这里栽过（写成 parents[1] → 指向 tools/），
    因此改成按标记文件探测，不依赖硬编码层级。
    """
    for cand in Path(__file__).resolve().parents:
        if (cand / "src" / "cslr" / "recognition"
                / "gloss_sequence.py").exists():
            return cand
    raise RuntimeError("repo root not found (looking for "
                       "src/cslr/recognition/gloss_sequence.py)")


REPO = _find_repo()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402

LM_FEAT = REPO / "artifacts/part3_features"          # 旧特征（离线已提好，够用）
UNKNOWN = "<unk>"
N_FRAMES = 48
# 片段池化后的维度块（见 memory：P36 铁律，探针必须片段池化）
S_HAND = (0, 126)
S_POSE = (126, 182)
S_FACE = (182, 186)     # presence
S_DIFF = (186, 368)


def read_csv(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(fh)}


def toks_of(g: str) -> list[str]:
    return [t.strip() for t in g.split("/") if t.strip()]


# ---------------------------------------------------------------- Q1 & Q2
def q1_q2(lab_tr: dict, lab_dv: dict, out: dict) -> None:
    print("=" * 72)
    print("Q1/Q2：<unk> 从哪来，去了会怎样")
    print("=" * 72)

    tr_tok = collections.Counter()
    for g in lab_tr.values():
        tr_tok.update(toks_of(g))
    dv_tok = collections.Counter()
    for g in lab_dv.values():
        dv_tok.update(toks_of(g))
    n_dv = sum(dv_tok.values())

    print("\n【<unk> 的两个来源】")
    print("  build_ordered_vocabulary 的过滤链：")
    print("    counts = token_counts(...)")
    print("    candidates = [t for t,c in counts if c >= min_frequency]   <- 来源A")
    print("    candidates = candidates[:max_tokens]                       <- 来源B")
    print("    tokens = ('<unk>', *candidates)                            <- <unk> 固定占 index 0")

    for cap, mf in ((300, 2), (1828, 2), (None, 2), (None, 1)):
        voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=mf,
                                         max_tokens=cap)
        T = set(voc.tokens)
        n_unk = sum(1 for t in dv_tok if t not in T)
        tok_unk = sum(c for t, c in dv_tok.items() if t not in T)
        # dev 里真没在 train 出现过的词（与词表设置无关）
        truly_oov = [t for t in dv_tok if t not in tr_tok]
        print("\n  min_frequency=%-2s max_tokens=%-6s -> 词表 %5d  "
              "dev token 判 unk %5d (%.1f%%)"
              % (mf, cap, voc.size - 1, tok_unk, 100 * tok_unk / n_dv))
        print("       dev 词种判 unk %3d；其中 train 真没出现过的 %d 个"
              % (n_unk, len(truly_oov)))

    # 无词表上限 + min_freq=1 时，dev 里还剩多少 unk
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=1,
                                     max_tokens=None)
    T = set(voc.tokens)
    resid = {t: c for t, c in dv_tok.items() if t not in T}
    print("\n  【理论天花板】min_frequency=1, max_tokens=None（用上 train 全部词种）")
    print("       词表 %d 个（CTC 输出类 = 词表+1(blank) = %d）"
          % (voc.size - 1, voc.size))
    print("       dev 残留 unk token = %d / %d = %.1f%%"
          % (sum(resid.values()), n_dv, 100 * sum(resid.values()) / n_dv))
    print("       残留词种 = %d 个：%s" % (len(resid), list(resid)[:8]))

    # 训练频次分布 —— 决定这些词学不学得动
    cnt = collections.Counter()
    for t in resid:
        cnt[tr_tok.get(t, 0)] += 1
    print("\n  残留词的 train 频次分布：")
    for k in sorted(cnt):
        print("       train 出现 %2d 次的词：%d 个" % (k, cnt[k]))
    print("  => 这些词在 train 里根本没样本，**无论词表多大都学不会**。")

    out["q1_q2"] = {
        "vocab_logic": ("candidates=[t for t,c in counts if c>=min_frequency] "
                        "-> candidates[:max_tokens] -> tokens=('<unk>',*candidates)"),
        "unk_two_sources": ["min_frequency 过滤掉低频词",
                            "max_tokens 截断长尾"],
        "train_gloss_types": len(tr_tok),
        "dev_tokens_total": n_dv,
        "settings": [],
        "theoretical_ceiling": {
            "min_frequency": 1, "max_tokens": None,
            "vocab_size": voc.size - 1,
            "ctc_classes": voc.size,
            "dev_residual_unk_tokens": sum(resid.values()),
            "dev_residual_unk_ratio": round(sum(resid.values()) / n_dv, 4),
            "residual_types": len(resid),
            "residual_train_freq_hist": {str(k): v for k, v in sorted(cnt.items())},
        },
    }
    for cap, mf in ((300, 2), (1828, 2), (None, 2), (None, 1)):
        v2, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=mf,
                                        max_tokens=cap)
        T2 = set(v2.tokens)
        out["q1_q2"]["settings"].append({
            "min_frequency": mf, "max_tokens": cap,
            "vocab_size": v2.size - 1, "ctc_classes": v2.size,
            "dev_unk_tokens": sum(c for t, c in dv_tok.items() if t not in T2),
            "dev_unk_ratio": round(sum(c for t, c in dv_tok.items()
                                       if t not in T2) / n_dv, 4),
        })


# ---------------------------------------------------------------- Q3
def segment_pool(feat48: np.ndarray, n_gloss: int) -> np.ndarray:
    """(48, 368) -> (n_gloss, 368) 片段池化（P36 铁律：必须片段池化）"""
    edges = np.linspace(0, feat48.shape[0], n_gloss + 1).round().astype(int)
    out = []
    for i in range(n_gloss):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        blk = feat48[a:b]
        out.append(np.concatenate([blk.mean(0), blk.std(0)]))
    return np.stack(out)


def q3(lab_tr: dict, lab_dv: dict, args, out: dict) -> None:
    print("\n" + "=" * 72)
    print("Q3：改成「每个词单独训练二分类」可行吗")
    print("=" * 72)
    print("""
思路：不再做 300 类联合 CTC，而是对每个 gloss w 训练一个二分类器
      「这一段视频片段里有没有 w」。判别式：
      score_w(seg) = 1 if w in seg else 0
      推理时对每段算所有 w 的分数，取 argmax（或 top-k）拼成序列。
      好处：每个二分类器只需学「w vs 其余」，负样本极多，不受类别不均衡影响。
      代价：要做 len(vocab) 次前向 —— 比一次 CTC 慢 300 倍（可并行/蒸馏）。
""")

    # ---- 采样：只用离线旧特征，够判断可行性
    rng = np.random.RandomState(0)
    tr_ids = [s for s in sorted(lab_tr) if s.endswith(tuple("0123456789"))
              and (LM_FEAT / "train" / (s + ".landmark.npy")).exists()]
    rng.shuffle(tr_ids)
    tr_ids = tr_ids[: args.n_train]

    # 先按 video 切分 holdout（避免同视频片段泄漏）
    n_ho = max(1, int(len(tr_ids) * args.holdout_frac))
    ho_ids, tr_ids = tr_ids[:n_ho], tr_ids[n_ho:]

    # ---- 收集片段（流式，避免 OOM）
    def collect(ids):
        segs, ys, groups = [], [], []
        for sid in ids:
            toks = toks_of(lab_tr[sid])
            if not toks or len(toks) > 12:
                continue
            p = LM_FEAT / "train" / (sid + ".landmark.npy")
            if not p.exists():
                continue
            f = np.load(p).astype(np.float32)
            if f.shape != (N_FRAMES, 368) or not np.isfinite(f).all():
                continue
            S = segment_pool(f, len(toks))
            segs.append(S)
            ys.append(toks)
            groups.append(np.full(len(toks), sid))
        return (np.concatenate(segs), ys, np.concatenate(groups))

    Xtr, ytr, gtr = collect(tr_ids)
    Xho, yho, gho = collect(ho_ids)
    ytr_flat = [t for row in ytr for t in row]
    yho_flat = [t for row in yho for t in row]
    print("\n  train 片段 %d（%d 视频）  holdout 片段 %d（%d 视频）"
          % (len(ytr_flat), len(tr_ids), len(yho_flat), len(ho_ids)))
    n_cls = len(set(ytr_flat) | set(yho_flat))
    print("  词种 %d" % n_cls)

    # ---- 只用分块标准化，避免 hands 方差主导
    def blockwise_zscore(Xtr, Xho):
        # P37/P38 结论：分块 z-score 后等权 L2 是最稳的视图
        def prep(A, ref):
            out = A.copy()
            for s0, s1 in ((0, 126), (126, 182), (182, 368)):
                mu = ref[:, s0:s1].mean(0)
                sd = ref[:, s0:s1].std(0) + 1e-6
                out[:, s0:s1] = (A[:, s0:s1] - mu) / sd
            n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
            return out / n
        return prep(Xtr, Xtr), prep(Xho, Xtr)

    Ztr, Zho = blockwise_zscore(Xtr, Xho)
    print("  特征：分块 z-score + L2 归一化 -> dim=%d" % Ztr.shape[1])

    # ---- 对高频词逐个做二分类探针（logistic regression, liblinear）
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    freq = collections.Counter(ytr_flat)
    targets = [t for t, c in freq.most_common(args.n_words) if c >= 15]
    print("\n  对 train 频次 >=15 的前 %d 个词各训练一个二分类器：" % len(targets))
    print("  %-8s %6s %6s %8s %8s %8s" %
          ("gloss", "pos", "neg", "AUC", "AP", "base_rate"))
    rows = []
    for w in targets:
        ytr_b = np.array([1 if t == w else 0 for t in ytr_flat])
        yho_b = np.array([1 if t == w else 0 for t in yho_flat])
        if ytr_b.sum() < 15 or yho_b.sum() == 0:
            continue
        # 负样本下采样 —— 否则 1:200 的不均衡让 liblinear 退化成常数预测
        neg_idx = np.flatnonzero(ytr_b == 0)
        rng.shuffle(neg_idx)
        neg_idx = neg_idx[: int(ytr_b.sum() * 20)]
        idx = np.concatenate([np.flatnonzero(ytr_b == 1), neg_idx])
        y = ytr_b[idx]
        if len(np.unique(y)) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0,
                                 class_weight="balanced")
        clf.fit(Ztr[idx], y)
        p = clf.decision_function(Zho)
        try:
            auc = roc_auc_score(yho_b, p)
        except ValueError:
            continue
        ap = average_precision_score(yho_b, p) if 0 < yho_b.sum() < len(yho_b) else float("nan")
        rows.append({"gloss": w, "pos_tr": int(ytr_b.sum()),
                     "pos_ho": int(yho_b.sum()), "auc": float(auc),
                     "ap": float(ap), "base_rate": float(yho_b.mean())})
        print("  %-8s %6d %6d %8.4f %8.4f %8.4f"
              % (w, ytr_b.sum(), yho_b.sum(), auc, ap, yho_b.mean()))

    aucs = np.array([r["auc"] for r in rows])
    aps = np.array([r["ap"] for r in rows])
    bases = np.array([r["base_rate"] for r in rows])
    print("\n  === 汇总（per-gloss 二分类，holdout）===")
    print("  二分类器数 %d" % len(rows))
    print("  AUC  mean %.4f  median %.4f  >0.5 的比例 %.1f%%"
          % (aucs.mean(), np.median(aucs), 100 * (aucs > 0.5).mean()))
    print("  AP   mean %.4f  median %.4f  (base_rate mean %.5f)"
          % (aps.mean(), np.median(aps), bases.mean()))
    print("  lift(AP/base) median %.2fx"
          % float(np.median(aps / np.maximum(bases, 1e-9))))

    # 对照：300 类联合分类在同一 holdout 上的 macro-AUC（P36-P38 口径）
    print("\n  === 对照：同一特征上的联合多类（片段级）===")
    from sklearn.linear_model import LogisticRegression as LR
    cls_set = sorted(set(ytr_flat))
    cls_set = [c for c in cls_set if freq[c] >= 3]
    remap = {c: i for i, c in enumerate(cls_set)}
    sub = [i for i, t in enumerate(ytr_flat) if t in remap]
    ytr_m = np.array([remap[ytr_flat[i]] for i in sub])
    Xtr_m = Ztr[sub]
    ho_idx = [i for i, t in enumerate(yho_flat) if t in remap]
    yho_m = np.array([remap[yho_flat[i]] for i in ho_idx])
    clf = LR(max_iter=3000, C=1.0, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr_m, ytr_m)
    P = clf.predict_proba(Zho[ho_idx])
    per = []
    for k in range(len(cls_set)):
        yb = (yho_m == k).astype(int)
        if 0 < yb.sum() < len(yb):
            per.append(roc_auc_score(yb, P[:, k]))
    per = np.array(per)
    print("  联合 %d 类：macro-AUC = %.4f" % (len(cls_set), per.mean()))
    print("  单词二分类 mean AUC = %.4f" % aucs.mean())
    print("  => 二分类%s联合" % ("优于" if aucs.mean() > per.mean() else "**低于**"))

    out["q3"] = {
        "n_train_segments": len(ytr_flat),
        "n_holdout_segments": len(yho_flat),
        "n_vocab": n_cls,
        "feature": "segment-pooled (mean||std), blockwise z-score + L2, dim=%d"
                   % Ztr.shape[1],
        "n_binary_probes": len(rows),
        "binary_auc_mean": round(float(aucs.mean()), 4),
        "binary_auc_median": round(float(np.median(aucs)), 4),
        "binary_auc_above_half_ratio": round(float((aucs > 0.5).mean()), 4),
        "binary_ap_mean": round(float(aps.mean()), 4),
        "base_rate_mean": round(float(bases.mean()), 5),
        "joint_multiclass": {"n_classes": len(cls_set),
                             "macro_auc": round(float(per.mean()), 4)},
        "verdict": ("binary better" if aucs.mean() > per.mean()
                    else "joint better"),
        "per_gloss": rows[:40],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=900)
    ap.add_argument("--n-words", type=int, default=40)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--out", default=str(REPO / "artifacts/metrics/blank-gov/p57.json"))
    a = ap.parse_args()

    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    out = {"generated": str(REPO / "artifacts/metrics/blank-gov/p57.json")}
    q1_q2(lab_tr, lab_dv, out)
    q3(lab_tr, lab_dv, a, out)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print("\n收据 -> %s" % a.out)


if __name__ == "__main__":
    main()