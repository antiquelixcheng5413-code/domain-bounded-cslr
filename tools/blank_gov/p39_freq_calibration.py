# -*- coding: utf-8 -*-
"""P39 · 用 macroAUC 校准 P30 的「中频词样本量不足」结论

## 为什么做这个实验

P30 得出过一个影响很大的结论：
    「训练频次 11-100 的词 WER 0.9178，是 dev 误差的主要来源；
      100+ 桶只有 0.4003。=> 真正的瓶颈是中频词的样本量。」

该结论已写进长期记忆，并被 P29/P34/P35 等多轮实验当作前提引用。
**但它有两处从未排除的口径漏洞。**

### 漏洞 ① 用 WER 而不是 macroAUC

P30 收据 `reference_used: "folded tokens"`，分桶指标是 `wer`。
P38 实测证明 WER/acc 在本任务上会**掩盖真实信号**：
    full368  macroAUC = 0.6091（远高于随机 0.5）
    full368  heldout acc = 0.0260（看着像随机）
若 WER 口径与 acc 一样失真，那 11-100 桶的 0.9178 可能是口径产物。

### 漏洞 ② OOV 桶的 0.1707 是折叠产物

P9 实测：cap300 词表下 dev OOV 占 30.4%，
宽松 WER 虚高上界 **0.1650**（OOV 全折叠成同一个 `<unk>`，吐一个即命中）。
所以「OOV 桶很准」不代表模型认出了未见词。

## 预注册判据（跑之前写死）

在同一套已验证的探针上（片段池化 + 视频级 80/20 heldout + 三口径），
按训练频次分桶，观察：

- 若 **11-100 桶的 macroAUC 显著低于 100+ 桶** → P30 结论在稳健口径下成立
- 若 **11-100 桶的 macroAUC 与 100+ 桶相当或更高** → **结论翻转**，
  「误差集中在中频词」是 WER 口径的假象

判据阈值：两桶 macroAUC 差值 < 0.02 视为「相当」（即无区分）。

## 为什么这个问题值得花半小时

它是目前对「为什么 dev WER 上不去」最有力的解释，
且被多轮实验当作前提。若翻转，六轮实验的归因都要重写。

**零训练成本**：全部基于已有 `.landmark.npy`，不训练任何深度模型。

## 口径说明

- 用 train split 的特征与标签，不触碰 dev/test
- 视频级 80/20 切分，train 侧建原型，test 侧评估（泛化口径）
- 片段池化：**按 gloss 数等分时间轴，每段对应 1 个 gloss**（P36 铁律，
  绝不能用整句池化 —— 那会让句内多个 gloss 共用混合向量，任务不可解）
- 同时报 WER / balanced acc / macroAUC 三口径，便于与 P30 直接对照
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


def read_csv(p):
    rows = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["Number"]] = r["Gloss"]
    return rows


def l2n(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def macro_auc(scores, pos_mask):
    """二分类 macro-AUC：正类=目标类，负类=其余，按分数算秩。

    自己实现避免 sklearn 依赖；对类别极不均衡稳健（P30 的教训）。
    """
    pos = np.asarray(scores)[pos_mask]
    neg = np.asarray(scores)[~pos_mask]
    if len(pos) == 0 or len(neg) == 0:
        return None
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
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p39-freq-calibration.json")
    a = ap.parse_args()

    from cslr.recognition.gloss_sequence import build_ordered_vocabulary

    t0 = time.time()
    lab = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    voc, counts = build_ordered_vocabulary(
        lab.values(), min_frequency=2, max_tokens=300)
    feat = REPO / "artifacts/part3_features/train"
    avail = sorted(p.name[:-len(".landmark.npy")] for p in feat.glob("*.landmark.npy"))
    picked = random.Random(a.seed).sample(avail, min(a.n_videos, len(avail)))
    print("词表 {}，视频 {} 个（train split）".format(voc.size, len(picked)))

    # ---- 片段池化（P36 铁律）----
    #
    # ⚠️ 关键：不能像第一版那样用 `t in voc` 过滤。
    # 实测 cap300 词表下 33.3% 的 CSV token 不在词表内（941/2824）。
    # 若过滤掉，OOV 桶会整体消失，「OOV 桶 WER 0.1707 是否折叠产物」
    # 这条（P30 漏洞 ②）就无法检验。
    # 正确做法：全部 gloss 都建查询，OOV 归入自己的桶。
    rows = []
    for sid in picked:
        toks = [t.strip() for t in lab[sid].split("/") if t.strip()]
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
            seg = L[bn[i]:bn[i + 1]]
            rows.append({
                "sid": sid, "tok": t,
                "v": np.concatenate([seg.mean(axis=0), seg.std(axis=0)]),
            })
    print("片段 {} 条，原始 gloss 类 {}".format(
        len(rows), len({r["tok"] for r in rows})))

    # ---- 按训练频次分桶 ----
    # OOV = 该 gloss 在训练集中出现 0 次（即建词表时被排除的）
    def bucket_of(tok):
        f = counts.get(tok, 0)
        if f >= 101:
            return "100+"
        if f >= 11:
            return "11-100"
        if f >= 3:
            return "3-10"
        if f >= 1:
            return "1-2"
        return "OOV"

    for r in rows:
        r["bucket"] = bucket_of(r["tok"])
    dist = collections.Counter(r["bucket"] for r in rows)
    tot = len(rows)
    print("\n片段的训练频次分布（未过滤 OOV）:")
    for k in ("100+", "11-100", "3-10", "1-2", "OOV"):
        if dist.get(k):
            print("  {:<8s} {:5d}  {:5.1f}%".format(k, dist[k], dist[k] / tot * 100))

    # ---- 视频级 80/20 切分 ----
    uniq = sorted({r["sid"] for r in rows})
    sh = uniq[:]
    random.Random(7).shuffle(sh)
    TEST = set(sh[:max(int(len(uniq) * 0.2), 1)])
    te_mask = np.array([r["sid"] in TEST for r in rows])
    tr_i, te_i = np.where(~te_mask)[0], np.where(te_mask)[0]
    y = np.array([r["tok"] for r in rows])
    V = l2n(np.stack([r["v"] for r in rows]))
    print("\n视频级切分：train {} 片段 / test {} 片段".format(len(tr_i), len(te_i)))

    # ---- 训练侧建原型 ----
    # 只保留训练侧出现 >=2 次的类（原型至少要 2 个样本才有意义）
    cnt_tr = collections.Counter(y[tr_i].tolist())
    keys = sorted(c for c, n in cnt_tr.items() if n >= 2)
    P = {c: V[tr_i[y[tr_i] == c]].mean(axis=0) for c in keys}
    M = l2n(np.stack([P[c] for c in keys]))
    kidx = {c: i for i, c in enumerate(keys)}
    print("可评估类 {} 个".format(len(keys)))

    sim = V[te_i] @ M.T
    pred = [keys[i] for i in sim.argmax(axis=1)]
    gold = y[te_i]
    gbucket = np.array([bucket_of(g) for g in gold])

    # ---- 三口径分桶 ----
    res = {}
    print()
    print("=" * 92)
    print("分频次桶的三口径对照（test 集，{} 个可评估类）".format(len(keys)))
    print("=" * 92)
    print("  {:<8s} {:>6s} {:>9s} {:>10s} {:>11s}".format(
        "桶", "样本", "acc", "balanced", "macroAUC"))
    for b in ("100+", "11-100", "3-10", "1-2", "OOV"):
        m = gbucket == b
        n = int(m.sum())
        if n < 10:
            if n:
                print("  {:<8s} {:>6d}  样本不足".format(b, n))
            continue
        p_ = np.array(pred)[m]
        g_ = gold[m]
        acc = float(np.mean([a_ == b_ for a_, b_ in zip(p_, g_)]))
        bal = float(np.mean([
            np.mean([a_ == b_ for a_, b_ in zip(p_, g_) if b_ == c])
            for c in set(g_.tolist())
        ]))
        aucs = []
        for c in set(g_.tolist()):
            mc = m & (gold == c)
            k = int(mc.sum())
            if k < 3 or c not in kidx:
                continue
            if int((~mc).sum()) < 1:
                continue          # 负类为空时 AUC 无定义，跳过而非报错
            # sim 是 test 集 (n_test, n_classes)，mc 是 test 集上的布尔掩码 —— 对齐
            a_auc = macro_auc(sim[:, kidx[c]], mc)
            if a_auc is not None:
                aucs.append(a_auc)
        mauc = float(np.mean(aucs)) if aucs else None
        res[b] = {"n_test": n, "acc": round(acc, 4), "balanced_acc": round(bal, 4),
                  "macro_auc": round(mauc, 4) if mauc is not None else None,
                  "n_classes_scored": len(aucs)}
        print("  {:<8s} {:>6d} {:>9.4f} {:>10.4f} {:>11s}".format(
            b, n, acc, bal,
            "{:.4f}".format(mauc) if mauc is not None else "n/a"))

    # ---- 与 P30 对照 ----
    print()
    print("=" * 92)
    print("与 P30 的 WER 口径对照")
    print("=" * 92)
    p30 = {"100+": 0.4003, "11-100": 0.9178, "OOV": 0.1707}
    print("  {:<8s} {:>12s} {:>12s} {:>10s}".format(
        "桶", "P30 WER", "P39 acc", "P39 macroAUC"))
    for b in ("100+", "11-100", "OOV"):
        r = res.get(b)
        print("  {:<8s} {:>12.4f} {:>12s} {:>10s}".format(
            b, p30[b],
            "{:.4f}".format(r["acc"]) if r else "n/a",
            "{:.4f}".format(r["macro_auc"]) if r and r["macro_auc"] is not None else "n/a"))

    # ---- 判决 ----
    verdict = None
    diff = None
    r_hi, r_mid = res.get("100+"), res.get("11-100")
    if r_hi and r_mid and r_hi["macro_auc"] is not None and r_mid["macro_auc"] is not None:
        diff = r_mid["macro_auc"] - r_hi["macro_auc"]
        print()
        print("  11-100 桶 macroAUC - 100+ 桶 macroAUC = {:+.4f}".format(diff))
        if diff < -0.02:
            verdict = ("P30 结论在稳健口径下【成立】：11-100 桶 macroAUC 显著更低"
                       "（{:+.4f} < -0.02），误差确实集中在中频词".format(diff))
        elif diff > 0.02:
            verdict = ("⚠️ 【结论翻转】11-100 桶 macroAUC 反而更高（{:+.4f} > +0.02）"
                       "=> 「误差集中在中频词」是 WER 口径的假象".format(diff))
        else:
            verdict = ("两桶 macroAUC 相当（差 {:+.4f}，|差|<0.02）"
                       "=> P30 的 WER 口径无法区分两桶，结论不成立于稳健口径".format(diff))
        print("  {}".format(verdict))
    else:
        print("\n  样本不足，无法判决")

    out = {
        "experiment": "P39 recalibrate P30's 'mid-frequency words are the bottleneck' "
                      "under macroAUC (P30 used WER, which P38 showed is misleading here)",
        "n_videos": len(picked), "n_segments": len(rows),
        "n_classes_scored": len(keys), "vocab_size": int(voc.size),
        "seed": a.seed,
        "pooling": "片段池化（P36 铁律）：按 gloss 数等分时间轴，每段对应 1 个 gloss",
        "split": "视频级 80/20，train 侧建原型，test 侧评估（泛化口径）",
        "bucket_distribution": {k: {"n": dist[k], "share": round(dist[k] / tot, 4)}
                                for k in ("100+", "11-100", "3-10", "1-2", "OOV")
                                if dist.get(k)},
        "three_metrics_by_bucket": res,
        "p30_wer_for_comparison": p30,
        "macroauc_diff_mid_minus_high": round(diff, 4) if diff is not None else None,
        "verdict": verdict,
        "criterion": "11-100 桶 macroAUC 显著低于 100+ 桶（差 < -0.02）-> P30 成立；"
                     "差 > +0.02 -> 结论翻转；|差| < 0.02 -> 口径不区分",
        "all_data_real": True,
        "reads_test_split": False,
        "no_training": "本实验不训练任何深度模型，全部基于已有 .landmark.npy",
        "minutes": round((time.time() - t0) / 60, 2),
        "p30_holes_being_tested": [
            "① P30 分桶指标是 WER；P38 实测 WER/acc 在本任务上掩盖真实信号"
            "（full368 macroAUC 0.6091 vs acc 0.0260）",
            "② P30 的 OOV 桶 WER 0.1707 是折叠产物；"
            "P9 实测 dev OOV 占 30.4%，宽松 WER 虚高上界 0.1650",
        ],
    }
    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n收据 -> {}".format(p))


if __name__ == "__main__":
    main()
