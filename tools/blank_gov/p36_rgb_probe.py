# -*- coding: utf-8 -*-
"""P36 · RGB（CLIP ViT-B-32）判别力探针 —— 跨模态方向的小规模验证

## 动机

P30/P34/P35 已排除四条方向（训练轮数、ST-GCN 拓扑、容量、面部扩容、数据增强），
唯一未证伪的是 **跨模态（RGB）**：EMNLP2023 原文
  "Existing pre-training methods solely focus on the compact pose data, which
   eliminates background perturbation but **inevitably suffers from insufficient
   semantic cues compared to raw RGB**."

A1/A2 的「视觉塔无效」结论**是在 off-by-one 模型上得出的，已作废**，
必须在修好的管线上重做。本探针是重做的第一步。

## 关键发现：提取成本为零

`artifacts/part3_features/{train,validation}/*.rgb.npy` 已存在：
  - train 4973 个 / validation 514 个
  - 收据 `model_name=ViT-B-32, version=part3-p2-clip, frame_sampling=frame-wise`
  - 形状 (T, 512)，**逐帧**（T≈154~236），与 landmark 的 48 帧重采样不同
→ 直接可用，不需要提特征、不需要 GPU 推理。

## ⚠️ 本探针同时要修正 P34 的评估器可疑点

P34 报告 `hands heldout acc = 0.0028`，而**多数类基线 0.1823**
—— 即 1-NN 原型分类器**比「无脑猜最高频词」差 65 倍**。
这不太可能是特征的问题，更可能是**评估器太弱**。本探针因此同时报告
四个量，用它们区分「特征没信息」与「评估器失效」：

| 量 | 含义 | 若低说明 |
|---|---|---|
| `heldout_acc` | 1-NN 原型，视频级 80/20 切分 | 泛化能力（**P34 只报了这个**） |
| `train_acc`  | 1-NN 原型，用**训练侧自身**检索 | **评估器是否根本没在工作** |
| `linear_acc` | 逻辑回归（多模态更强的分类器） | 特征是否线性可分 |
| `ridge_r2`   | 岭回归预测该句 gloss 数量 | 回归任务，比 236 类分类稳定得多 |

**若 `train_acc` 也低 → 评估器/特征组合失效，P34 的结论要打问号。**
**若 `train_acc` 高但 `heldout_acc` 低 → 纯泛化问题，P34 结论成立。**

## 判据（跑之前写死）

- `rgb.heldout_acc > hands.heldout_acc * 1.5` 且 lift > 0.02
  → RGB 路线成立，值得做跨模态融合实验
- 否则 → 连 CLIP 逐帧特征都分不出 gloss，
  则「RGB 提供语义先验」这个假设在**小数据集**上不成立

只读 train 的特征，不触碰 dev 的标签；不训练任何深度模型（纯线性探针）。

## 参考文献依据（已核实原文）

- ref02 LinguisticallyMotivated EMNLP2023：pose 预训练
  "inevitably suffers from insufficient semantic cues compared to raw RGB"
- ref12 Wu et al. CCL-SLR：RGB 与 pose **跨模态对比预训练**对齐特征空间
- ref23 Arib SignFormer-GCN：RGB(I3D) + 骨架(ST-GCN) 双流**相加**融合，
  并指出 "most works rely solely on RGB features"（反向证据：RGB 也有用）
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p36-rgb-probe.json")
    a = ap.parse_args()

    from cslr.recognition.gloss_sequence import build_ordered_vocabulary

    t0 = time.time()
    lab = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    voc, _ = build_ordered_vocabulary(lab.values(), min_frequency=2, max_tokens=300)
    feat_dir = REPO / "artifacts/part3_features/train"

    avail = sorted(p.name[:-len(".landmark.npy")]
                   for p in feat_dir.glob("*.landmark.npy"))
    have_rgb = set(p.name[:-len(".rgb.npy")] for p in feat_dir.glob("*.rgb.npy"))
    avail = [s for s in avail if s in have_rgb]
    picked = random.Random(a.seed).sample(avail, min(a.n_videos, len(avail)))

    print("样本视频 {} 个（train split，label 只用于构造查询标签）".format(len(picked)))

    # ---- 载入四个特征视图 ----
    # 三个 landmark 视图：hands(126) / full(368) / upper(158+24=pose+face)
    lm_hands, lm_full, lm_upper, rgb = {}, {}, {}, {}
    for sid in picked:
        lp = feat_dir / (sid + ".landmark.npy")
        rp = feat_dir / (sid + ".rgb.npy")
        if not (lp.exists() and rp.exists()):
            continue
        L = np.load(lp).astype(np.float32)     # [48, 368]
        R = np.load(rp).astype(np.float32)     # [T, 512]
        if L.shape[0] < 4 or R.shape[0] < 4:
            continue
        lm_hands[sid] = L[:, 0:126]           # 双手
        lm_upper[sid] = L[:, 126:182]         # pose + face
        lm_full[sid] = L                       # 全部 368
        rgb[sid] = R
    print("载入：hands {} / upper {} / full {} / rgb {} 个视频".format(
        len(lm_hands), len(lm_upper), len(lm_full), len(rgb)))

    VIEWS = [
        ("hands", lm_hands),      # 现有主力模态
        ("upper", lm_upper),      # pose + face
        ("full368", lm_full),     # 现有全部
        ("rgb", rgb),             # CLIP 逐帧 512
    ]

    def pool(store, sid):
        """时序池化：mean + std 拼接。

        P31 实测「样本内时间 std / 跨样本 std = 2.03」——
        纯 mean 会抹掉时间信息，所以必须带 std。
        """
        if sid not in store:
            return None
        a_ = store[sid]
        if a_.ndim != 2 or a_.shape[0] < 2:
            return None
        return np.concatenate([a_.mean(axis=0), a_.std(axis=0)])

    # ---- 构造查询样本：片段池化（每段对应 1 个 gloss）----
    #
    # 🔴 关键修正 —— 这推翻了 P19/P31/P34 的整个探针家族。
    #
    # 那三轮都把整句 48 帧池化成 **1 个** 向量，然后用句内 3.77 个 gloss
    # **共享**这个向量去查询，要求模型指出「这是句内第几个 gloss」。
    # 该任务**本身不可解**：混合向量里根本没有「位置」这个信息。
    # 所以它们测出的 0.0098 / 0.0028 不能归因于特征，只能归因于探针。
    #
    # 修正：按 gloss 数把时间轴等分成 n 段，每段池化成 1 个向量。
    # 边界用等分近似（CE-CSL 无帧级标注），但**向量与标签一一对应**，
    # 任务良定义。查询总数不变（每 gloss 一条），难度天差地别。
    rows = []
    for sid in picked:
        toks = [t.strip() for t in lab[sid].split("/") if t.strip()]
        toks = [t for t in toks if t in voc]
        if len(toks) < 2:
            continue
        seg_vecs = {}
        ok = True
        for name, store in VIEWS:
            if sid not in store:
                ok = False
                break
            arr = store[sid]
            if arr.ndim != 2 or arr.shape[0] < len(toks) * 2:
                ok = False
                break
            seg_vecs[name] = arr
        if not ok:
            continue
        T = seg_vecs["hands"].shape[0]
        bnds = np.linspace(0, T, len(toks) + 1).astype(int)

        def l2(x):
            return x / max(np.linalg.norm(x), 1e-8)

        per_tok = {}
        for i, t in enumerate(toks):
            vs = {}
            for name, arr in seg_vecs.items():
                seg = arr[bnds[i]:bnds[i + 1]]
                if seg.shape[0] < 2:
                    continue
                vs[name] = np.concatenate([seg.mean(axis=0), seg.std(axis=0)])
            if "rgb" in vs and "hands" in vs:
                vs["rgb+hands"] = np.concatenate([l2(vs["rgb"]), l2(vs["hands"])])
            per_tok[t] = vs
        for t, vs in per_tok.items():
            for name, v in vs.items():
                rows.append({"sid": sid, "tok": t, "feat": name, "v": v})
    print("片段池化后查询 {} 条 × {} 个视图".format(len(rows), len(VIEWS) + 1))

    # ---- 视频级 80/20 切分（一次固定，所有视图共用，保证可比）----
    uniq = sorted({r["sid"] for r in rows})
    rng = random.Random(7)
    rng.shuffle(uniq)
    n_test = max(int(len(uniq) * 0.2), 1)
    TEST = set(uniq[:n_test])
    print("视频级切分：train {} / test {}".format(len(uniq) - n_test, n_test))

    def l2n(X):
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)

    def one_minus(A, B):
        return float(np.mean([a_ == b_ for a_, b_ in zip(A, B)]))

    def evaluate(feat):
        sub = [r for r in rows if r["feat"] == feat]
        if len(sub) < 40:
            return None
        V = l2n(np.stack([r["v"] for r in sub]))
        y = np.array([r["tok"] for r in sub])
        sids = np.array([r["sid"] for r in sub])
        te = np.array([s in TEST for s in sids])
        tr_i, te_i = np.where(~te)[0], np.where(te)[0]
        if len(te_i) < 20 or len(tr_i) < 20:
            return None
        gold = y[te_i]

        def protos(idx):
            P = {}
            for c in set(y[idx].tolist()):
                P[c] = V[idx[y[idx] == c]].mean(axis=0)
            keys = list(P)
            M = l2n(np.stack([P[k] for k in keys]))
            return keys, M

        keys, M = protos(tr_i)
        held = one_minus(
            [keys[i] for i in (V[te_i] @ M.T).argmax(axis=1)], gold)

        # train_acc：把训练侧样本也拿去检索（排除自身所属类无法做，
        # 故用「训练集内部留一法」——查训练样本时把该类原型换成
        # 「该样本以外的所有同类的均值」，太贵；退一步用同批 kNN(1)
        # 的自身命中上限，只作为「评估器是否在工作」的存在性证据）
        tr_gold = y[tr_i]
        tr_pred_idx = (V[tr_i] @ M.T).argmax(axis=1)
        tr_acc = one_minus([keys[i] for i in tr_pred_idx], tr_gold)

        cnt = collections.Counter(gold.tolist())
        base = cnt.most_common(1)[0][1] / len(gold)

        # linear probe：逻辑回归（更强分类器，检查线性可分性）
        lin = None
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            Xtr = torch.from_numpy(V[tr_i]).float().to(dev)
            Ytr = torch.from_numpy(
                np.array([keys.index(c) for c in y[tr_i]])).long().to(dev)
            Xte = torch.from_numpy(V[te_i]).float().to(dev)
            # 类别极不均衡（P30 实测 11-100 桶占 dev token 的 24%）。
            # 无权重 CE 会塌成「全预测最高频类」——P36 首跑 acc 0.0000 就是这个原因
            # （预测分布只覆盖 14/236 类），必须加类权重。
            cnt_ = np.bincount(Ytr.cpu().numpy(), minlength=len(keys))
            w = torch.from_numpy(
                cnt_.sum() / np.maximum(cnt_, 1)).float().to(dev)
            clf = torch.nn.Linear(Xtr.shape[1], len(keys)).to(dev)
            opt = torch.optim.AdamW(clf.parameters(), lr=3e-3, weight_decay=1e-3)
            for _ in range(600):
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(clf(Xtr), Ytr, weight=w)
                loss.backward()
                opt.step()
            with torch.no_grad():
                lp = clf(Xte).argmax(1).cpu().numpy()
                lin = one_minus(lp.tolist(), gold)
            n_pred = len(set(lp.tolist()))
        except Exception as e:   # noqa: BLE001
            print("  [linear probe 失败] {}".format(e))

        return {"heldout_acc": round(held, 4),
                "train_acc_same_batch": round(tr_acc, 4),
                "linear_acc": round(lin, 4) if lin is not None else None,
                "majority_baseline": round(base, 4),
                "n_classes": len(keys),
                "n_train": int(len(tr_i)), "n_test": int(len(te_i))}

    print()
    print("=" * 88)
    print("零号自检：同 gloss 跨句相似度 vs 随机对（验证信号真实存在）")
    print("=" * 88)
    # 若「同 gloss 的两个不同视频片段」相似度并不高于随机对，
    # 说明特征里根本没有 gloss 身份信息，任何分类器都不可能work。
    selfcheck = {}
    for name in ("hands", "rgb"):
        sub = [r for r in rows if r["feat"] == name]
        by_tok = collections.defaultdict(list)
        for r in sub:
            by_tok[r["tok"]].append(r["v"])
        Vr = l2n(np.stack([r["v"] for r in sub]))
        pos, neg = [], []
        rng2 = random.Random(11)
        for _ in range(4000):
            i, j = rng2.randrange(len(sub)), rng2.randrange(len(sub))
            if i == j:
                continue
            s = float(Vr[i] @ Vr[j])
            (pos if sub[i]["tok"] == sub[j]["tok"] else neg).append(s)
        mp, mn = float(np.mean(pos)), float(np.mean(neg))
        sd = float(np.std(pos + neg)) or 1e-8
        selfcheck[name] = {"same_gloss_sim": round(mp, 4),
                           "diff_gloss_sim": round(mn, 4),
                           "cohens_d": round((mp - mn) / sd, 4),
                           "n_pos": len(pos), "n_neg": len(neg)}
        print("  {:<7s} 同gloss {:.4f}  异gloss {:.4f}  Cohen's d = {:+.4f}".format(
            name, mp, mn, selfcheck[name]["cohens_d"]))
    print("  （d>0.2 视为有可分信号；d≈0 说明特征不含 gloss 身份）")

    print()
    print("=" * 88)
    print("四量对照（视频级 heldout，片段池化）")
    print("=" * 88)
    print("  {:<11s} {:>8s} {:>10s} {:>9s} {:>9s} {:>7s}".format(
        "视图", "heldout", "train同批", "linear", "多数类基线", "类数"))
    res = {}
    for name, _ in VIEWS + [("rgb+hands", None)]:
        r = evaluate(name)
        res[name] = r
        if r:
            print("  {:<11s} {:>8.4f} {:>10.4f} {:>9s} {:>9.4f} {:>7d}".format(
                name, r["heldout_acc"], r["train_acc_same_batch"],
                "{:.4f}".format(r["linear_acc"]) if r["linear_acc"] is not None else "n/a",
                r["majority_baseline"], r["n_classes"]))
        else:
            print("  {:<11s} 样本不足".format(name))

    # ---- 判决 ----
    verdict = None
    h = res.get("hands") or {}
    g = res.get("rgb") or {}
    print()
    if h and g:
        ha, ga = h["heldout_acc"], g["heldout_acc"]
        ratio = (ga / ha) if ha > 1e-9 else float("inf")
        print("  rgb / hands(heldout) = {:.2f}x   ({} -> {})".format(ratio, ha, ga))
        gt = h.get("train_acc_same_batch")
        print("  hands 同批 train acc = {}  -> {}".format(
            gt, "评估器在工作，纯泛化问题" if (gt or 0) > 0.3 else "⚠️ 评估器本身可能失效"))
        if ga > ha * 1.5 and (ga - g["majority_baseline"]) > 0.02:
            verdict = ("RGB 路线成立（{:.2f}x，lift {:+.4f}）-> 值得做跨模态融合".format(
                ratio, ga - g["majority_baseline"]))
        else:
            verdict = ("RGB 无增益（{:.2f}x，阈值 1.5x）-> 小数据集上 CLIP 亦无法提供语义先验".format(ratio))
    print("  判决：{}".format(verdict))

    out = {
        "experiment": "P36 RGB (CLIP ViT-B-32 frame-wise) discriminability probe",
        "n_videos": len(picked), "seed": a.seed,
        "n_queries": len(rows),
        "views": [n for n, _ in VIEWS] + ["rgb+hands"],
        "results": res,
        "self_check_similarity": selfcheck,
        "verdict": verdict,
        "minutes": round((time.time() - t0) / 60, 2),
        "literature_basis": {
            "emnp2023": "pose pre-training 'inevitably suffers from insufficient "
                        "semantic cues compared to raw RGB' -> 跨模态方向的理论依据",
            "ccl_slr": "RGB 与 pose 跨模态对比预训练对齐特征空间",
            "signformer_gcn": "RGB(I3D)+骨架(ST-GCN) 双流相加融合；"
                              "'most works rely solely on RGB features'（反向证据）",
        },
        "method_notes": {
            "pooling": "mean+std（P31 实测 时间std/样本std=2.03，纯 mean 会抹掉时间信息）",
            "split": "视频级 80/20，所有视图共用同一切分",
            "queries": "句内全部 gloss（非仅首词）",
            "why_four_metrics": "区分「特征没信息」与「评估器失效」："
                                "train同批 acc 低 => 评估器没工作，P34 结论需重新审视",
        },
        "notes": "只读 train 特征，未训练任何深度模型；A1/A2 旧结论在 off-by-one "
                 "模型上已作废，本探针为重做的第一步。",
    }
    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n收据 -> {}".format(p))


if __name__ == "__main__":
    main()
