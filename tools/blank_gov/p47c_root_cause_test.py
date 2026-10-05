# -*- coding: utf-8 -*-
"""P47c · 反向验证：既然解码层救不了，那训练层能救吗？

## P47b 的结论（实测）
- 正确词在 log-prob 里的排名：top-1 仅 **10.6%**，top-5 仅 44.8%
- 高频词的平均 log-prob 是 -38 ~ -54（全部非 blank 类平均 log-prob 约 -7.8）
  → 高频词**不是分数高**，而是**其它词分数更低**
- 87.4% 的帧 argmax 是 blank，平均只有 6.1 个非 blank 帧/句

⇒ 「坍缩」的真实机制不是「高频词分数高」，而是
  **正确词的分数压不过 blank 和一堆低分词**。
  这是「输出能量不足」，不是「先验偏置」。所以解码期调先验必然无效（P47 已证）。

## 那么问题出在哪？两个可测量的候选
H1「特征里没有该词的信息」→ 无论怎么训都学不出来
H2「训练目标让模型倾向保守（少输出）」→ 加权/换目标能改善

### 测 H2：过拟合 8 条训练样本，看模型能否完美拟合
若连 8 条都拟合不到（WER > 0.2），说明容量/目标层面有硬问题；
若能拟合到 0，说明模型有能力，纯粹是泛化问题。

### 测 H1：用「词级判别探针」直接测特征是否含该词信息
在真实 dev 特征上做片段池化（不是整句池化 —— P36 铁律），
用线性探针预测 gloss 身份，看 macro-AUC。
这是不依赖任何分类器 bug 的判据。

## 本脚本做两件事
1. 8 条样本过拟合（P20 同口径）
2. 词级探针：片段池化 + 线性探针 macro-AUC（P36/P38 正确口径）
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402
from cslr.recognition.training import ctc_loss, decode_batch            # noqa: E402
from p40_rgb_main import DualInputCTC, read_csv                         # noqa: E402

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM = REPO / "artifacts/part3_features"
UNK = "<unk>"


def overfit_8():
    """测 H2：模型能否完美拟合 8 条样本。"""
    print("=== 测试 1：8 条真实样本过拟合（P20 同口径）===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)
    paths = sorted((LM / "train").glob("*.landmark.npy"))[:8]
    feats, tgts = [], []
    for p in paths:
        sid = p.name.replace(".landmark.npy", "")
        ids = voc.encode(lab_tr.get(sid, ""))
        if not ids or len(ids) > 24:
            continue
        feats.append(np.load(p).astype(np.float32))
        tgts.append([i + 1 for i in ids])
    print("  可用样本 = %d" % len(feats), flush=True)

    torch.manual_seed(0)
    x = torch.from_numpy(np.stack(feats)).to(device)
    T = max(len(t) for t in tgts)
    y = torch.zeros(len(tgts), T, dtype=torch.long, device=device)
    tl = torch.zeros(len(tgts), dtype=torch.long, device=device)
    for i, t in enumerate(tgts):
        y[i, :len(t)] = torch.tensor(t, device=device)
        tl[i] = len(t)
    il = torch.full((len(tgts),), x.shape[1], dtype=torch.long, device=device)

    model = DualInputCTC(lm_dim=368, rgb_dim=512, vocab=int(voc.size),
                         hidden=256, layers=2, dropout=0.3,
                         use_rgb=False, use_lm=True, mode="add").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(1, 601):
        model.train()
        opt.zero_grad()
        logits = model(x, il, None, None)
        loss = ctc_loss(logits, il, y, tl, il)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                lp = torch.log_softmax(
                    model(x, il, None, None).float(), -1).cpu().numpy()
                dec, _, _ = decode_batch(
                    lp, np.full((len(tgts),), lp.shape[1], dtype=np.int64), 1)
            from p40_rgb_main import levenshtein
            E = sum(levenshtein(voc.decode(list(dec[i])),
                                voc.decode(list(voc.encode(
                                    lab_tr[paths[i].name.replace(
                                        ".landmark.npy", "")]))))
                    for i in range(len(tgts)))
            N = sum(len(t) for t in tgts)
            print("  step %3d loss=%.4f  trainWER=%.4f  ndist=%d"
                  % (step, float(loss), E / max(N, 1),
                     len({t for s in dec for t in s})), flush=True)
    return True


def probe_word_level(n_videos=250):
    """测 H1：特征里有没有 gloss 身份信息（P36 正确口径：片段池化）。"""
    print("\n=== 测试 2：词级判别探针（片段池化）===", flush=True)
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)

    X, y = [], []
    n_used = 0
    for split, labs in (("train", lab_tr), ("validation", lab_dv)):
        for sid in sorted(labs)[:n_videos if split == "train" else n_videos]:
            p = LM / split / (sid + ".landmark.npy")
            if not p.exists():
                continue
            raw = labs[sid]
            toks = [t for t in raw.split("/") if t]
            if len(toks) < 2:
                continue
            f = np.load(p).astype(np.float32)            # (48, 368)
            seg = len(f) // len(toks)
            for k, t in enumerate(toks):
                if t == UNK:                # OOV 无法作为分类目标，跳过
                    continue
                a, b = k * seg, (k + 1) * seg
                if b <= a:
                    continue
                blk = f[a:b]
                # P36 铁律：片段池化，concat(mean, std)
                X.append(np.concatenate([blk.mean(0), blk.std(0)]))
                y.append(voc.index_of(t))
            n_used += 1
            if n_used >= (n_videos * 2 if split == "train" else n_videos):
                break
    X = np.stack(X).astype(np.float32)
    y = np.array(y)
    print("  片段数 = %d，特征维 = %d，类别数 = %d"
          % (len(X), X.shape[1], len(set(y.tolist()))), flush=True)

    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    n_cls = len(set(y.tolist()))
    # 只保留样本数 >= 5 的类，否则线性探针无法学
    cnt = collections.Counter(y.tolist())
    keep = {c for c, n in cnt.items() if n >= 5}
    mask = np.array([c in keep for c in y])
    X, y = X[mask], y[mask]
    remap = {c: i for i, c in enumerate(sorted(keep))}
    y = np.array([remap[c] for c in y])
    print("  过滤后：片段 = %d，类别 = %d" % (len(X), len(remap)), flush=True)
    if len(remap) < 2:
        print("  类别太少，无法训练探针")
        return None

    idx = np.random.RandomState(0).permutation(len(X))
    X, y = X[idx], y[idx]
    n_tr = int(len(X) * 0.7)
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lin = nn.Linear(X.shape[1], len(remap)).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=3e-3, weight_decay=1e-4)
    xt = torch.from_numpy(Xtr).to(dev)
    yt = torch.from_numpy(ytr).long().to(dev)
    xv = torch.from_numpy(Xte).to(dev)
    for ep in range(400):
        lin.train()
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lin(xt), yt)
        loss.backward()
        opt.step()
    lin.eval()
    with torch.no_grad():
        logits = lin(xv)
        pred = logits.argmax(1).cpu().numpy()
        prob = torch.softmax(logits, -1).cpu().numpy()
    yte_t = yte
    acc = float((pred == yte_t).mean())
    # macro-AUC（P38 铁律：主判据）
    aucs = []
    for c in range(len(remap)):
        if (yte_t == c).sum() == 0:
            continue
        s = prob[:, c]
        pos, neg = s[yte_t == c], s[yte_t != c]
        if len(pos) == 0 or len(neg) == 0:
            continue
        r = np.argsort(np.argsort(np.concatenate([pos, neg])))
        aucs.append((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
                    / (len(pos) * len(neg)))
    macro_auc = float(np.mean(aucs)) if aucs else 0.0
    base = 1.0 / len(remap)
    print("  线性探针：acc=%.4f（随机 %.4f）  **macro-AUC=%.4f**"
          % (acc, base, macro_auc), flush=True)
    print("  结论：%s"
          % ("特征含可分信息（AUC>0.6），H1 不成立"
             if macro_auc > 0.6 else
             "特征几乎无可分信息（AUC<0.6），H1 成立 —— 训练层也救不了"))
    return {"acc": round(acc, 4), "macro_auc": round(macro_auc, 4),
            "n_frag": int(len(X)), "n_cls": len(remap),
            "chance": round(base, 4)}


def main() -> None:
    overfit_8()
    r = probe_word_level()
    p = REPO / "artifacts/metrics/blank-gov/p47c-root-cause-test.json"
    p.write_text(json.dumps(
        {"experiment": "P47c", "date": "2026-10-05",
         "question": "解码层救不了，那训练层/特征层能救吗",
         "p47_negative_result": "词频先验校准单调变差，坍缩未被撬动",
         "p47b_mechanism": "正确词 top-1 仅 10.6%、top-5 仅 44.8%；"
                           "87.4% 帧 argmax 是 blank；高频词不是分数高而是别的词更低",
         "word_level_probe": r},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
