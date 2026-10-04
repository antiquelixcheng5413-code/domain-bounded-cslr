# -*- coding: utf-8 -*-
"""P31 · 原始特征层诊断：MediaPipe landmark 到底缺了什么（重提特征前必做）

## 动机

P30 确诊瓶颈是「中频词样本量」（11-100 桶 WER 0.9178），
P29 证实 ST-GCN 拓扑无效。既然模型侧与容量侧都到顶，
下一步必须回到**原始数据处理**。但重提特征前必须先量化「缺什么」，
否则会白跑（mediapipe 在 WSL 下曾多次 `E_UNEXPECTED` 崩溃）。

## 三个可判决的量（全部真实数据，无重提特征）

### 量1 · 检出率：多少帧根本没有手部坐标

已有特征带 presence 标记（`[182:186]`）。P30 附带实测（800 条 train）：
```
pose 覆盖率 mean 0.7630   全部帧都检出的样本仅 27.6%
左手覆盖率 mean 0.6536   全部帧都检出的样本仅 17.0%
```
**需精确化**：按 token 统计，有多少 dev token 落在「手缺失」的帧区间里。
若一个词的 48 帧里有 35% 帧手部是全零，那这个词根本不可能被认出来 ——
**这是数据层缺陷，不是模型问题**。

### 量2 · 手部静止度：landmark 是否真的在动

若 landmark 相对 pose 原点的位移很小，说明 MediaPipe 输出被过度归一化
（`extractor.py:_normalization_from_pose` 用肩距做 scale），
运动信息被压扁 → 直接对应 P30 的「学不出运动不变表示」。

量：帧间位移的均值/方差，以及「相邻帧几乎不动」的比例。

### 量3 · 静态可分性：单帧特征能否区分不同 gloss

用 mean-pooled 特征做 1-NN（train 原型 → dev 样本）。
P19 测过 0.0762（很��），但那次 ref 用首词、且没区分「手是否缺失」。
**若「手完整帧」的样本 1-NN 显著高于「手缺失帧」，
则说明检出率就是瓶颈。**

## 判据（跑之前写死）

- 若 dev token 中 **>30% 落在手部缺失区间** → 检出率是硬瓶颈，
  **必须重提特征**（且要走可靠路径，不是 WSL+mediapipe）
- 若「手完整样本」的 1-NN 准确率 **≥2x 于「手缺失样本」**
  → 缺失帧是主因，重提特征有明确收益
- 若两者都不成立 → 特征层不是瓶颈，数据增强才是方向

只读 train/dev，**不触碰 test split**。
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary

HANDS = slice(0, 126)
POSE = slice(126, 158)
FACE = slice(158, 182)
PRES = slice(182, 186)


def read_split(split):
    table = {"train": "train.csv", "validation": "dev.csv"}
    if split not in table:
        raise ValueError("split 必须是 {} 之一".format(sorted(table)))
    out = {}
    with open(REPO / "data/raw/CE-CSL/label" / table[split],
              newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["Number"]] = r["Gloss"]
    return out


def link_dir(split):
    root = REPO / ".vocab_link" / split
    root.mkdir(parents=True, exist_ok=True)
    src = REPO / "artifacts/part3_features" / split
    for p in sorted(src.glob("*.landmark.npy")):
        sid = p.name[: -len(".landmark.npy")]
        dst = root / (sid + ".npy")
        if not dst.exists():
            try:
                dst.symlink_to(p)
            except OSError:
                pass
    return root


def load(split_root, sid):
    return np.load(split_root / (sid + ".npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--out",
                    default="artifacts/metrics/blank-gov/p31-feature-layer.json")
    a = ap.parse_args()

    tr_labels = read_split("train")
    va_labels = read_split("validation")
    tr_root = link_dir("train")
    va_root = link_dir("validation")
    voc, counts = build_ordered_vocabulary((g for g in tr_labels.values()),
                                           min_frequency=2, max_tokens=300)
    tr_ids = sorted(k for k in tr_labels if (tr_root / (k + ".npy")).exists())
    va_ids = sorted(k for k in va_labels if (va_root / (k + ".npy")).exists())
    print("train {} / dev {} 可用特征".format(len(tr_ids), len(va_ids)))

    # ---------------- 量1 · 检出率 ----------------
    print()
    print("=" * 74)
    print("量1 · 检出率（presence 标记）与「手缺失帧」占比")
    print("=" * 74)
    tok_total = tok_bad = 0
    sent_all_present = 0
    frac_bad_list = []
    per_sample = []
    for sid in va_ids:
        x = load(va_root, sid)
        pres = x[:, PRES]                     # [T,4] = pose,L,R,face
        t = len(x)
        lh = pres[:, 1]
        rh = pres[:, 2]
        # 双手都缺的帧
        both_missing = ((lh == 0) & (rh == 0))
        frac_missing = float(both_missing.mean())
        frac_bad_list.append(frac_missing)
        n_tok = len([t2 for t2 in va_labels[sid].split("/") if t2.strip()])
        tok_total += n_tok
        tok_bad += n_tok * frac_missing
        if frac_missing == 0.0:
            sent_all_present += 1
        per_sample.append({"sid": sid, "frac_frames_both_hands_missing": frac_missing})
    frac_arr = np.array(frac_bad_list)
    print("  双手全缺帧占比  mean {:.4f}  median {:.4f}  p90 {:.4f}".format(
        frac_arr.mean(), np.median(frac_arr), np.percentile(frac_arr, 90)))
    print("  完全无缺失帧的句子 {}/{} = {:.1%}".format(
        sent_all_present, len(va_ids), sent_all_present / max(len(va_ids), 1)))
    print("  **按 token 加权：{}/{} = {:.1%} 的 dev token 落在「至少一帧双手全缺」的句子里**".format(
        int(tok_bad), tok_total, tok_bad / max(tok_total, 1)))
    # 分布
    bins = collections.Counter()
    for f in frac_bad_list:
        b = "0" if f == 0 else "0-0.1" if f < .1 else "0.1-0.3" if f < .3 else \
            "0.3-0.6" if f < .6 else ">0.6"
        bins[b] += 1
    print("  句子分布:", dict(bins))

    # ---------------- 量2 · 运动量 ----------------
    print()
    print("=" * 74)
    print("量2 · landmark 运动量（过度归一化会压扁运动信息）")
    print("=" * 74)
    sub = tr_ids[: a.n_train]
    disp_all = []
    near_static = 0
    total_pairs = 0
    for sid in sub:
        x = load(tr_root, sid)
        h = x[:, HANDS].reshape(len(x), 2, 21, 3)
        pr = x[:, PRES]
        for f in range(len(x) - 1):
            # 只在双手都检出的帧对上量位移
            if pr[f, 1] == 0 or pr[f, 2] == 0 or pr[f + 1, 1] == 0 or pr[f + 1, 2] == 0:
                continue
            d = np.linalg.norm(h[f + 1] - h[f], axis=-1)   # [2,21]
            v = float(d.mean())
            disp_all.append(v)
            total_pairs += 1
            if v < 1e-3:
                near_static += 1
    if disp_all:
        da = np.array(disp_all)
        print("  帧间平均位移（归一化后单位）: mean {:.5f} median {:.5f} p90 {:.5f}".format(
            da.mean(), np.median(da), np.percentile(da, 90)))
        print("  位移 < 1e-3 的比例（几乎静止）: {:.4f}".format(
            near_static / max(total_pairs, 1)))
        print("  注：extractor.py 用肩距归一化，若该值过小说明运动被压扁")

    # ---------------- 量3 · 静态可分性，按检出状态分组 ----------------
    print()
    print("=" * 74)
    print("量3 · 1-NN 静态可分性（按「手是否完整」分组）")
    print("=" * 74)

    def proto(ids, root, min_count=2):
        acc = collections.defaultdict(list)
        for sid in ids:
            x = load(root, sid)
            pres = x[:, PRES]
            if (pres[:, 1] == 0).all() or (pres[:, 2] == 0).all():
                continue          # 手完全不存在的样本不参与原型
            v = x[:, HANDS].mean(axis=0)
            toks = [t.strip() for t in
                    (tr_labels if root == tr_root else va_labels)[sid].split("/") if t.strip()]
            for t in toks:
                if t in voc:
                    acc[t].append(v)
        return {k: np.mean(np.stack(vs), axis=0) for k, vs in acc.items() if len(vs) >= min_count}

    ptr = proto(tr_ids, tr_root)
    print("  原型数（仅用手完整的 train 样本）{}".format(len(ptr)))
    keys = list(ptr.keys())
    M = np.stack([ptr[k] for k in keys])
    M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-8)
    kset = set(keys)

    groups = {"hand_complete": ([], []), "hand_missing": ([], [])}
    for sid in va_ids:
        x = load(va_root, sid)
        pres = x[:, PRES]
        frac_missing = float(((pres[:, 1] == 0) & (pres[:, 2] == 0)).mean())
        v = x[:, HANDS].mean(axis=0)
        g = "hand_missing" if frac_missing > 0 else "hand_complete"
        toks = [t.strip() for t in va_labels[sid].split("/") if t.strip()]
        if toks:
            groups[g][0].append(v)
            groups[g][1].append(toks[0])

    out_groups = {}
    for g, (vs, ts) in groups.items():
        if not vs:
            continue
        Q = np.stack(vs)
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        sim = Q @ M.T
        idx = sim.argmax(axis=1)
        correct = n = 0
        for row, gt in enumerate(ts):
            if gt not in kset:
                continue
            n += 1
            if keys[idx[row]] == gt:
                correct += 1
        acc = correct / max(n, 1)
        out_groups[g] = {"n": n, "acc": round(acc, 4)}
        print("  {:<14s} n={:4d}  1-NN acc={:.4f}".format(g, n, acc))

    acc_c = out_groups.get("hand_complete", {}).get("acc", 0.0)
    acc_m = out_groups.get("hand_missing", {}).get("acc", 0.0)
    ratio = (acc_c / acc_m) if acc_m > 1e-9 else float("inf")
    print("  完整/缺失 组准确率之比 = {:.2f}x".format(ratio))

    # ---------------- 判决 ----------------
    tok_frac = tok_bad / max(tok_total, 1)
    print()
    print("=" * 74)
    print("判决（判据跑前写死）")
    print("=" * 74)
    verdicts = []
    if tok_frac > 0.30:
        verdicts.append("检出率是硬瓶颈（{:.1%} dev token 落在双手全缺的句子上）-> 必须重提特征".format(tok_frac))
    else:
        verdicts.append("检出率不是主要瓶颈（{:.1%} <= 30%）".format(tok_frac))
    if acc_m > 1e-9 and ratio >= 2.0:
        verdicts.append("缺失帧组的 1-NN 显著更差（{:.2f}x）-> 补齐检出有明确收益".format(ratio))
    else:
        verdicts.append("检出状态对可分性影响不大（{:.2f}x < 2）-> 重提特征收益不确定".format(ratio))
    verdict = " | ".join(verdicts)
    print("  " + verdict)

    receipt = {
        "experiment": "P31 what is missing in the raw landmark features",
        "purpose": "quantify the gap before spending cost on re-extraction",
        "criterion": {
            "detection_bottleneck": ">30% of dev tokens fall in sentences with both hands missing",
            "separability_gain": "1-NN acc ratio complete/missing >= 2"},
        "all_data_real": True, "reads_test_split": False,
        "q1_detection": {
            "mean_frac_both_hands_missing": round(float(frac_arr.mean()), 4),
            "median": round(float(np.median(frac_arr)), 4),
            "p90": round(float(np.percentile(frac_arr, 90)), 4),
            "sentences_with_no_missing": sent_all_present,
            "n_sentences": len(va_ids),
            "token_weighted_frac": round(tok_frac, 4),
            "sentence_distribution": dict(bins),
        },
        "q2_motion": {
            "mean_frame_displacement": round(float(np.mean(disp_all)), 5) if disp_all else None,
            "median_frame_displacement": round(float(np.median(disp_all)), 5) if disp_all else None,
            "frac_static_lt_1e-3": round(near_static / max(total_pairs, 1), 4) if disp_all else None,
            "n_pairs": total_pairs,
            "note": "extractor.py normalises by shoulder distance; very small values mean "
                    "motion information is compressed",
        },
        "q3_separability": {"by_group": out_groups, "acc_ratio_complete_over_missing": round(ratio, 3)},
        "verdict": verdict,
    }
    outp = REPO / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(outp))


if __name__ == "__main__":
    main()
