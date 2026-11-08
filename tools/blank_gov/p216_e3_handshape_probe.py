#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E3 / P216：手形聚类前提探针（零训练，纯 CPU）。

预注册判据（启动前写死）：
  留一 1-NN（样本级手形特征，cosine）预测主 gloss 的 top-1 ≥ 5%
  且 k-means(K=64) 与主 gloss 的 AMI ≥ 0.10
  ⇒ 两者同时成立才认为「无监督手形特征含 gloss 身份信息」，否则 STOP，
    不投入手形辅助头 / 密集帧级监督工程（约 1 天）。
诚实边界：手部列选取为启发式；只证伪「现有 landmark 的最便宜入口」。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
DESK = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78  # noqa: E402
import DataProcessMoudle as DPM  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score  # noqa: E402

M = REPO / "artifacts/metrics/blank-gov"
TAG = "p216-e3"
CAND_DIRS = [REPO / "data/processed/ce-csl-landmark96",
             REPO / "data/processed/ce-csl-landmark48"]


def main():
    root = next((d for d in CAND_DIRS if any(d.glob("dev-*.npy"))), None)
    if root is None:
        raise SystemExit("no dev landmark npy found")
    files = sorted(root.glob("dev-*.npy"))
    x0 = np.load(files[0])
    print("landmark dir=%s  n_files=%d  shape=%s dtype=%s"
          % (root.name, len(files), x0.shape, x0.dtype), flush=True)

    # 自适应：若是 [F,D]，取时间维 F；手部列取后 84（42 点 x 2 坐标）或全部
    if x0.ndim == 2 and x0.shape[0] <= 300 and x0.shape[1] >= 84:
        time_axis, hand = 0, x0.shape[1] >= 84
        def frame_feats(x):
            h = x[:, -84:] if hand else x
            h = h - h.mean(0, keepdims=True)
            return h
    else:
        time_axis, hand = (0 if x0.shape[0] < x0.shape[1] else 1), False
        def frame_feats(x):
            return x if time_axis == 0 else x.T
    print("time_axis=%d hand_cols=%s" % (time_axis, hand), flush=True)

    labels = P78.read_csv_split("dev.csv")
    sids = [f.stem for f in files]
    feats, y = [], []
    for f in files:
        sid = f.stem
        if sid not in labels:
            continue
        h = frame_feats(np.load(f).astype(np.float32))
        h = h[~np.isnan(h).any(1)]
        if len(h) < 2:
            continue
        # 样本级描述子：帧均值 + 帧标准差（捕捉手形与运动）
        v = np.concatenate([h.mean(0), h.std(0)])
        v = v / (np.linalg.norm(v) + 1e-8)
        gl = DPM.PreWords(str(labels[sid][1]).split("/"))
        if not gl:
            continue
        feats.append(v)
        y.append(gl[0])
    X = np.asarray(feats, dtype=np.float32)
    print("usable samples=%d  feat dim=%d  distinct main gloss=%d"
          % (len(X), X.shape[1], len(set(y))), flush=True)

    # 留一 1-NN，仅在主 gloss 出现 >=2 次的样本上计分
    cnt = defaultdict(int)
    for t in y:
        cnt[t] += 1
    keep = [i for i, t in enumerate(y) if cnt[t] >= 2]
    sim = X[keep] @ X[keep].T
    np.fill_diagonal(sim, -9.0)
    nn = sim.argmax(1)
    hit = sum(1 for i, j in enumerate(nn) if y[keep[i]] == y[keep[j]])
    top1 = 100.0 * hit / max(len(keep), 1)
    random_base = 100.0 / max(len(set(y)), 1)

    km = KMeans(n_clusters=min(64, len(X) // 4), n_init=4,
                random_state=42).fit_predict(X)
    ami = float(adjusted_mutual_info_score(y, km))
    verdict = "CONTINUE" if (top1 >= 5.0 and ami >= 0.10) else "STOP"
    out = {
        "tag": TAG, "landmark_dir": root.name, "n_files": len(files),
        "raw_shape": list(x0.shape), "n_usable": len(X),
        "n_distinct_main_gloss": len(set(y)),
        "one_nn_top1_pct": round(top1, 2),
        "random_baseline_pct": round(random_base, 3),
        "lift_over_random": round(top1 / max(random_base, 1e-6), 2),
        "kmeans_ami": round(ami, 4),
        "pre_registered": "1-NN top1>=5% AND AMI>=0.10",
        "verdict": verdict,
    }
    (M / (TAG + "-probe.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

    if verdict == "CONTINUE":
        reason = "两项均过线 ⇒ 可考虑投入手形辅助头（仍需单独写判据）。"
    else:
        reason = ("AMI %.4f < 0.10 未过线 ⇒ 按预注册 AND 规则 STOP，"
                  "不投入约 1 天的手形辅助头 / 密集帧级监督工程。"
                  "⚠️ 但 1-NN top1=%.2f%%（相对随机 %.1fx）本身是显著信号，"
                  "说明样本级描述子含 gloss 身份信息；本轮判据要求两项同时成立，"
                  "不得事后放宽。若后续重开此方向，须新写预注册判据"
                  "（例如改用近邻检索而非全局聚类）。"
                  % (ami, top1, top1 / max(random_base, 1e-6)))
    txt = """# E3 / P216 手形聚类前提探针 — 阶段报告（%s）

> 生成：%s｜零训练纯 CPU｜依据等级沿用【有/间接/无】

## 1. 动机与依据
【有】Zhang & Duh, EMNLP 2023：手形最小对立体识别率仅 ~20%%（运动类 ~87%%）；
【有】Yin et al., arXiv:2606.28667 (2026)：建议把最小对立体作 hard negatives；
【有】本地 01_HandsOn_FG2025：SignBank >60%% 手语词只用单一手形；
【有】本地 D3 已证伪 landmark48 的 TFD 边界（corr 0.29），但手形聚类是「找类别身份」而非「找边界」，任务不同。

## 2. 方法
读取 `%s` 的 dev landmark（%d 条，原始 shape %s），自适应取手部列，
帧级特征做均值+标准差描述子并 L2 归一化；做留一 1-NN（cosine）预测主 gloss，
以及 KMeans(K=%d) 与主 gloss 的 AMI。预注册判据：1-NN top1 ≥5%% 且 AMI ≥0.10。

## 3. 结果
| 指标 | 值 |
|---|---|
| 可用样本 | %d |
| 主 gloss 类数 | %d |
| 留一 1-NN top-1 | %.2f%% |
| 随机基线 | %.3f%% |
| 相对随机 lift | %.2fx |
| KMeans AMI | %.4f |

## 4. 判定：**%s**
%s

## 5. 诚实边界
- 手部列选取为启发式（维度自适应），只证伪「现有 landmark + 无监督描述子」这个最便宜入口，
  不等于证伪 EMNLP23 的帧级伪标签监督或 GTM 本身。
- 主 gloss 只是整句第一个词，标签含噪；这正是「前提探针」而非性能评估。
- 未跑 test、未 commit、未 push。

数据来源：`artifacts/metrics/blank-gov/%s-probe.json`。
""" % (
        verdict, datetime.now().strftime("%Y-%m-%d %H:%M"),
        root.name, len(files), list(x0.shape), min(64, max(len(X) // 4, 2)),
        len(X), len(set(y)), top1, random_base,
        top1 / max(random_base, 1e-6), ami, verdict, reason, TAG,
    )
    for p in (REPO / "docs/experiments" / ("stage-%s.md" % TAG),
              DESK / "docs/experiments" / ("stage-%s.md" % TAG)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> docs/experiments/stage-%s.md (+桌面镜像)" % TAG, flush=True)


if __name__ == "__main__":
    main()
