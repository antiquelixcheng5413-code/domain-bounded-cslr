#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P209 / D3 阶段一：零训练 TFD 伪边界探针（纯 CPU，与 GPU 训练可并行）。

目的：在投入 GTM / Sign Spotting 工程之前，判断 landmark48 的 TFD 边界是否
携带「CTC 峰值对齐没有覆盖」的可用区间信息。

预注册判据（启动前写死）：
  A. CTC 监督稀疏度：非 blank 主导帧占比 < 25% 且 每词被监督帧数 < 3
     ⇒ 确认「弱监督」前提成立，密集监督理论上有空间。
  B. TFD 边界质量：边界数与真实 gloss 词数的 Pearson r >= 0.5，
     且 |边界数 - 词数| 的均值 <= 2.0
     ⇒ 无参数 TFD 能恢复词级区间，D3 值得投入。
     若 B 不成立 ⇒ **D3 判负，不投入 GTM/Sign Spotting 工程**。
  C. 辅助：TFD 边界与 CTC spike 的位置重合度（对照随机基线）。

TFD 定义（本地 07_TS²-TFD / 说明.txt）：
  当前帧特征与局部时序平均特征的 L2 距离；边界 = 超自适应百分位阈值的局部最大值。
  无训练参数、复杂度 O(N)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools/blank_gov"))
import official_wer as OW  # noqa: E402

LM_DIR = REPO / "data/processed/ce-csl-landmark48"
POST = REPO / "artifacts/metrics/blank-gov/p188_logp_p184.pt"
OUT = REPO / "artifacts/metrics/blank-gov/p209-d3-tfd-probe.json"


def tfd_boundaries(x, win=5, pct=75.0):
    """x: [T, D]。返回 (tfd 信号, 边界索引 list)。"""
    T = x.shape[0]
    local = np.zeros_like(x)
    half = win // 2
    for i in range(T):
        lo, hi = max(0, i - half), min(T, i + half + 1)
        local[i] = x[lo:hi].mean(axis=0)
    sig = np.linalg.norm(x - local, axis=1)
    thr = np.percentile(sig, pct)
    peaks = [i for i in range(1, T - 1)
             if sig[i] >= sig[i - 1] and sig[i] > sig[i + 1] and sig[i] > thr]
    return sig, peaks


def ctc_spikes(logp, nonblank_thr=0.5):
    """[T, C] log-prob → (非 blank 主导帧布尔, 非 blank 概率)。"""
    p = logp.float().exp().numpy()
    nb = 1.0 - p[:, 0]
    am = logp.argmax(-1).numpy()
    spike = (am != 0) & (nb > nonblank_thr)
    return spike, nb


def hit_rate(bounds, spike, n_bins, tol=3):
    """归一化坐标下，边界 ±tol 内命中 spike 的比例。"""
    if not bounds or not spike.any():
        return 0.0
    s = np.linspace(0.0, 1.0, len(spike))
    sp = s[spike]
    b = np.asarray(bounds, dtype=float) / max(n_bins - 1, 1)
    hit = sum(1 for x in b if np.min(np.abs(sp - x)) <= tol / max(n_bins - 1, 1))
    return hit / len(b)


def main():
    d = torch.load(POST, map_location="cpu", weights_only=False)
    items = d["items"]
    print("载入后验: %d 条, img=%s" % (len(items), d.get("img")), flush=True)

    # dev 标签（真实 gloss 词数）
    import p78_train_official_tfnet as P78
    labels = P78.read_csv_split("dev.csv")
    print("dev labels: %d" % len(labels), flush=True)

    rec = []
    for sid, it in items.items():
        lm_path = LM_DIR / ("%s.npy" % sid)
        if not lm_path.exists():
            continue
        x = np.load(lm_path).astype(np.float32)
        lp = it["logp"]
        spike, nb = ctc_spikes(lp)
        sig, bounds = tfd_boundaries(x)
        ref = labels[sid][1] if sid in labels else ""
        n_words = len([t for t in ref.split("/") if t])
        rec.append({
            "sid": sid,
            "n_words": n_words,
            "T_lm": int(x.shape[0]),
            "T_ctc": int(lp.shape[0]),
            "spike_frames": int(spike.sum()),
            "spike_ratio": float(spike.mean()),
            "nb_mean": float(nb.mean()),
            "n_bounds": len(bounds),
            "hit": hit_rate(bounds, spike, x.shape[0]),
            "hit_random": hit_rate(
                list(np.random.default_rng(0).integers(0, x.shape[0], len(bounds))),
                spike, x.shape[0]),
        })

    n = len(rec)
    print("有效样本: %d" % n, flush=True)
    spike_ratio = float(np.mean([r["spike_ratio"] for r in rec]))
    words = np.array([r["n_words"] for r in rec], dtype=float)
    spikes = np.array([r["spike_frames"] for r in rec], dtype=float)
    bounds = np.array([r["n_bounds"] for r in rec], dtype=float)
    frames_per_word = float((spikes / np.maximum(words, 1)).mean())
    r_words_bounds = float(np.corrcoef(words, bounds)[0, 1])
    mae_bounds = float(np.abs(bounds - words).mean())
    hit = float(np.mean([r["hit"] for r in rec]))
    hit_rand = float(np.mean([r["hit_random"] for r in rec]))

    print("\n[A] CTC 监督稀疏度", flush=True)
    print("  非 blank 主导帧占比 %.4f" % spike_ratio, flush=True)
    print("  每词被监督帧数 %.2f (参考均长 %.2f 词)"
          % (frames_per_word, words.mean()), flush=True)
    print("\n[B] TFD 边界质量", flush=True)
    print("  平均边界数 %.2f vs 平均词数 %.2f" % (bounds.mean(), words.mean()),
          flush=True)
    print("  corr(词数, 边界数) = %.4f" % r_words_bounds, flush=True)
    print("  |边界数-词数| 均值 = %.2f" % mae_bounds, flush=True)
    print("\n[C] TFD 边界 vs CTC spike", flush=True)
    print("  hit %.4f / random %.4f / lift %.2f"
          % (hit, hit_rand, hit / max(hit_rand, 1e-8)), flush=True)

    A = (spike_ratio < 0.25 and frames_per_word < 3.0)
    B = (r_words_bounds >= 0.5 and mae_bounds <= 2.0)
    verdict = "PROCEED" if (A and B) else "STOP"
    print("\n[预注册判定] A(弱监督成立)=%s  B(TFD可恢复词区间)=%s => %s"
          % (A, B, verdict), flush=True)
    if not B:
        print("  ⇒ landmark48 的 TFD 边界不足以支撑密集帧级监督，"
              "D3 判负，不投入 GTM / Sign Spotting 工程", flush=True)

    out = {
        "tag": "p209-d3-tfd",
        "n": n,
        "posterior": str(POST.relative_to(REPO)),
        "landmark": str(LM_DIR.relative_to(REPO)),
        "pre_registered": {
            "A": "spike_ratio<0.25 且 frames_per_word<3",
            "B": "corr(词数,边界数)>=0.5 且 |边界数-词数|均值<=2",
            "rule": "A且B成立才投入 GTM/Sign Spotting；否则 D3 判负",
        },
        "A": {"spike_ratio": round(spike_ratio, 4),
              "frames_per_word": round(frames_per_word, 3),
              "mean_words": round(float(words.mean()), 3),
              "passed": bool(A)},
        "B": {"mean_bounds": round(float(bounds.mean()), 3),
              "corr_words_bounds": round(r_words_bounds, 4),
              "mae_bounds_words": round(mae_bounds, 3),
              "passed": bool(B)},
        "C": {"hit": round(hit, 4), "hit_random": round(hit_rand, 4),
              "lift": round(hit / max(hit_rand, 1e-8), 3)},
        "verdict": verdict,
        "note": "TFD 无训练参数、纯 CPU；命中率用归一化时间坐标 ±3/47 容差",
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("收据 -> %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
