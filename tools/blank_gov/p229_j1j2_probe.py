#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P229 / J1+J2：第六轮方向的**零成本前提诊断**（纯 CPU，不启动实验、不占 GPU、不跑 test）。

J1 —— CTC 强制对齐能否切出可用的 per-gloss 片段（K1 语料合成的一票否决）。
J2 —— 句子级置信度能否支撑选择性预测/拒识（K3 的一票否决）。

判据写死在 docs/planning/OPT-DIRECTIONS-6-2026-11-08.md §3，早于看到结果。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(p))

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402
import DataProcessMoudle as DPM  # noqa: E402

M = REPO / "artifacts/metrics/blank-gov"
TAG = "p229-j1j2"
# J1 判据：区间长度中位（输入帧）>=16 且 >=8 输入帧的 gloss 占比 >=0.80
J1_MIN_MEDIAN_FRAMES = 16
J1_MIN_COVERAGE = 0.80
# 编码器时间步 -> 输入帧换算（dev 中位输入 181 帧 / T_conv 中位 46 步）
STEP_TO_FRAME = 181.0 / 46.0
# J2 判据：Spearman |rho| >=0.30 且 70% coverage 下 WER 相对下降 >=25%
J2_MIN_RHO = 0.30
J2_COVERAGE = 0.70
J2_MIN_WER_DROP = 0.25


def spearman(x, y):
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        return (np.bincount(inv, weights=r) / cnt)[inv]
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def forced_align(logp, wid):
    """标准 CTC Viterbi 强制对齐：扩展状态 [0,w0,0,w1,...,0]，返回每帧状态索引。"""
    L = len(wid)
    S = 2 * L + 1
    ext = [0] * S
    for i, w in enumerate(wid):
        ext[2 * i + 1] = w
    T = logp.shape[0]
    if T < S:
        return None
    dp = np.full((T, S), -np.inf, dtype=np.float64)
    bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = logp[0, ext[0]]
    dp[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        emis = np.asarray([logp[t, e] for e in ext], dtype=np.float64)
        for s in range(S):
            best, arg = dp[t - 1, s], 0
            if s >= 1 and dp[t - 1, s - 1] > best:
                best, arg = dp[t - 1, s - 1], 1
            if (s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2]
                    and dp[t - 1, s - 2] > best):
                best, arg = dp[t - 1, s - 2], 2
            dp[t, s] = best + emis[s]
            bp[t, s] = arg
    s = int(np.argmax(dp[T - 1, [-1, -2]]))
    s = S - 1 if s == 0 else S - 2
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = s
    for t in range(T - 1, 0, -1):
        s = s - int(bp[t, s])
        path[t - 1] = s
    return path, ext


def main():
    d = torch.load(M / "p188_logp_p184.pt", map_location="cpu", weights_only=False)
    labels = P78.read_csv_split("dev.csv")
    word2idx, n, _ = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    clean = json.loads((M / "p227-h2a-clean-uwer.json").read_text(encoding="utf-8"))
    per_item = {x["sid"]: x for x in clean["per_item"]}

    spike_counts, span_steps, span_frames = [], [], []
    sent_conf, sent_wer = [], []
    n_fail = 0
    for sid, it in d["items"].items():
        if sid not in labels:
            continue
        ref = labels[sid][1]
        toks = OW.split_glued(OW.split_gloss_sequence(ref, OW.CFG))
        wid = [word2idx[t] for t in toks if t in word2idx]
        logp = it["logp"].numpy().astype(np.float64)
        if len(wid) == 0:
            continue
        fa = forced_align(logp, wid)
        if fa is None:
            n_fail += 1
            continue
        path, ext = fa
        # gloss i 的 spike 位置与占据帧数
        pos, cnt = [], []
        for i in range(len(wid)):
            f = np.where(path == 2 * i + 1)[0]
            cnt.append(len(f))
            pos.append(float(f[0]) if len(f) else -1.0)
        spike_counts.extend(cnt)
        # 用有效 spike 位置的中点切分连续区间；缺 spike 的 gloss 用线性插值补位置
        good = [i for i, p in enumerate(pos) if p >= 0]
        if not good:
            n_fail += 1
            continue
        for i in range(len(wid)):
            if pos[i] < 0:
                # 最近的左右已知 spike 之间等分
                left = max([j for j in good if j < i], default=None)
                right = min([j for j in good if j > i], default=None)
                if left is None and right is None:
                    pos[i] = logp.shape[0] / 2
                elif left is None:
                    pos[i] = max(0.0, pos[right] - (right - i))
                elif right is None:
                    pos[i] = pos[left] + (i - left)
                else:
                    pos[i] = pos[left] + (pos[right] - pos[left]) * (i - left) / (right - left)
        edges = [0.0]
        for i in range(len(wid) - 1):
            edges.append((pos[i] + pos[i + 1]) / 2.0)
        edges.append(float(logp.shape[0]))
        for i in range(len(wid)):
            span = max(edges[i + 1] - edges[i], 0.0)
            span_steps.append(span)
            span_frames.append(span * STEP_TO_FRAME)

        # ---- J2 句子级置信度：每帧最大 log-prob 的均值 ----
        if sid in per_item:
            pi = per_item[sid]
            sent_conf.append(float(logp.max(axis=1).mean()))
            sent_wer.append(pi["edits"] / max(pi["ref_n"], 1))

    sp = np.asarray(spike_counts, float)
    ss = np.asarray(span_steps, float)
    sf = np.asarray(span_frames, float)
    med_frames = float(np.median(sf))
    coverage = float((sf >= 8).mean())
    j1 = bool(med_frames >= J1_MIN_MEDIAN_FRAMES
              and coverage >= J1_MIN_COVERAGE)

    conf = np.asarray(sent_conf, float)
    wer = np.asarray(sent_wer, float)
    rho = spearman(conf, wer)          # 期望负相关：置信度越高 WER 越低
    order = np.argsort(-conf)          # 置信度从高到低
    k = int(len(order) * J2_COVERAGE)
    base_wer = float(wer.mean())
    kept_wer = float(wer[order[:k]].mean())
    drop = (base_wer - kept_wer) / max(base_wer, 1e-9)
    j2 = bool(abs(rho) >= J2_MIN_RHO and rho < 0 and drop >= J2_MIN_WER_DROP)

    out = {
        "tag": TAG,
        "note": ("零成本 CPU 前提诊断，不启动实验、不占 GPU、不跑 test；"
                 "判据写死于 OPT-DIRECTIONS-6 §3"),
        "J1_forced_alignment": {
            "n_sentences": len(d["items"]),
            "n_gloss_instances": int(len(sp)),
            "n_align_failed": n_fail,
            "spike_frames_per_gloss": {
                "median": float(np.median(sp)), "mean": round(float(sp.mean()), 3),
                "p10": float(np.percentile(sp, 10)), "p90": float(np.percentile(sp, 90)),
            },
            "span_encoder_steps": {
                "median": round(float(np.median(ss)), 2),
                "mean": round(float(ss.mean()), 2),
                "p10": round(float(np.percentile(ss, 10)), 2),
                "p90": round(float(np.percentile(ss, 90)), 2),
            },
            "span_input_frames": {
                "median": round(med_frames, 2),
                "mean": round(float(sf.mean()), 2),
                "p10": round(float(np.percentile(sf, 10)), 2),
                "p90": round(float(np.percentile(sf, 90)), 2),
                "share_ge_8_frames": round(coverage, 4),
            },
            "step_to_frame_factor": round(STEP_TO_FRAME, 3),
            "rule": "median>=16 输入帧 且 >=8帧占比>=0.80",
            "verdict": "CLIP_BANK_FEASIBLE" if j1 else "STOP",
        },
        "J2_selective": {
            "n_sentences": int(len(conf)),
            "confidence": "mean over frames of max log-prob",
            "rho_conf_vs_sentence_wer": round(rho, 4),
            "base_sentence_wer_pct": round(base_wer * 100, 2),
            "kept_wer_at_70pct_coverage_pct": round(kept_wer * 100, 2),
            "relative_wer_drop_pct": round(drop * 100, 2),
            "rule": "Spearman |rho|>=0.30 (负向) 且 70% coverage 下 WER 相对下降>=25%",
            "verdict": "SELECTIVE_FEASIBLE" if j2 else "STOP",
        },
    }
    (M / (TAG + "-probe.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== J1 强制对齐片段可行性 ===")
    print("  gloss 实例 %d / 对齐失败句 %d" % (len(sp), n_fail))
    print("  spike 帧/词  中位 %.1f 均值 %.3f" % (np.median(sp), sp.mean()))
    print("  区间(编码器步) 中位 %.2f p10 %.2f p90 %.2f"
          % (np.median(ss), np.percentile(ss, 10), np.percentile(ss, 90)))
    print("  区间(输入帧) 中位 %.2f  >=8帧占比 %.4f  => %s"
          % (med_frames, coverage, out["J1_forced_alignment"]["verdict"]))
    print("=== J2 选择性预测可行性 ===")
    print("  rho=%.4f  base WER %.2f%% -> 70%% coverage WER %.2f%% (相对降 %.2f%%) => %s"
          % (rho, base_wer * 100, kept_wer * 100, drop * 100,
             out["J2_selective"]["verdict"]))
    print("收据 ->", TAG + "-probe.json")


if __name__ == "__main__":
    main()
