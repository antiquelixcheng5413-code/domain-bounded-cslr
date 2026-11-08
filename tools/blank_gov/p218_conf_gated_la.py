#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P218：置信度门控的类先验校正（零训练，纯 CPU）。

动机（来自今晚 F0 的实测）：全局 τ·log π 校正 + blank bonus 在 τ=0.3/β=2.0 时
低频桶 +2.60pp、uWER +1.67pp、总量 +0.10pp、插入不增，但 uWER 差 0.33pp 未过 R1 线。
F0 的高频桶仍恶化 0.74pp，说明全局校正误伤了模型已经很确信的高频功能词。

本项假设：只在「模型不确信的非 blank 帧」上抬升低频类，可减少对确信帧的误伤。
依据：【间接】Menon et al. ICLR 2021 logit adjustment（类先验校正）；
      【有】本地 F0(P215) 实测结果；
      【无】置信度门控这一具体做法无直接论文支撑，属工程假设，明确标注。

预注册（启动前写死，08:10）：
  网格 τ × β × θ；代表组合 = 满足护栏 R3/R4/R5/R6 的组合中 uWER 最小者；
  若无组合过护栏，代表 = 基线（自动 FAIL）。代表写入 p218-uwer.json 交统一判定器
  night_judge.py 按 R1–R6 判定。网格全表另存 p218-scan.json。
  ⚠️ 在 dev 上选参有过拟合风险：PASS 只代表值得 2 seed 复跑，不是已确认提升。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
DESK = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402
import DataProcessMoudle as DPM  # noqa: E402
from p197_beam_lm import greedy_fast  # noqa: E402
from p210_d4_eval import align_ops  # noqa: E402

M = REPO / "artifacts/metrics/blank-gov"
TAG = "p218"
TAUS = [0.2, 0.3, 0.5]
BETAS = [1.0, 2.0]
THETAS = [0.50, 0.80, 0.95]


def train_freq(labels_tr):
    f = defaultdict(int)
    for _sid, (_p, seq) in labels_tr.items():
        for t in DPM.PreWords(str(seq).split("/")):
            f[t] += 1
    return f


def main():
    d = torch.load(M / "p188_logp_p184.pt", map_location="cpu", weights_only=False)
    labels_dv = P78.read_csv_split("dev.csv")
    labels_tr = P78.read_csv_split("train.csv")
    word2idx, n, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    freq = train_freq(labels_tr)

    cnt = np.zeros(n + 1, dtype=np.float64)
    for t, k in freq.items():
        g = word2idx.get(t)
        if g is not None and 0 <= g < len(cnt):
            cnt[g] += k
    cnt[1:] = np.maximum(cnt[1:], 1.0)
    logpi = np.zeros(n + 1, dtype=np.float32)
    logpi[1:] = np.log(cnt[1:] / cnt[1:].sum()).astype(np.float32)

    sids = [s for s in d["items"] if s in labels_dv]
    refs = [OW.split_glued(OW.split_gloss_sequence(labels_dv[s][1], OW.CFG))
            for s in sids]
    print("items=%d" % len(sids), flush=True)

    def evaluate(tau, beta, theta):
        per_cls = defaultdict(lambda: {"n": 0, "err": 0})
        lo = {"n": 0, "err": 0}
        hi = {"n": 0, "err": 0}
        ins = sub = dele = nref = 0
        gated_frames = total_frames = 0
        per_item, hyps_all = [], []
        for s, r in zip(sids, refs):
            lp = d["items"][s]["logp"].numpy().copy()
            if tau > 0:
                # lp 已是 log-prob：argmax 与 max 可直接在 lp 上算，
                # 置信度只需每帧一次 exp，避免对 [T,3516] 全矩阵做 exp（快 ~100x）
                arg = lp.argmax(1)
                conf = np.exp(lp.max(1))
                gate = ((arg != 0) & (conf < theta)).astype(np.float32)
                gated_frames += int(gate.sum())
                total_frames += len(gate)
                lp[:, 1:] -= (tau * gate[:, None]) * logpi[1:]
            if beta:
                lp[:, 0] += beta
            h = OW.split_glued(list(greedy_fast(lp, idx2word)))
            hyps_all.append(list(h))
            ops, ii = align_ops(r, h)
            for t, op in zip(r, ops):
                bkt = lo if freq.get(t, 0) <= 10 else hi
                per_cls[t]["n"] += 1
                per_cls[t]["err"] += (op != "match")
                bkt["n"] += 1
                bkt["err"] += (op != "match")
            s_ = sum(o == "sub" for o in ops)
            d_ = sum(o == "del" for o in ops)
            ins += ii; sub += s_; dele += d_; nref += len(r)
            per_item.append({"sid": str(s), "ref_n": len(r),
                             "edits": OW.levenshtein(r, h), "ins": ii,
                             "sub": s_, "del": d_})
        cls_rates = [v["err"] / v["n"] for v in per_cls.values() if v["n"] > 0]
        return {
            "WER_official": round(100.0 * sum(x["edits"] for x in per_item) / max(nref, 1), 2),
            "uWER_pct": round(float(np.mean(cls_rates)) * 100.0, 2),
            "low_bucket_WER": round(100.0 * lo["err"] / max(lo["n"], 1), 2),
            "high_bucket_WER": round(100.0 * hi["err"] / max(hi["n"], 1), 2),
            "ins": int(ins), "sub": int(sub), "del": int(dele),
            "micro_subdel_pct": round(100.0 * (sub + dele) / max(nref, 1), 2),
            "gated_frame_pct": round(100.0 * gated_frames / max(total_frames, 1), 2),
            "per_item": per_item, "hyps": hyps_all,
            "low_n": lo["n"], "high_n": hi["n"], "n_ref": nref,
        }

    grid = {}
    base = evaluate(0.0, 0.0, 1.0)
    grid["0.00/0.00/1.00"] = {k: v for k, v in base.items()
                              if k not in ("per_item", "hyps")}
    for tau in TAUS:
        for beta in BETAS:
            for theta in THETAS:
                r = evaluate(tau, beta, theta)
                key = "%.2f/%.2f/%.2f" % (tau, beta, theta)
                grid[key] = {k: v for k, v in r.items()
                             if k not in ("per_item", "hyps")}
                print("%s WER %.2f uWER %.2f low %.2f high %.2f ins %d gate %.1f%%"
                      % (key, r["WER_official"], r["uWER_pct"],
                         r["low_bucket_WER"], r["high_bucket_WER"],
                         r["ins"], r["gated_frame_pct"]), flush=True)

    guard = []
    for k, r in grid.items():
        if (r["high_bucket_WER"] - base["high_bucket_WER"] <= 1.0
                and r["WER_official"] - base["WER_official"] <= 0.30
                and (r["ins"] - base["ins"]) / max(base["ins"], 1) <= 0.10
                and r["micro_subdel_pct"] - base["micro_subdel_pct"] <= 0.30):
            guard.append((r["uWER_pct"], k))
    chosen = min(guard)[1] if guard else "0.00/0.00/1.00"
    tau, beta, theta = [float(x) for x in chosen.split("/")]
    rep = evaluate(tau, beta, theta)
    print("chosen =", chosen, " guard-passing =", len(guard), flush=True)

    uwer = {
        "tag": TAG, "ckpt": "p188_logp_p184.pt (confidence-gated post-hoc)",
        "n": len(sids), "WER_official": rep["WER_official"],
        "uWER_pct": rep["uWER_pct"],
        "uWER_n_classes": len(set(t for r in refs for t in r)),
        "low_freq_max": 10,
        "low_bucket": {"n_tokens": rep["low_n"], "WER_pct": rep["low_bucket_WER"]},
        "high_bucket": {"n_tokens": rep["high_n"], "WER_pct": rep["high_bucket_WER"]},
        "edits": {"ins": rep["ins"], "sub": rep["sub"], "del": rep["del"],
                  "n_ref_tokens": rep["n_ref"],
                  "micro_subdel_pct": rep["micro_subdel_pct"],
                  "ins_rate_pct": round(100.0 * rep["ins"] / max(rep["n_ref"], 1), 2)},
        "nonblank_frame_ratio_pct": rep["gated_frame_pct"],
        "per_item": rep["per_item"],
        "note": "P218 代表组合 tau=%.2f beta=%.2f theta=%.2f（预注册选择规则）"
                % (tau, beta, theta),
    }
    (M / (TAG + "-uwer.json")).write_text(
        json.dumps(uwer, ensure_ascii=False, indent=2), encoding="utf-8")
    (M / (TAG + "-scan.json")).write_text(json.dumps(
        {"grid": grid, "chosen": chosen, "guard_count": len(guard),
         "selection_rule": "guard R3/R4/R5/R6 first, then min uWER; "
                           "if none pass -> baseline (auto FAIL)",
         "taus": TAUS, "betas": BETAS, "thetas": THETAS},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # 附带诊断：替换对是否语义相近（用 F2 生成的 gloss 文本嵌入）
    sem = {"available": False}
    emb_p = REPO / "artifacts/cache/p214_vta_target.npy"
    if emb_p.exists():
        E = np.load(emb_p)
        pairs, rng = [], np.random.default_rng(42)
        for r, h in zip(refs, rep["hyps"]):
            ops, _ = align_ops(r, h)
            # 简化：逐位置对齐不严格，只看「同一编辑串里出现的 ref/hyp 差异」
            for i, (t, op) in enumerate(zip(r, ops)):
                if op == "sub" and i < len(h):
                    a, b = word2idx.get(t), word2idx.get(h[i])
                    if a and b and 0 < a < len(E) and 0 < b < len(E):
                        pairs.append((int(a), int(b)))
        if pairs:
            cs = [float(E[a] @ E[b]) for a, b in pairs]
            ri = rng.integers(1, len(E), size=(len(pairs), 2))
            rc = [float(E[a] @ E[b]) for a, b in ri]
            sem = {"available": True, "n_sub_pairs": len(pairs),
                   "mean_cos_error_pairs": round(float(np.mean(cs)), 4),
                   "mean_cos_random_pairs": round(float(np.mean(rc)), 4),
                   "lift": round(float(np.mean(cs)) / max(abs(float(np.mean(rc))), 1e-6), 2)}
    (M / (TAG + "-semantic.json")).write_text(
        json.dumps(sem, ensure_ascii=False, indent=2), encoding="utf-8")
    print("semantic diag:", sem, flush=True)

    txt = """# P218 置信度门控类先验校正 — 阶段报告（零训练诊断）

> 生成：%s｜dev 515 句纯 CPU 推理｜统一判定见 `p218-judge.json`

## 1. 动机与依据

- 【间接】Menon et al., ICLR 2021 logit adjustment：训练/解码时按类先验校正可改善长尾。
- 【有】本地 F0（P215）：τ=0.3/β=2.0 得到低频桶 **+2.60pp**、uWER +1.67pp、
  总量 +0.10pp、插入不增，但 uWER 差 **0.33pp** 未过 R1 线；高频桶仍恶化 0.74pp
  ⇒ 全局校正误伤了模型已确信的高频功能词。
- 【无】「只在低置信度非 blank 帧施加校正」这一门控做法无直接论文支撑，属工程假设。

## 2. 方法（预注册）

对 dev 后验每帧：若 `argmax != blank` 且 `max p < θ`，则该帧对非 blank 类施加
`−τ·log π_c`（blank 类不动）；再全局加 `β` 的 blank bonus 抑制插入。
网格 τ∈%s × β∈%s × θ∈%s + 基线。
**代表组合选择规则**：满足护栏 R3/R4/R5/R6 的组合中取 uWER 最低；无则取基线（自动 FAIL）。
交 `night_judge.py` 按统一 R1–R6 判定。

## 3. 结果

| | WER | uWER | 低频桶 | 高频桶 | ins | micro | 门控帧占比 |
|---|---|---|---|---|---|---|---|
| 基线 | %.2f | %.2f | %.2f | %.2f | %d | %.2f | — |
| 代表 τ=%.2f/β=%.2f/θ=%.2f | %.2f | %.2f | %.2f | %.2f | %d | %.2f | %.1f%% |

过护栏组合数 = %d；完整网格见 `p218-scan.json`。

**替换对语义相似度诊断**（用 F2 的 Qwen gloss 嵌入）：%s

## 4. 诚实边界

- 在 dev 上做网格选参**有过拟合风险**：护栏优先 + 预注册选择规则只能降低、不能消除该风险。
  即便 PASS，也只代表值得投 2 seed + 配对 bootstrap 复跑，不是已确认的性能提升，更不进 test。
- 低频桶仅 692 token，未做单独配对 bootstrap；不能套 ±0.77pp 总量噪声地板。
- 门控阈值 θ 是工程假设，无文献支撑，结论不可外推到其它数据集/骨干。
- 语义诊断的 sub 对按位置近似对齐（非严格 Levenshtein 回溯），只作定性参考。

## 5. 状态

未跑 test、未 commit、未 push。
""" % (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        TAUS, BETAS, THETAS,
        base["WER_official"], base["uWER_pct"], base["low_bucket_WER"],
        base["high_bucket_WER"], base["ins"], base["micro_subdel_pct"],
        tau, beta, theta,
        rep["WER_official"], rep["uWER_pct"], rep["low_bucket_WER"],
        rep["high_bucket_WER"], rep["ins"], rep["micro_subdel_pct"],
        rep["gated_frame_pct"], len(guard),
        ("错误对 cos=%.4f vs 随机对 cos=%.4f（lift %.2fx，n=%d）"
         % (sem["mean_cos_error_pairs"], sem["mean_cos_random_pairs"],
            sem["lift"], sem["n_sub_pairs"])) if sem.get("available")
        else "嵌入不可用，跳过",
    )
    for p in (REPO / "docs/experiments/stage-P218-gated-la.md",
              DESK / "docs/experiments/stage-P218-gated-la.md"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> stage-P218-gated-la.md (+桌面镜像)", flush=True)


if __name__ == "__main__":
    main()
