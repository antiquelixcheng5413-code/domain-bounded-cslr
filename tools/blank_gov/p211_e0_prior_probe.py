#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P211 / E0：分类头先验诊断 + post-hoc logit adjustment τ 扫描（纯 CPU）。

两件事：
  A. 前提诊断（一票否决）：
     1) NormLinear 余弦分类器 ⇒ 类权重范数被强制归一，检验 corr(freq, ||w_c||)；
     2) 检验 corr(freq, 该类平均 logp)，复现/推翻 P47b 在 TFNet 上的结论；
     3) 高频标点 `。` 的平均 logp vs 其它非 blank 类。
  B. τ 扫描：对已落盘 dev 后验做 z_c − τ·log π_c（blank 不动），
     greedy 解码后报 WER_official + uWER + 低频桶 + 高频桶 + 配对 bootstrap。

预注册判据（启动前写死）：低频桶 WER 改善 >=2pp 且 uWER 改善 >=1pp，
且 WER_official 退化 <=0.3pp；否则 E0 判负，不做 cRT 重训。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402
import DataProcessMoudle as DPM  # noqa: E402
from p197_beam_lm import greedy_fast  # noqa: E402
from p210_d4_eval import align_ops  # noqa: E402

OUT = REPO / "artifacts/metrics/blank-gov/p211-e0-prior-probe.json"
LOW_MAX = 10


def build_freq(word2idx, n_cls):
    tr = P78.read_csv_split("train.csv")
    c = np.zeros(n_cls, dtype=np.float64)
    for _sid, (_p, seq) in tr.items():
        for t in DPM.PreWords(str(seq).split("/")):
            g = word2idx.get(t)
            if g is not None and 0 <= g < n_cls:
                c[g] += 1
    return c


def evaluate_all(refs, hyps, freq_by_tok):
    res = OW.evaluate(refs, hyps)
    per_cls = defaultdict(lambda: [0, 0])
    bucket = defaultdict(lambda: [0, 0])
    edits = []
    for ref, hyp in zip(refs, hyps):
        r = OW.split_glued(OW.split_gloss_sequence(ref, OW.CFG))
        h = OW.split_glued(list(hyp))
        ops, ins = align_ops(r, h)
        e = sum(o != "match" for o in ops) + ins
        edits.append(e)
        for t, op in zip(r, ops):
            per_cls[t][0] += 1
            per_cls[t][1] += (op != "match")
            b = "low" if freq_by_tok.get(t, 0) <= LOW_MAX else "high"
            bucket[b][0] += 1
            bucket[b][1] += (op != "match")
    uwer = 100.0 * np.mean([v[1] / v[0] for v in per_cls.values() if v[0] > 0])
    return {
        "WER_official": res["WER_official"],
        "uWER": round(float(uwer), 2),
        "low": round(100.0 * bucket["low"][1] / max(bucket["low"][0], 1), 2),
        "high": round(100.0 * bucket["high"][1] / max(bucket["high"][0], 1), 2),
        "micro_subdel": round(100.0 * sum(v[1] for v in bucket.values())
                              / max(sum(v[0] for v in bucket.values()), 1), 2),
        "edits": edits,
    }


def paired_bootstrap(base, cand, n_ref, n_boot=20000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(base, float) - np.asarray(cand, float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].sum(1) / n_ref * 100.0
    return (float(d.sum() / n_ref * 100.0),
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def main():
    word2idx, word_set_num, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    n_cls = word_set_num + 1
    freq = build_freq(word2idx, n_cls)

    # ── A. 前提诊断
    ck = torch.load(REPO / "artifacts/checkpoints/p184-tfnet-freeze-best.pt",
                    map_location="cpu", weights_only=False)
    W = ck["model_state"]["classifier11.weight"].float()   # [1024, 3516]
    wnorm = W.norm(dim=0).numpy()
    diag = {
        "classifier": "NormLinear (F.normalize(weight, dim=0)) → cosine classifier",
        "weight_norm_min": float(wnorm.min()),
        "weight_norm_max": float(wnorm.max()),
        "weight_norm_std": float(wnorm.std()),
        "corr_freq_wnorm": round(float(np.corrcoef(freq[1:], wnorm[1:])[0, 1]), 4),
    }
    post = torch.load(REPO / "artifacts/metrics/blank-gov/p188_logp_p184.pt",
                      map_location="cpu", weights_only=False)
    items = post["items"]
    all_lp = torch.cat([it["logp"] for it in items.values()], dim=0).numpy()
    mean_lp = all_lp.mean(axis=0)
    diag["corr_freq_mean_logp"] = round(
        float(np.corrcoef(np.log1p(freq[1:]), mean_lp[1:])[0, 1]), 4)
    dot_idx = word2idx.get("。")
    if dot_idx is not None:
        diag["punct_logp"] = round(float(mean_lp[dot_idx]), 2)
    nb = mean_lp[1:]
    diag["nonblank_mean_logp"] = round(float(nb.mean()), 2)
    diag["P47b_replicated"] = bool(diag["corr_freq_mean_logp"] < 0.1)
    print("[A] 前提诊断:", json.dumps(diag, ensure_ascii=False), flush=True)

    # ── B. τ 扫描
    labels_dv = P78.read_csv_split("dev.csv")
    # 评估用 token 频次（split_glued 口径）
    ftok = defaultdict(int)
    for _sid, (_p, seq) in P78.read_csv_split("train.csv").items():
        for t in DPM.PreWords(str(seq).split("/")):
            for g in OW.split_glued([t]):
                ftok[g] += 1

    sids = list(items.keys())
    refs = [labels_dv[s][1] for s in sids if s in labels_dv]
    logps = [items[s]["logp"].numpy() for s in sids if s in labels_dv]
    n_ref = sum(len(OW.split_glued(OW.split_gloss_sequence(r, OW.CFG)))
                for r in refs)

    # π: train 频次归一化（按 idx，blank 不参与）
    pi = np.ones(n_cls, dtype=np.float64)
    pi[1:] = np.maximum(freq[1:], 1.0) / max(freq[1:].sum(), 1.0)
    logpi = np.log(pi)

    grid = {}
    base = None
    for tau in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        hyps = []
        for lp in logps:
            adj = lp.copy()
            adj[:, 1:] -= tau * logpi[1:]
            hyps.append(greedy_fast(adj, idx2word))
        m = evaluate_all(refs, hyps, ftok)
        if base is None:
            base = m
            m["delta_pp"] = 0.0
            m["ci95"] = [0.0, 0.0]
        else:
            d, lo, hi = paired_bootstrap(base["edits"], m["edits"], n_ref)
            m["delta_pp"] = round(d, 3)
            m["ci95"] = [round(lo, 3), round(hi, 3)]
        grid[str(tau)] = m
        print("[B] tau=%.1f WER %.2f uWER %.2f low %.2f high %.2f Δ%+.2f CI %s"
              % (tau, m["WER_official"], m["uWER"], m["low"], m["high"],
                 m["delta_pp"], m["ci95"]), flush=True)

    # 预注册判据
    cands = [(float(t), m) for t, m in grid.items() if float(t) > 0]
    passed = [t for t, m in cands
              if base["low"] - m["low"] >= 2.0
              and base["uWER"] - m["uWER"] >= 1.0
              and m["WER_official"] - base["WER_official"] <= 0.3]
    verdict = "PROCEED_CRT" if passed else "FAIL"
    print("\n[判定] 满足判据的 tau:", passed, "=>", verdict, flush=True)

    out = {
        "tag": "p211-e0",
        "n_dev": len(refs),
        "A_prior_diagnosis": diag,
        "B_tau_grid": grid,
        "pre_registered": {
            "rule": "低频桶改善>=2pp 且 uWER改善>=1pp 且 WER_official退化<=0.3pp",
            "note": "主看低频桶；仅总量改善不算正向（D4 教训）",
        },
        "passed_taus": passed,
        "verdict": verdict,
        "caveat": ("τ 扫描直接复用已落盘 dev 后验，属 post-hoc；"
                   "P47b 在 VAC 上已证伪同类做法，此处检验是否在 TFNet 复现"),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("收据 ->", OUT, flush=True)


if __name__ == "__main__":
    main()
