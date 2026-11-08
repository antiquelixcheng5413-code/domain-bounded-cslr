#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F0 / P215：插入率感知解码扫描（零训练，纯 CPU）。

E0(P211) 发现 post-hoc logit adjustment 能把 uWER/低频桶拉下来，但插入爆炸
（3.17% -> 7.92%）导致总量崩到 45.21。本项检验给 blank 加 bonus 抑制插入后，
尾部收益能否变成净收益。

预注册（启动前写死）：
  网格 tau x beta，在【满足护栏 R3/R4/R5/R6】的组合中取 uWER 最小者作为唯一代表；
  若无组合满足护栏，代表 = 基线 (tau=0,beta=0)，自动判负。
  代表组合写 <tag>-uwer.json 交统一判定器，全网格另存 <tag>-scan.json。
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

M = REPO / "artifacts/metrics/blank-gov"
TAG = "p215-f0"
TAUS = [0.0, 0.1, 0.2, 0.3, 0.5]
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0]


def train_freq(labels_tr):
    f = defaultdict(int)
    for _sid, (_p, seq) in labels_tr.items():
        for t in DPM.PreWords(str(seq).split("/")):
            f[t] += 1
    return f


def main():
    d = torch.load(REPO / "artifacts/metrics/blank-gov/p188_logp_p184.pt",
                   map_location="cpu", weights_only=False)
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
    refs = []
    for s in sids:
        refs.append(OW.split_glued(OW.split_gloss_sequence(labels_dv[s][1], OW.CFG)))
    print("items=%d  logpi[1:] min=%.3f max=%.3f" %
          (len(sids), float(logpi[1:].min()), float(logpi[1:].max())), flush=True)

    def evaluate(tau, beta):
        per_cls = defaultdict(lambda: {"n": 0, "err": 0})
        per_bucket = defaultdict(lambda: {"n": 0, "err": 0})
        ins = sub = dele = nref = 0
        per_item = []
        hyps_all = []
        for s, r in zip(sids, refs):
            lp = d["items"][s]["logp"].numpy().copy()
            lp[:, 1:] -= tau * logpi[1:]
            if beta:
                lp[:, 0] += beta
            h = OW.split_glued(list(greedy_fast(lp, idx2word)))
            hyps_all.append(list(h))
            ops, ii = align_ops(r, h)
            for t, op in zip(r, ops):
                b = "low" if freq.get(t, 0) <= 10 else "high"
                per_cls[t]["n"] += 1
                per_cls[t]["err"] += (op != "match")
                per_bucket[b]["n"] += 1
                per_bucket[b]["err"] += (op != "match")
            s_ = sum(o == "sub" for o in ops)
            d_ = sum(o == "del" for o in ops)
            ins += ii; sub += s_; dele += d_; nref += len(r)
            per_item.append({"sid": str(s), "ref_n": len(r),
                             "edits": OW.levenshtein(r, h), "ins": ii,
                             "sub": s_, "del": d_})
        low, high = per_bucket["low"], per_bucket["high"]
        cls_rates = [v["err"] / v["n"] for v in per_cls.values() if v["n"] > 0]
        out = {
            "WER_official": round(100.0 * sum(x["edits"] for x in per_item)
                                  / max(nref, 1), 2),
            "uWER_pct": round(float(np.mean(cls_rates)) * 100.0, 2),
            "low_bucket_WER": round(100.0 * low["err"] / max(low["n"], 1), 2),
            "high_bucket_WER": round(100.0 * high["err"] / max(high["n"], 1), 2),
            "ins": int(ins), "sub": int(sub), "del": int(dele),
            "micro_subdel_pct": round(100.0 * (sub + dele) / max(nref, 1), 2),
            "ins_rate_pct": round(100.0 * ins / max(nref, 1), 2),
            "per_item": per_item, "hyps": hyps_all,
            "low_n": low["n"], "high_n": high["n"], "n_ref": nref,
        }
        return out

    grid = {}
    for tau in TAUS:
        for beta in BETAS:
            r = evaluate(tau, beta)
            grid["%.2f/%.2f" % (tau, beta)] = {
                k: v for k, v in r.items() if k not in ("per_item", "hyps")}
            print("tau=%.2f beta=%.2f  WER %.2f  uWER %.2f  low %.2f  high %.2f  ins %d"
                  % (tau, beta, r["WER_official"], r["uWER_pct"],
                     r["low_bucket_WER"], r["high_bucket_WER"], r["ins"]), flush=True)

    base = grid["0.00/0.00"]
    # 预注册选择规则：先过护栏，再取 uWER 最小
    guard = []
    for key, r in grid.items():
        if (r["high_bucket_WER"] - base["high_bucket_WER"] <= 1.0
                and r["WER_official"] - base["WER_official"] <= 0.30
                and (r["ins"] - base["ins"]) / max(base["ins"], 1) <= 0.10
                and r["micro_subdel_pct"] - base["micro_subdel_pct"] <= 0.30):
            guard.append((r["uWER_pct"], key))
    chosen = min(guard)[1] if guard else "0.00/0.00"
    print("chosen (pre-registered rule) =", chosen,
          " guard-passing combos =", len(guard), flush=True)
    tau, beta = [float(x) for x in chosen.split("/")]
    rep = evaluate(tau, beta)

    def as_uwer(r):
        return {
            "tag": TAG, "ckpt": "p188_logp_p184.pt (post-hoc)", "n": len(sids),
            "WER_official": r["WER_official"], "uWER_pct": r["uWER_pct"],
            "uWER_n_classes": len(set(t for rr in refs for t in rr)),
            "low_freq_max": 10,
            "low_bucket": {"n_tokens": r["low_n"], "WER_pct": r["low_bucket_WER"]},
            "high_bucket": {"n_tokens": r["high_n"], "WER_pct": r["high_bucket_WER"]},
            "edits": {"ins": r["ins"], "sub": r["sub"], "del": r["del"],
                      "n_ref_tokens": r["n_ref"],
                      "micro_subdel_pct": r["micro_subdel_pct"],
                      "ins_rate_pct": r["ins_rate_pct"]},
            "nonblank_frame_ratio_pct": None,
            "per_item": r["per_item"],
            "note": "F0 解码扫描代表组合 tau=%.2f beta=%.2f（预注册选择规则）"
                    % (tau, beta),
        }

    (M / (TAG + "-uwer.json")).write_text(
        json.dumps(as_uwer(rep), ensure_ascii=False, indent=2), encoding="utf-8")
    (M / (TAG + "-scan.json")).write_text(
        json.dumps({"grid": grid, "chosen": chosen, "guard_count": len(guard),
                    "selection_rule": "guard R3/R4/R5/R6 first, then min uWER; "
                                      "if none pass -> baseline (auto FAIL)"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("收据 -> %s / %s-scan.json" % (M / (TAG + "-uwer.json"), TAG), flush=True)


if __name__ == "__main__":
    main()
