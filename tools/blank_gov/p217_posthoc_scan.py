#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S5 备用探索：训练期改动（LA / VTA / CSGC）× 解码期补偿（tau / blank bonus）组合扫描。

⚠️ 预注册为【探索性诊断】，明确不做 PASS/FAIL 判定、不进 test：
   目的是看「训练期先验解耦」与「解码期插入抑制」是否互补，为下一轮方向提供信息，
   而不是在 dev 上挑一个最好组合报成绩（避免 p-hacking）。
只使用 dev 515 句；纯推理；不修改任何模型权重。
"""
from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
DESK = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402
import DataProcessMoudle as DPM  # noqa: E402
from p207_d1_dump_nbest import build_model, collect_logps  # noqa: E402
from p197_beam_lm import greedy_fast  # noqa: E402
from p210_d4_eval import align_ops  # noqa: E402

P78.OFFICIAL_PAD = True
P78.RGB_FIX = True
P78.BLANK_BIAS = 0.0
M = REPO / "artifacts/metrics/blank-gov"
CKPTS = [
    ("p184-base", "p184-tfnet-freeze-best.pt"),
    ("p212-e1", "p212-e1-csgc-best.pt"),
    ("p213-f1", "p213-f1-best.pt"),
    ("p214-f2", "p214-f2-best.pt"),
]
TAUS = [0.0, 0.1, 0.2, 0.3]
BETAS = [0.0, 1.0, 2.0]


def main():
    labels_dv = P78.read_csv_split("dev.csv")
    labels_tr = P78.read_csv_split("train.csv")
    word2idx, n, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    freq = defaultdict(int)
    for _sid, (_p, seq) in labels_tr.items():
        for t in DPM.PreWords(str(seq).split("/")):
            freq[t] += 1
    cnt = np.zeros(n + 1, dtype=np.float64)
    for t, k in freq.items():
        g = word2idx.get(t)
        if g is not None and 0 <= g < len(cnt):
            cnt[g] += k
    cnt[1:] = np.maximum(cnt[1:], 1.0)
    logpi = np.zeros(n + 1, dtype=np.float32)
    logpi[1:] = np.log(cnt[1:] / cnt[1:].sum()).astype(np.float32)

    _, tf = P78.build_transforms(hflip=False, img_size=160, y_shift=0,
                                 temporal_rescale=0.0)
    ds = P78.RGBSeqDataset("dev", labels_dv, word2idx, tf, False)
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=P78.collate, num_workers=0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    for tag, fn in CKPTS:
        ck = REPO / "artifacts/checkpoints" / fn
        if not ck.exists():
            print("skip missing", fn, flush=True)
            continue
        mdl = build_model(ck, 1024, n, dev)
        items = collect_logps(mdl, loader, dev, labels_dv)
        del mdl
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
        rs = [OW.split_glued(OW.split_gloss_sequence(it["ref"], OW.CFG))
              for it in items]
        rows = []
        for tau in TAUS:
            for beta in BETAS:
                per_cls = defaultdict(lambda: {"n": 0, "err": 0})
                lo = {"n": 0, "err": 0}
                hi = {"n": 0, "err": 0}
                ins = sub = dele = nref = 0
                for it, r in zip(items, rs):
                    lp = it["logp"].numpy().copy() if hasattr(it["logp"], "numpy") \
                        else np.asarray(it["logp"])
                    lp[:, 1:] -= tau * logpi[1:]
                    if beta:
                        lp[:, 0] += beta
                    h = OW.split_glued(list(greedy_fast(lp, idx2word)))
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
                rows.append({
                    "tau": tau, "beta": beta,
                    "WER_official": round(100.0 * (ins + sub + dele) / max(nref, 1), 2),
                    "uWER_pct": round(float(np.mean([
                        v["err"] / v["n"] for v in per_cls.values() if v["n"] > 0])) * 100, 2),
                    "low_bucket_WER": round(100.0 * lo["err"] / max(lo["n"], 1), 2),
                    "high_bucket_WER": round(100.0 * hi["err"] / max(hi["n"], 1), 2),
                    "ins": int(ins),
                    "micro_subdel_pct": round(100.0 * (sub + dele) / max(nref, 1), 2),
                })
        results[tag] = rows
        b = rows[0]
        best = min((r for r in rows
                    if r["WER_official"] <= b["WER_official"] + 0.30
                    and r["ins"] <= b["ins"] * 1.10),
                   key=lambda r: r["uWER_pct"], default=b)
        print("%s: baseline WER %.2f uWER %.2f low %.2f | guard-best tau=%.1f beta=%.1f "
              "-> WER %.2f uWER %.2f low %.2f ins %d"
              % (tag, b["WER_official"], b["uWER_pct"], b["low_bucket_WER"],
                 best["tau"], best["beta"], best["WER_official"], best["uWER_pct"],
                 best["low_bucket_WER"], best["ins"]), flush=True)

    out = {"tag": "p217-posthoc", "kind": "exploratory_no_verdict",
           "grid": {"tau": TAUS, "beta": BETAS}, "results": results,
           "note": "探索性诊断；不做 PASS/FAIL，不进 test，不用于报性能"}
    (M / "p217-posthoc.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# S5 / P217 训练期改动 × 解码期补偿组合扫描 — 探索性诊断", "",
             "> 生成：%s｜⚠️ **不判 PASS/FAIL、不进 test**｜dev 515 句纯推理" %
             datetime.now().strftime("%Y-%m-%d %H:%M"), "",
             "## 性质", "",
             "预注册为探索性诊断：看训练期先验解耦（F1 LA-CTC / F2 VTA / E1 CSGC）与",
             "解码期补偿（τ·log π 校正 + blank bonus 抑制插入）是否互补。",
             "**不做 PASS/FAIL 判定、不挑最好组合报成绩**（避免 dev 上 p-hacking）。", "",
             "## 各 ckpt 在 τ=0/β=0 与「护栏内 uWER 最低」组合", "",
             "| ckpt | 基线 WER / uWER / 低频桶 | 组合 | WER | uWER | 低频桶 | ins |",
             "|---|---|---|---|---|---|---|"]
    for tag, rows in results.items():
        b = rows[0]
        best = min((r for r in rows
                    if r["WER_official"] <= b["WER_official"] + 0.30
                    and r["ins"] <= b["ins"] * 1.10),
                   key=lambda r: r["uWER_pct"], default=b)
        lines.append("| `%s` | %.2f / %.2f / %.2f | τ=%.1f β=%.1f | %.2f | %.2f | %.2f | %d |"
                     % (tag, b["WER_official"], b["uWER_pct"], b["low_bucket_WER"],
                        best["tau"], best["beta"], best["WER_official"],
                        best["uWER_pct"], best["low_bucket_WER"], best["ins"]))
    lines += ["", "## 结论边界", "",
              "- 所有数字来自同一 dev 集，属描述性观察；低频桶仅 692 token，未做单独配对 bootstrap。",
              "- 若某 ckpt 出现「基线不退化且尾部明显改善」，那也只是下一轮实验的**选题依据**，",
              "  不是已确认的性能提升，需新写预注册判据后重跑 2 seed。",
              "- 未跑 test、未 commit、未 push。", "",
              "数据来源：`artifacts/metrics/blank-gov/p217-posthoc.json`。"]
    txt = "\n".join(lines) + "\n"
    for p in (REPO / "docs/experiments/stage-P217-posthoc.md",
              DESK / "docs/experiments/stage-P217-posthoc.md"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("收据 + 报告已落盘", flush=True)


if __name__ == "__main__":
    main()
