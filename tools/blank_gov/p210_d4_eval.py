#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P210 / D4 评估：除 WER_official 外，报 uWER 与按 train 频次分桶的 WER。

判据要求「不能只看总 WER」：FGXM(PRL25) 指出 CSLR 高频功能词好、低频实词差，
并提出 uWER（对 gloss 类别等权，而非按出现次数加权）作为无偏指标。

用法：
  venv/bin/python tools/blank_gov/p210_d4_eval.py \
    --ckpt artifacts/checkpoints/p210-d4-cb-best.pt --tag p210-d4
  # 基线同样跑一次 --tag p184-base --ckpt .../p184-tfnet-freeze-best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402
from p207_d1_dump_nbest import build_model, collect_logps  # noqa: E402
from p197_beam_lm import greedy_fast  # noqa: E402

P78.OFFICIAL_PAD = True
P78.RGB_FIX = True
P78.BLANK_BIAS = 0.0


def align_ops(ref, hyp):
    """Levenshtein 回溯，返回每个 ref token 的操作 + 插入数。

    ops[i] ∈ {"match","sub","del"}；ins 为本句插入总数。
    """
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j, ins = [], n, m, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (
                ref[i - 1] != hyp[j - 1]):
            ops.append("match" if ref[i - 1] == hyp[j - 1] else "sub")
            i -= 1
            j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append("del")
            i -= 1
        else:
            ins += 1
            j -= 1
    return list(reversed(ops)), ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--low-freq-max", type=int, default=10,
                    help="train 频次 <= 该值算低频桶（P57 用 10）")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_dv = P78.read_csv_split("dev.csv")
    labels_tr = P78.read_csv_split("train.csv")
    word2idx, word_set_num, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")

    # train 频次：走训练实际路径 DPM.PreWords → word2idx，再映射回评估用的
    # split_glued token 口径。低频桶判据针对「训练里见得少的 gloss」。
    freq = defaultdict(int)
    for _sid, (_p, seq) in labels_tr.items():
        for t in P78.DPM.PreWords(str(seq).split("/")):
            for g in OW.split_glued([t]):
                freq[g] += 1

    _, dv_tf = P78.build_transforms(hflip=False, img_size=a.img_size,
                                    y_shift=0, temporal_rescale=0.0)
    dv_ds = P78.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    loader = DataLoader(dv_ds, batch_size=a.batch, shuffle=False,
                        collate_fn=P78.collate, num_workers=0)
    mdl = build_model(REPO / a.ckpt, 1024, word_set_num, device)
    items = collect_logps(mdl, loader, device, labels_dv)
    print("dev 前向完成: %d 条" % len(items), flush=True)

    refs = [it["ref"] for it in items]
    hyps = [greedy_fast(it["logp"], idx2word) for it in items]
    res = OW.evaluate(refs, hyps)

    # 逐类别聚合（等权 uWER）+ 频次桶
    per_cls = defaultdict(lambda: {"n": 0, "err": 0})
    per_bucket = defaultdict(lambda: {"n": 0, "err": 0})
    for ref, hyp in zip(refs, hyps):
        r = OW.split_glued(OW.split_gloss_sequence(ref, OW.CFG))
        h = OW.split_glued(list(hyp))
        ops, ins = align_ops(r, h)
        assert len(ops) == len(r), "align length mismatch"
        for t, op in zip(r, ops):
            f = freq.get(t, 0)
            b = "low" if f <= a.low_freq_max else "high"
            per_cls[t]["n"] += 1
            per_cls[t]["err"] += (op != "match")
            per_bucket[b]["n"] += 1
            per_bucket[b]["err"] += (op != "match")
        # 插入按桶均摊（uWER 主要看 ref token，插入单独报）
    cls_rates = [v["err"] / v["n"] for v in per_cls.values() if v["n"] > 0]
    uwer = float(np.mean(cls_rates)) * 100.0
    low = per_bucket["low"]
    high = per_bucket["high"]
    low_wer = 100.0 * low["err"] / max(low["n"], 1)
    high_wer = 100.0 * high["err"] / max(high["n"], 1)

    out = {
        "tag": a.tag,
        "ckpt": a.ckpt,
        "n": len(items),
        "WER_official": res["WER_official"],
        "uWER_pct": round(uwer, 2),
        "uWER_n_classes": len(cls_rates),
        "low_freq_max": a.low_freq_max,
        "low_bucket": {"n_tokens": low["n"], "WER_pct": round(low_wer, 2)},
        "high_bucket": {"n_tokens": high["n"], "WER_pct": round(high_wer, 2)},
        "note": ("uWER 对 dev 中出现过的 gloss 类别等权平均（FGXM PRL25 无偏指标）；"
                 "桶 WER 按 train 频次是否 <=10 划分，仅统计 ref token 的 sub+del"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    out_path = REPO / "artifacts/metrics/blank-gov" / ("%s-uwer.json" % a.tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("收据 -> %s" % out_path, flush=True)


if __name__ == "__main__":
    main()
