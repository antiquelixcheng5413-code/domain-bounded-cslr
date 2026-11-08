#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P207 / D1 阶段一：落盘 dev 的 8-best CTC 候选，供 LM 重排反复使用（零训练）。

审计要求：脚本必须先复现已知答案（greedy 34.41 / beam 1-best 34.34 /
8-best oracle 25.44）才允许落盘，否则说明管线未对齐，直接退出。
用法：
  venv/bin/python tools/blank_gov/p207_d1_dump_nbest.py --out artifacts/cache/p207_d1_nbest.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
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
from p197_beam_lm import (  # noqa: E402
    ctc_prefix_beam, greedy_fast, build_model, collect_logps,
)

P78.OFFICIAL_PAD = True
P78.RGB_FIX = True
P78.BLANK_BIAS = 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/p184-tfnet-freeze-best.pt")
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--nbest", type=int, default=8)
    ap.add_argument("--out", default="artifacts/cache/p207_d1_nbest.pkl")
    ap.add_argument("--max-dev", type=int, default=0, help=">0 时冒烟用")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_dv = P78.read_csv_split("dev.csv")
    word2idx, word_set_num, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")

    _, dv_tf = P78.build_transforms(hflip=False, img_size=a.img_size,
                                    y_shift=0, temporal_rescale=0.0)
    dv_ds = P78.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    if a.max_dev > 0:
        dv_ds.items = dv_ds.items[:a.max_dev]
    loader = DataLoader(dv_ds, batch_size=a.batch, shuffle=False,
                        collate_fn=P78.collate, num_workers=0)
    print("dev items = %d, vocab = %d" % (len(dv_ds), word_set_num + 1), flush=True)

    mdl = build_model(REPO / a.ckpt, 1024, word_set_num, device)
    items = collect_logps(mdl, loader, device, labels_dv)
    print("前向完成：%d 条" % len(items), flush=True)
    del mdl
    if device == "cuda":
        torch.cuda.empty_cache()

    refs = [it["ref"] for it in items]
    g_hyps = [greedy_fast(it["logp"], idx2word) for it in items]
    g_res = OW.evaluate(refs, g_hyps)
    print("[greedy]       WER_official %.2f%%" % g_res["WER_official"], flush=True)

    # ── 生成 N-best
    nb_list = []
    for i, it in enumerate(items):
        nb = ctc_prefix_beam(it["logp"], a.beam, a.topk, a.nbest)
        nb_list.append(nb)
        if (i + 1) % 100 == 0:
            print("  beam %d/%d" % (i + 1, len(items)), flush=True)

    ref_tok = [OW.split_glued(OW.split_gloss_sequence(g, OW.CFG)) for g in refs]
    oracle_e, oracle_n = 0, 0
    for rt, nb in zip(ref_tok, nb_list):
        best = min(OW.levenshtein(rt, OW.split_glued([idx2word[c] for c in pre]))
                   for pre, _ in nb)
        oracle_e += best
        oracle_n += len(rt)
    oracle_wer = 100.0 * oracle_e / max(1, oracle_n)
    print("[beam oracle]  WER_official %.2f%% (N=%d)" % (oracle_wer, a.nbest), flush=True)

    b1 = [[idx2word[c] for c in nb[0][0]] for nb in nb_list]
    b1_res = OW.evaluate(refs, b1)
    print("[beam 1-best]  WER_official %.2f%%" % b1_res["WER_official"], flush=True)

    # ── 管线自检：冒烟模式只验证能跑，不判方向；全量必须复现已知答案
    if a.max_dev > 0:
        print("[smoke] 管线可运行，跳过数值校验", flush=True)
    else:
        ok = (abs(g_res["WER_official"] - 34.41) < 0.05
              and abs(b1_res["WER_official"] - 34.34) < 0.05
              and abs(oracle_wer - 25.44) < 0.05)
        print("[self-check] greedy %.2f / beam1 %.2f / oracle %.2f => %s"
              % (g_res["WER_official"], b1_res["WER_official"], oracle_wer,
                 "PASS" if ok else "FAIL"), flush=True)
        if not ok:
            print("管线未复现 P197 已知答案，拒绝落盘", flush=True)
            sys.exit(2)

    out = {
        "ckpt": a.ckpt,
        "beam": a.beam, "topk": a.topk, "nbest": a.nbest,
        "n": len(items),
        "greedy_wer": g_res["WER_official"],
        "beam1_wer": b1_res["WER_official"],
        "oracle_wer": round(oracle_wer, 2),
        "items": [
            {
                "sid": it["sid"],
                "ref": it["ref"],
                "greedy": [str(t) for t in g_hyps[i]],
                "nbest": [
                    {"tokens": [idx2word[c] for c in pre], "lp_ctc": float(lp)}
                    for pre, lp in nb_list[i]
                ],
            }
            for i, it in enumerate(items)
        ],
    }
    out_path = REPO / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print("落盘 -> %s (%.1f MB)" % (out_path, out_path.stat().st_size / 1e6),
          flush=True)


if __name__ == "__main__":
    main()
