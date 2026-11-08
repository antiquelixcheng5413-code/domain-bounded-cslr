#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""夜间链统一评估器：WER_official + uWER + 低频/高频桶 + ins/sub/del + micro
   + 非 blank 帧占比 + 逐句编辑数（供配对 bootstrap）。

在 p210_d4_eval.py 的口径上补齐 E0/D4 两条教训要求的字段：
  - E0：uWER/低频桶只统计 ref token 的 sub+del，看不见插入 => 必须显式报 ins
  - D4：高频桶恶化 + micro 反而变差 = 类别间重分配，不是净增益
用法：
  venv/bin/python tools/blank_gov/night_eval_full.py \
    --ckpt artifacts/checkpoints/<tag>-best.pt --tag <tag>
输出：artifacts/metrics/blank-gov/<tag>-uwer.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
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


def train_freq(labels_tr):
    """与训练实际 target 路径一致：DPM.PreWords(gloss.split('/'))。"""
    f = defaultdict(int)
    for _sid, (_p, seq) in labels_tr.items():
        for t in DPM.PreWords(str(seq).split("/")):
            f[t] += 1
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--low-freq-max", type=int, default=10)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels_dv = P78.read_csv_split("dev.csv")
    labels_tr = P78.read_csv_split("train.csv")
    word2idx, word_set_num, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    freq = train_freq(labels_tr)

    _, dv_tf = P78.build_transforms(hflip=False, img_size=a.img_size,
                                    y_shift=0, temporal_rescale=0.0)
    ds = P78.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False,
                        collate_fn=P78.collate, num_workers=0)
    mdl = build_model(REPO / a.ckpt, 1024, word_set_num, dev)
    items = collect_logps(mdl, loader, dev, labels_dv)
    print("dev 前向完成: %d 条" % len(items), flush=True)

    refs = [it["ref"] for it in items]
    hyps = [greedy_fast(it["logp"], idx2word) for it in items]
    res = OW.evaluate(refs, hyps)

    per_cls = defaultdict(lambda: {"n": 0, "err": 0})
    per_bucket = defaultdict(lambda: {"n": 0, "err": 0})
    ins = sub = dele = nref = 0
    nb_frames = total_frames = 0
    per_item = []
    for it, ref, hyp in zip(items, refs, hyps):
        r = OW.split_glued(OW.split_gloss_sequence(ref, OW.CFG))
        h = OW.split_glued(list(hyp))
        ops, ii = align_ops(r, h)
        assert len(ops) == len(r), "align length mismatch"
        for t, op in zip(r, ops):
            b = "low" if freq.get(t, 0) <= a.low_freq_max else "high"
            per_cls[t]["n"] += 1
            per_cls[t]["err"] += (op != "match")
            per_bucket[b]["n"] += 1
            per_bucket[b]["err"] += (op != "match")
        s = sum(o == "sub" for o in ops)
        d = sum(o == "del" for o in ops)
        ins += ii
        sub += s
        dele += d
        nref += len(r)
        e = OW.levenshtein(r, h)
        _low_n = sum(1 for t in r if freq.get(t, 0) <= a.low_freq_max)
        _low_err = sum(1 for t, op in zip(r, ops)
                       if freq.get(t, 0) <= a.low_freq_max and op != "match")
        per_item.append({"sid": str(it.get("sid", "")), "ref_n": len(r),
                         "edits": e, "ins": ii, "sub": s, "del": d,
                         "low_n": _low_n, "low_err": _low_err})
        lp = it["logp"]
        path = lp.argmax(-1).reshape(-1).tolist()
        nb_frames += sum(1 for x in path if x != 0)
        total_frames += len(path)

    cls_rates = [v["err"] / v["n"] for v in per_cls.values() if v["n"] > 0]
    low, high = per_bucket["low"], per_bucket["high"]
    out = {
        "tag": a.tag,
        "ckpt": a.ckpt,
        "n": len(items),
        "WER_official": res["WER_official"],
        "uWER_pct": round(float(np.mean(cls_rates)) * 100.0, 2),
        "uWER_n_classes": len(cls_rates),
        "low_freq_max": a.low_freq_max,
        "low_bucket": {"n_tokens": low["n"],
                       "WER_pct": round(100.0 * low["err"] / max(low["n"], 1), 2)},
        "high_bucket": {"n_tokens": high["n"],
                        "WER_pct": round(100.0 * high["err"] / max(high["n"], 1), 2)},
        "edits": {"ins": int(ins), "sub": int(sub), "del": int(dele),
                  "n_ref_tokens": int(nref),
                  "micro_subdel_pct": round(100.0 * (sub + dele) / max(nref, 1), 2),
                  "ins_rate_pct": round(100.0 * ins / max(nref, 1), 2)},
        "nonblank_frame_ratio_pct": round(100.0 * nb_frames / max(total_frames, 1), 2),
        "per_item": per_item,
        "note": ("uWER 对 dev 出现过的 gloss 类别等权；桶按 train 频次<=10 划分，"
                 "只统计 ref token 的 sub+del；插入数必须单列（E0 教训）"),
    }
    p = REPO / "artifacts/metrics/blank-gov" / ("%s-uwer.json" % a.tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    slim = {k: v for k, v in out.items() if k != "per_item"}
    print(json.dumps(slim, ensure_ascii=False, indent=2), flush=True)
    print("收据 -> %s" % p, flush=True)


if __name__ == "__main__":
    main()
