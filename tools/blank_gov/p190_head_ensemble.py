#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P190：模型内部多头集成（一次前向，零训练成本）

TFNet 前向返回 5 个分类头（Net.py:474-490）：
    logProbs1 = classifier11(temporal_model 时间支路)          ← 推理时被替换
    logProbs2 = classifier22(x)        时间支路*未过时序模型*
    logProbs3 = classifier33(temporal_model1 频域支路)
    logProbs4 = classifier44(x1)       频域支路*未过时序模型*
    logProbs5 = classifier55(时间+频域注意力相加)  ← 推理时 logProbs1 = logProbs5

所以推理可用的**互异**头有 4 个：o[0](=o[4] 融合)、o[1]、o[2]、o[3]。
本脚本一次前向落盘 4 个头，再逐组合做 log 概率平均 + 贪心解码。

用法：
  python tools/blank_gov/p190_head_ensemble.py --ckpt <pt> --img 224 --tag official224
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import p78_train_official_tfnet as P  # noqa: E402
from official_wer import evaluate  # noqa: E402
from p189_beam_eval import greedy, variant_scores  # noqa: E402

LOGDIR = REPO / "artifacts/metrics/blank-gov"

COMBOS = {
    "h0": [0],
    "h1": [1],
    "h2": [2],
    "h3": [3],
    "h0+h1": [0, 1],
    "h0+h2": [0, 2],
    "h0+h1+h3": [0, 1, 3],
    "all4": [0, 1, 2, 3],
}


def log(*a):
    print("[P190 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def load_sd(path):
    ck = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    return ck.get("moduleNet_state_dict") or ck["model_state"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    word2idx, word_set_num, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    labels = P.read_csv_split(args.split + ".csv")
    P.OFFICIAL_PAD = True
    P.RGB_FIX = True
    P.BLANK_BIAS = 0.0
    _, tf = P.build_transforms(hflip=False, img_size=args.img)
    ds = P.RGBSeqDataset(args.split, labels, word2idx, tf, False)

    model = P.Net.moduleNet(1024, word_set_num + 1, "TFNet",
                            torch.device("cuda"), "CE-CSL", True).to("cuda")
    model.load_state_dict(load_sd(args.ckpt), strict=True)
    model.eval()
    ls = torch.nn.LogSoftmax(dim=-1)

    out, t0 = {}, time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            vid, _t, _tl, dl, _tl_len, sids, _ids = P.collate([ds[i]])
            o = model(vid.to("cuda"), dl, False)
            heads = [ls(o[j]).squeeze(1).half().cpu() for j in range(4)]
            for b in sids:
                out[str(b)] = {"heads": heads, "dl": int(dl[0])}
    dst = LOGDIR / ("p190_heads_%s.pt" % args.tag)
    torch.save({"img": args.img, "ckpt": str(args.ckpt), "items": out}, dst)
    log("dump %s img=%d n=%d (%.0fs)" % (args.tag, args.img, len(out),
                                        time.time() - t0))

    order = sorted(out)
    refs = [labels[sid][1] for sid in order]
    # 顺带检查 o[0] 是否真等于 o[4]（推理时 logProbs1 = logProbs5）
    rows = {}
    for name, hs in COMBOS.items():
        hyps = []
        for sid in order:
            rec = out[sid]
            t = P._conv_len(rec["dl"])
            logp = torch.stack([rec["heads"][j].float() for j in hs]).mean(0)
            hyps.append(greedy(logp, t, idx2word))
        m = variant_scores(refs, hyps)
        m["corpus"] = round(evaluate(refs, hyps)["WER_official"], 2)
        rows[name] = m
        log("%-10s %s" % (name, json.dumps(m, ensure_ascii=False)))

    dst2 = LOGDIR / ("p190-head-ensemble-%s.json" % args.split)
    dst2.write_text(json.dumps({"tag": args.tag, "img": args.img,
                                "n": len(refs), "rows": rows},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 ->", dst2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
