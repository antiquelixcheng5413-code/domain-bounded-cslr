#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P192：权重插值（P184 + P184b model soup，零训练成本）

两个 ckpt 同架构、同一初始化、同一训练配置，仅"是否冻分类头"不同 ⇒ 权重可平均。
判据：soup dev 需比 P184 单独（34.34，见 p189-beam-protocol-dev.json）低 ≥1.556pp
（噪声地板）才判正向；否则记"无收益"。

用法：
  python tools/blank_gov/p192_soup.py --split dev
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

import torch  # noqa: E402
import p78_train_official_tfnet as P  # noqa: E402
from official_wer import evaluate  # noqa: E402
from p189_beam_eval import greedy, variant_scores  # noqa: E402

CKDIR = REPO / "artifacts/checkpoints"
LOGDIR = REPO / "artifacts/metrics/blank-gov"


def log(*a):
    print("[P192 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--a", default="p184-tfnet-freeze-best")
    ap.add_argument("--b", default="p184b-frozen-heads-best")
    args = ap.parse_args()

    word2idx, word_set_num, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    labels = P.read_csv_split(args.split + ".csv")
    P.OFFICIAL_PAD = True
    P.RGB_FIX = True
    P.BLANK_BIAS = 0.0
    _, tf = P.build_transforms(hflip=False, img_size=160)
    ds = P.RGBSeqDataset(args.split, labels, word2idx, tf, False)

    ck_a = torch.load(str(CKDIR / (args.a + ".pt")), map_location="cpu",
                      weights_only=False)
    ck_b = torch.load(str(CKDIR / (args.b + ".pt")), map_location="cpu",
                      weights_only=False)
    sd_a = ck_a.get("model_state", ck_a)
    sd_b = ck_b.get("model_state", ck_b)
    soup = {k: (sd_a[k].float() + sd_b[k].float()) / 2.0
            for k in sd_a}
    log("soup = 0.5*%s + 0.5*%s（%d 张量）"
        % (args.a, args.b, len(soup)))

    model = P.Net.moduleNet(1024, word_set_num + 1, "TFNet",
                            torch.device("cuda"), "CE-CSL", True).to("cuda")
    model.load_state_dict(soup, strict=True)
    model.eval()
    ls = torch.nn.LogSoftmax(dim=-1)

    refs, hyps = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            vid, _t, _tl, dl, _tl_len, sids, _ids = P.collate([ds[i]])
            o = model(vid.to("cuda"), dl, False)
            hyps += P.greedy_decode(ls(o[0]), dl, idx2word)
            for b in sids:
                refs.append(labels[b][1])
    log("推理 %.0fs" % (time.time() - t0))

    m = variant_scores(refs, hyps)
    m["corpus"] = round(evaluate(refs, hyps)["WER_official"], 2)
    log("soup %s -> %s" % (args.split, json.dumps(m, ensure_ascii=False)))
    dst = LOGDIR / ("p192-soup-%s.json" % args.split)
    dst.write_text(json.dumps({"a": args.a, "b": args.b, "split": args.split,
                               "n": len(refs), "WER": m},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 ->", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
