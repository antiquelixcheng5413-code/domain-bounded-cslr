#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P188：多模型 logit 集成（dev 验证）

三种成员输出时间步均由 dataLen 决定，与输入分辨率无关 ⇒ log 概率可直接对齐平均。

两步走，避免 224 推理时的显存碎片 OOM：
  1) dump：每个成员单独一个进程，把 log 概率落盘
       python tools/blank_gov/p188_ensemble.py dump --ckpt <pt> --img 160 --tag p184
  2) combine：读取若干 dump，加权平均后 greedy 解码算官方 WER
       python tools/blank_gov/p188_ensemble.py combine \
           --inputs p184,p184b,official --weights 1,1,1

权重文件支持两种格式：官方 moduleNet_state_dict / p78 model_state。
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

LOGDIR = REPO / "artifacts/metrics/blank-gov"


def log(*a):
    print("[P188 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def build_model(word_set_num):
    return P.Net.moduleNet(1024, word_set_num + 1, "TFNet",
                           torch.device("cuda"), "CE-CSL", True).to("cuda")


def load_sd(path):
    ck = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    return ck.get("moduleNet_state_dict") or ck["model_state"]


def dump(args):
    word2idx, word_set_num, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    labels = P.read_csv_split("dev.csv")
    P.OFFICIAL_PAD = True
    P.RGB_FIX = True
    _, tf = P.build_transforms(hflip=False, img_size=args.img)
    ds = P.RGBSeqDataset("dev", labels, word2idx, tf, False)
    if args.max_dev:
        ds.items = ds.items[:args.max_dev]

    model = build_model(word_set_num)
    model.load_state_dict(load_sd(args.ckpt), strict=True)
    model.eval()
    ls = torch.nn.LogSoftmax(dim=-1)

    out = {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            vid, _t, _tl, dl, _tl_len, sids, _ids = P.collate([ds[i]])
            o = model(vid.to("cuda"), dl, False)
            logp = ls(o[0]).squeeze(1).half().cpu()   # [T,C]
            for b in sids:
                out[str(b)] = {"logp": logp, "dl": int(dl[0])}
    dst = LOGDIR / ("p188_logp_%s.pt" % args.tag)
    torch.save({"img": args.img, "ckpt": str(args.ckpt), "items": out}, dst)
    log("dump %s  img=%d  n=%d  (%.0fs) -> %s"
        % (args.tag, args.img, len(out), time.time() - t0, dst.name))
    return 0


def combine(args):
    word2idx, word_set_num, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    labels = P.read_csv_split("dev.csv")
    tags = [x.strip() for x in args.inputs.split(",") if x.strip()]
    weights = [float(x) for x in args.weights.split(",")] if args.weights \
        else [1.0] * len(tags)
    assert len(weights) == len(tags), "weights 与 inputs 数量不一致"

    acc = {}
    for tag, w in zip(tags, weights):
        blob = torch.load(str(LOGDIR / ("p188_logp_%s.pt" % tag)),
                          map_location="cpu", weights_only=False)
        log("成员 %s  img=%d  weight=%.3f  n=%d"
            % (tag, blob["img"], w, len(blob["items"])))
        for b, rec in blob["items"].items():
            cur = acc.get(b)
            if cur is None:
                acc[b] = {"dl": rec["dl"], "logp": rec["logp"].float() * w}
            else:
                assert cur["dl"] == rec["dl"], "sid=%d 的时间步不一致" % b
                cur["logp"] += rec["logp"].float() * w

    refs, hyps = [], []
    P.BLANK_BIAS = 0.0
    with torch.no_grad():
        for b in sorted(acc):
            rec = acc[b]
            logp = rec["logp"].unsqueeze(1)              # [T,1,C]
            dl = torch.tensor([rec["dl"]])
            hyp = P.greedy_decode(logp, dl, idx2word)[0]
            hyps.append(hyp)
            refs.append(labels[b][1])

    o = evaluate(refs, hyps)
    log("集成 dev WER = %.2f%%  (n=%d)" % (o["WER_official"], len(refs)))
    dst = LOGDIR / "p188-ensemble-dev.json"
    dst.write_text(json.dumps({
        "tags": tags, "weights": weights, "n": len(refs),
        "WER_official": o["WER_official"], "metrics": o,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % dst)
    return 0


def _decode_all(items, weights, idx2word):
    """按给定权重合成 logp 并贪婪解码；weights 为 {tag: w}。"""
    P.BLANK_BIAS = 0.0
    hyps = {}
    with torch.no_grad():
        for sid in sorted(items["_order"]):
            logp = None
            dl = None
            for tag, w in weights.items():
                rec = items[tag][sid]
                dl = rec["dl"]
                cur = rec["logp"].float() * w
                logp = cur if logp is None else logp + cur
            hyp = P.greedy_decode(logp.unsqueeze(1),
                                  torch.tensor([dl]), idx2word)[0]
            hyps[sid] = hyp
    return hyps


def compare(args):
    """逐句配对检验：集成 vs 单成员，bootstrap 差值置信区间。"""
    import numpy as np
    from official_wer import CFG, split_gloss_sequence, split_glued, levenshtein

    word2idx, word_set_num, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    labels = P.read_csv_split("dev.csv")
    tags = [x.strip() for x in args.inputs.split(",") if x.strip()]
    weights = [float(x) for x in args.weights.split(",")] if args.weights \
        else [1.0] * len(tags)

    items = {"_order": None}
    for tag in tags:
        blob = torch.load(str(LOGDIR / ("p188_logp_%s.pt" % tag)),
                          map_location="cpu", weights_only=False)["items"]
        items[tag] = blob
        items["_order"] = sorted(blob)

    order = items["_order"]
    refs = [labels[sid][1] for sid in order]
    ref_tok = [split_glued(split_gloss_sequence(g, CFG)) for g in refs]
    n_i = np.array([len(r) for r in ref_tok], dtype=np.float64)

    def edits_of(hyps):
        out = np.zeros(len(order), dtype=np.float64)
        for i, sid in enumerate(order):
            h = split_glued(list(hyps[sid]))
            out[i] = levenshtein(ref_tok[i], h)
        return out

    single = {}
    for tag in tags:
        hyp = _decode_all(items, {tag: 1.0}, idx2word)
        single[tag] = edits_of(hyp)
        log("成员 %-8s dev WER = %.2f%%" % (tag, 100 * single[tag].sum() / n_i.sum()))

    ens_w = {t: w for t, w in zip(tags, weights)}
    ens = edits_of(_decode_all(items, ens_w, idx2word))
    log("集成 %s dev WER = %.2f%%" % (tags, 100 * ens.sum() / n_i.sum()))

    best_tag = min(single, key=lambda t: single[t].sum())
    diff = ens - single[best_tag]
    log("集成 vs 最佳单成员(%s)：per-sentence 更好 %d / 更差 %d / 持平 %d"
        % (best_tag, (diff < 0).sum(), (diff > 0).sum(), (diff == 0).sum()))

    rng = np.random.default_rng(12345)
    B = 20000
    idx = rng.integers(0, len(order), size=(B, len(order)))
    d_num = diff[idx].sum(axis=1)
    d_den = n_i[idx].sum(axis=1)
    boot = 100 * d_num / d_den
    lo, hi = np.percentile(boot, [2.5, 97.5])
    delta = 100 * diff.sum() / n_i.sum()
    log("ΔWER = %+.3f pp，95%% bootstrap CI [%+.3f, %+.3f]，跨0=%s"
        % (delta, lo, hi, "是" if lo <= 0 <= hi else "否"))

    dst = LOGDIR / "p188-ensemble-paired.json"
    dst.write_text(json.dumps({
        "tags": tags, "weights": weights, "best_single": best_tag,
        "single_WER": {t: round(100 * single[t].sum() / n_i.sum(), 3) for t in tags},
        "ensemble_WER": round(100 * ens.sum() / n_i.sum(), 3),
        "delta_pp": round(float(delta), 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
        "crosses_zero": bool(lo <= 0 <= hi),
        "per_sentence": {"better": int((diff < 0).sum()),
                         "worse": int((diff > 0).sum()),
                         "tie": int((diff == 0).sum())},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % dst)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--img", type=int, default=160)
    d.add_argument("--tag", required=True)
    d.add_argument("--max-dev", type=int, default=0)
    c = sub.add_parser("combine")
    c.add_argument("--inputs", required=True)
    c.add_argument("--weights", default="")
    m = sub.add_parser("compare")
    m.add_argument("--inputs", required=True)
    m.add_argument("--weights", default="")
    args = ap.parse_args()
    if args.cmd == "dump":
        return dump(args)
    if args.cmd == "combine":
        return combine(args)
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
