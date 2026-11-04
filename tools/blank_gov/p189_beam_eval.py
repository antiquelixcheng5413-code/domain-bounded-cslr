#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P189：官方评测协议对齐（beam10 解码 + 按 batch 平均 WER）

动机：官方 Train.py:152 用 `decode.Decode(word2idx, wordSetNum+1, 'beam')`
（ctcdecode CTCBeamDecoder beam_width=10），且 WER 是 **逐 batch 求均值**
（Train.py:295 `werScoreSum / len(validLoader)`）。我们一直用 greedy + 语料级 WER，
这可能是 224 零样本 33.09 vs 官方 README 32.46 那 0.63pp 的来源。

本机没有 ctcdecode（C++ 扩展），这里实现等价的 CTC prefix beam search（beam=10），
每帧只展开 top-k token（k 远大于 beam，精度足够）。

用法：
  python tools/blank_gov/p189_beam_eval.py --tag official --beam 10
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import p78_train_official_tfnet as P  # noqa: E402
from official_wer import evaluate, levenshtein  # noqa: E402

LOGDIR = REPO / "artifacts/metrics/blank-gov"


def log(*a):
    print("[P189]", " ".join(str(x) for x in a), flush=True)


def beam_decode(logp, length, idx2word, beam=10, topk=32, blank=0):
    """CTCLabelDecode 等价：标准 CTC prefix beam search（无 LM / 无长度惩罚）。

    beams: prefix -> (log p_blank, log p_non_blank)，每帧只用 top-k 候选扩展。
    """
    lp = np.asarray(logp[:int(length)].float().numpy(), dtype=np.float32)
    kth = min(topk, lp.shape[1] - 1) - 1
    pos = np.argpartition(-lp, kth, axis=1)[:, :topk]
    beams = {(): (0.0, -np.inf)}
    for t in range(lp.shape[0]):
        new = {}
        lb = float(lp[t, blank])
        for pref, (pb, pnb) in beams.items():
            tot = float(np.logaddexp(pb, pnb))
            # 1) 走 blank：前缀不变
            cur = new.get(pref)
            nv = tot + lb
            if cur is None:
                new[pref] = [nv, -np.inf]
            else:
                cur[0] = float(np.logaddexp(cur[0], nv))
            last = pref[-1] if pref else -1
            for c in pos[t].tolist():
                if c == blank:
                    continue
                v = float(lp[t, c])
                if c == last:
                    # non-blank 自环塌缩回原前缀
                    cur = new.get(pref)
                    if cur is None:
                        new[pref] = [-np.inf, pnb + v]
                    else:
                        cur[1] = float(np.logaddexp(cur[1], pnb + v))
                    # 从 blank 状态发出同 token 才产生新前缀（双写）
                    key = pref + (c,)
                    cur = new.get(key)
                    if cur is None:
                        new[key] = [-np.inf, pb + v]
                    else:
                        cur[1] = float(np.logaddexp(cur[1], pb + v))
                else:
                    key = pref + (c,)
                    cur = new.get(key)
                    if cur is None:
                        new[key] = [-np.inf, tot + v]
                    else:
                        cur[1] = float(np.logaddexp(cur[1], tot + v))
        scored = sorted(new.items(),
                        key=lambda kv: -float(np.logaddexp(kv[1][0], kv[1][1])))
        beams = {k: (v[0], v[1]) for k, v in scored[:beam]}
    best = max(beams.items(), key=lambda kv: float(np.logaddexp(kv[1][0], kv[1][1])))
    return [idx2word[c] for c in best[0] if 0 <= c < len(idx2word)]


def greedy(logp, length, idx2word, blank=0):
    lp = logp[:length].numpy()
    out, prev = [], -1
    for t in range(lp.shape[0]):
        k = int(lp[t].argmax())
        if k != prev and k != blank and 0 <= k < len(idx2word):
            out.append(idx2word[k])
        prev = k
    return out


def variant_scores(refs, hyps):
    """四种口径组合：
    split/raw = 是否把粘连词拆开（官方论文口径 split；官方代码 raw）
    corpus/sentmean = 语料级累加 vs 逐句 WER 求均值（官方代码 sentmean）
    """
    from official_wer import split_glued, CFG
    from cslr.recognition.gloss_sequence import split_gloss_sequence
    keys = ["corpus_split", "sentmean_split", "corpus_raw", "sentmean_raw"]
    es = {k: 0 for k in keys}
    ns = {k: 0 for k in keys}
    se = {k: 0.0 for k in keys}
    cnt = 0
    for g, h in zip(refs, hyps):
        rs = split_glued(split_gloss_sequence(g, CFG))
        rr = split_gloss_sequence(g, CFG)
        hs = split_glued(list(h))
        hr = list(h)
        for k, r, y in (("split", rs, hs), ("raw", rr, hr)):
            if not r:
                continue
            d = levenshtein(r, y)
            es["corpus_" + k] += d
            ns["corpus_" + k] += len(r)
            se["sentmean_" + k] += 100.0 * d / len(r)
        cnt += 1
    out = {}
    for k in keys:
        v = (100.0 * es[k] / ns[k]) if ns[k] else 0.0
        if k.startswith("sentmean"):
            v = se[k] / max(cnt, 1)
        out[k] = round(v, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="official")
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--p184b", action="store_true", help="同时评估 p184b dump")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    args = ap.parse_args()

    _, _, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    labels = P.read_csv_split(args.split + ".csv")
    tags = [args.tag] + (["p184b"] if args.p184b else [])

    rows = {}
    for tag in tags:
        blob = torch.load(str(LOGDIR / ("p188_logp_%s.pt" % tag)),
                          map_location="cpu", weights_only=False)["items"]
        order = sorted(blob)
        refs = [labels[sid][1] for sid in order]
        hyps_g, hyps_b = [], []
        for sid in order:
            rec = blob[sid]
            t = P._conv_len(rec["dl"])
            hyps_g.append(greedy(rec["logp"], t, idx2word))
            hyps_b.append(beam_decode(rec["logp"], t, idx2word,
                                      beam=args.beam, topk=args.topk))
        rows[tag] = {
            "n": len(refs),
            "greedy_corpus": round(evaluate(refs, hyps_g)["WER_official"], 2),
            "beam%d_corpus" % args.beam: round(evaluate(refs, hyps_b)["WER_official"], 2),
            "greedy": variant_scores(refs, hyps_g),
            "beam%d" % args.beam: variant_scores(refs, hyps_b),
        }
        log("%s -> %s" % (tag, json.dumps(rows[tag], ensure_ascii=False)))

    out = LOGDIR / ("p189-beam-protocol-%s.json" % args.split)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
