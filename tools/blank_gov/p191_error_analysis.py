#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P191：误差分析（直接读 p188 dump，无需再跑模型）

回答三件事：
  1) 误差构成：sub / del / ins 各占多少（对策完全不同）
  2) 生成率：输出 token 数 / 参考 token 数（<100% 欠生成）
  3) 高频混淆：哪些词最容易互错、哪些词被凭空插入/整片漏掉

用法：
  python tools/blank_gov/p191_error_analysis.py --tag official_test --split test
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import p78_train_official_tfnet as P  # noqa: E402
from official_wer import (evaluate, split_gloss_sequence, CFG,  # noqa: E402
                          split_glued)
from p189_beam_eval import greedy  # noqa: E402

LOGDIR = REPO / "artifacts/metrics/blank-gov"


def align(ref, hyp):
    """编辑距离回溯：返回 (ins, del, sub) 计数与明细。"""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + cost)
    ins = dele = sub = 0
    subs, inss, dels = Counter(), Counter(), Counter()
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if D[i][j] == D[i - 1][j - 1] + cost:
                if cost:
                    sub += 1
                    subs[(ref[i - 1], hyp[j - 1])] += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and D[i][j] == D[i - 1][j] + 1:
            dele += 1
            dels[ref[i - 1]] += 1
            i -= 1
            continue
        ins += 1
        inss[hyp[j - 1]] += 1
        j -= 1
    return ins, dele, sub, subs, inss, dels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="official_test")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    args = ap.parse_args()

    _, _, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    labels = P.read_csv_split(args.split + ".csv")
    blob = torch.load(str(LOGDIR / ("p188_logp_%s.pt" % args.tag)),
                      map_location="cpu", weights_only=False)["items"]

    order = sorted(blob)
    refs, hyps = [], []
    for sid in order:
        rec = blob[sid]
        refs.append(labels[sid][1])
        hyps.append(greedy(rec["logp"], P._conv_len(rec["dl"]), idx2word))

    o = evaluate(refs, hyps)
    tot_ins = tot_del = tot_sub = 0
    n_hyp = n_ref = exact = 0
    unk_hyp = unk_ref = 0
    subs, inss, dels = Counter(), Counter(), Counter()
    buckets = {}
    per_sent = []
    for r, h in zip(refs, hyps):
        rtok = split_glued(split_gloss_sequence(r, CFG))
        htok = split_glued(list(h))
        i_, d_, s_, sc, ic, dc = align(rtok, htok)
        tot_ins += i_
        tot_del += d_
        tot_sub += s_
        subs.update(sc)
        inss.update(ic)
        dels.update(dc)
        n_hyp += len(htok)
        n_ref += len(rtok)
        unk_hyp += sum(1 for t in htok if t == "<unk>")
        unk_ref += sum(1 for t in rtok if t == "<unk>")
        exact += (rtok == htok)
        per_sent.append(100.0 * (i_ + d_ + s_) / max(len(rtok), 1))
        L = len(rtok)
        k = "1-3" if L <= 3 else ("4-6" if L <= 6 else ("7-9" if L <= 9 else "10+"))
        b = buckets.setdefault(k, {"n": 0, "ins": 0, "del": 0, "sub": 0,
                                   "ref": 0, "hyp": 0, "exact": 0})
        b["n"] += 1
        b["ins"] += i_
        b["del"] += d_
        b["sub"] += s_
        b["ref"] += len(rtok)
        b["hyp"] += len(htok)
        b["exact"] += (rtok == htok)

    tot = tot_ins + tot_del + tot_sub
    out = {
        "tag": args.tag, "split": args.split, "n": len(refs),
        "WER_corpus_split": round(o["WER_official"], 2),
        "WER_sentmean": round(sum(per_sent) / len(per_sent), 2),
        "edits": {"sub": tot_sub, "del": tot_del, "ins": tot_ins,
                  "total": tot,
                  "sub_pct": round(100.0 * tot_sub / max(tot, 1), 1),
                  "del_pct": round(100.0 * tot_del / max(tot, 1), 1),
                  "ins_pct": round(100.0 * tot_ins / max(tot, 1), 1)},
        "gen_rate_pct": round(100.0 * n_hyp / max(n_ref, 1), 1),
        "n_ref_tokens": n_ref, "n_hyp_tokens": n_hyp,
        "exact_rate_pct": round(100.0 * exact / len(refs), 2),
        "unk_hyp": unk_hyp, "unk_ref": unk_ref,
        "top_sub": [[list(k), v] for k, v in subs.most_common(20)],
        "top_ins": inss.most_common(15),
        "top_del": dels.most_common(15),
        "by_len": {k: {"n": b["n"],
                       "wer": round(100.0 * (b["ins"] + b["del"] + b["sub"])
                                    / max(b["ref"], 1), 2),
                       "gen_rate": round(100.0 * b["hyp"] / max(b["ref"], 1), 1),
                       "exact": round(100.0 * b["exact"] / b["n"], 1)}
                   for k, b in buckets.items()},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

    dst = LOGDIR / ("p191-error-%s-%s.json" % (args.split, args.tag))
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[P191] 收据 ->", dst, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
