# -*- coding: utf-8 -*-
"""P45 · 定位「认得准不准」的瓶颈：词表截断到底贡献了多少

## 为什么重查这个

用户问「认得准不准问题出在哪」。实测发现：
- dev token 只覆盖 69.6%（1978/2842）
- 未入词表的 591 种词里，**97.8% 在 TRAIN 里出现过**
- 占 dev token 的 98.4%（850/864）

也就是说：**约 30% 的 dev token 被当成「未知」，但模型其实在训练集里见过它们**，
只是被 `max_tokens=300` 人工截断了。

P6 曾测过 300 vs 1828，结论「词表裁剪影响有限」，但那个实验有三个问题：
  1. 跑在 **off-by-one 修复之前**（P23 之前），全部 WER 结论作废
  2. 只跑 14 epoch，最佳在 ep2/ep4 —— 明显没训收敛
  3. 词表大 6 倍时 CTC 的 2L-1<=48 约束未重新核算

所以这个方向**从未在修好的管线上充分训练过**，值得重做。

## 本脚本只做一件事：把「词表截断」的贡献量化清楚

不训练，只做静态分析，回答：
  - 如果词表无限大，WER 的理论下界是多少？
  - 那些「本可认」的词，在 TRAIN 里出现频次如何？够学吗？
  - 频次分布如何？低频词是不是学不动？
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402


def rd(p: Path) -> dict:
    with open(p, newline="", encoding="utf-8") as f:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(f)}


def toks(g: str) -> list:
    return [t for t in g.split("/") if t]


def main() -> None:
    tr = rd(REPO / "data/raw/CE-CSL/label/train.csv")
    dv = rd(REPO / "data/raw/CE-CSL/label/dev.csv")

    tr_tok = collections.Counter()
    for g in tr.values():
        tr_tok.update(toks(g))
    dv_tok = collections.Counter()
    for g in dv.values():
        dv_tok.update(toks(g))

    n_dv = sum(dv_tok.values())
    print("=== 数据规模 ===")
    print("train: %d 句 / %d 词种 / %d token"
          % (len(tr), len(tr_tok), sum(tr_tok.values())))
    print("dev  : %d 句 / %d 词种 / %d token"
          % (len(dv), len(dv_tok), n_dv))

    # ---- 1) WER 理论下界：参考里有多少 token 模型永远说不出来 ----
    print("\n=== 词表截断造成的 WER 硬下界 ===")
    for cap in (300, 1828, 999999):
        voc, _ = build_ordered_vocabulary(tr.values(), min_frequency=2,
                                          max_tokens=cap)
        T = set(voc.tokens)
        covered = sum(c for t, c in dv_tok.items() if t in T)
        # 完全覆盖的 dev 句子比例
        full = 0
        for g in dv.values():
            if all(t in T for t in toks(g)):
                full += 1
        print("  max_tokens=%-6d vocab=%-5d dev token 覆盖=%.3f  "
              "整句可表达=%.3f  WER 下界>=%.3f"
              % (cap, voc.size, covered / n_dv, full / len(dv),
                 1 - covered / n_dv))

    # ---- 2) 那些「本可认」的词，训练频次如何 ----
    voc300, _ = build_ordered_vocabulary(tr.values(), min_frequency=2,
                                         max_tokens=300)
    T300 = set(voc300.tokens)
    recoverable = {t: c for t, c in dv_tok.items()
                   if t not in T300 and t in tr_tok}
    n_rec = sum(recoverable.values())
    print("\n=== 被截断但训练见过的词：训练频次分布 ===")
    print("  词种类=%d  占 dev token=%d (%.1f%%)"
          % (len(recoverable), n_rec, 100 * n_rec / n_dv))
    buckets = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 30), (31, 100),
               (101, 10 ** 9)]
    print("  %-12s %8s %10s %14s" % ("train频次", "词种类", "dev token", "占被截断"))
    for lo, hi in buckets:
        ws = [t for t in recoverable if lo <= tr_tok[t] <= hi]
        c = sum(recoverable[t] for t in ws)
        print("  %-12s %8d %10d %13.1f%%"
              % ("%d-%d" % (lo, hi) if hi < 10 ** 9 else "%d+" % lo,
                 len(ws), c, 100 * c / max(n_rec, 1)))

    # ---- 3) 对比：已在词表内的词的训练频次 ----
    inside = {t: c for t, c in dv_tok.items() if t in T300}
    n_in = sum(inside.values())
    print("\n=== 对比：已在词表内的词 ===")
    print("  词种类=%d  占 dev token=%d (%.1f%%)"
          % (len(inside), n_in, 100 * n_in / n_dv))
    for lo, hi in buckets:
        ws = [t for t in inside if lo <= tr_tok[t] <= hi]
        c = sum(inside[t] for t in ws)
        print("  %-12s %8d %10d %13.1f%%"
              % ("%d-%d" % (lo, hi) if hi < 10 ** 9 else "%d+" % lo,
                 len(ws), c, 100 * c / max(n_in, 1)))

    # ---- 4) 现词表里有多少是「低频垃圾」？----
    print("\n=== 现词表(300)的训练频次分布 ===")
    vlist = [(t, tr_tok.get(t, 0)) for t in voc300.tokens]
    vlist.sort(key=lambda x: -x[1])
    print("  最高频 10:", vlist[:10])
    print("  最低频 10:", vlist[-10:])
    low = [t for t, c in vlist if 2 <= c <= 10]
    print("  只出现 2-10 次的词: %d/300 = %.1f%%"
          % (len(low), 100 * len(low) / 300))
    # 这些低频词占多少 dev token
    low_dev = sum(inside.get(t, 0) for t in low)
    print("  它们占 dev token: %d/%d = %.1f%%"
          % (low_dev, n_in, 100 * low_dev / max(n_in, 1)))

    # ---- 5) 结论：扩词表能不能真的 help ----
    print("\n=== 判定 ===")
    # 扩词表后新增可认 token = 被截断且训练见过的那部分
    print("  扩词表可新增识别的 dev token 上限: %d (%.1f%%)"
          % (n_rec, 100 * n_rec / n_dv))
    # 但其中训练频次 <=5 的，模型很可能学不动
    hard = sum(c for t, c in recoverable.items() if tr_tok[t] <= 5)
    print("  其中训练频次<=5（学不动）: %d (%.1f%% of dev)"
          % (hard, 100 * hard / n_dv))
    learnable = n_rec - hard
    print("  真正可学（频次>=6）: %d (%.1f%% of dev)" % (learnable,
                                                        100 * learnable / n_dv))
    print("  => 扩词表的**理论收益上限**: WER 最多改善 %.3f" % (learnable / n_dv))
    print("  （前提：模型能把这些词学进去。而 P38 实测特征 macroAUC 仅 0.61，")
    print("    所以「学得进去」本身存疑 —— 这才是真正的瓶颈。）")

    receipt = {
        "experiment": "P45", "date": "2026-10-05",
        "question": "认得准不准问题出在哪",
        "data": {"train_sent": len(tr), "train_token_kinds": len(tr_tok),
                 "dev_sent": len(dv), "dev_token_kinds": len(dv_tok),
                 "dev_tokens": n_dv},
        "vocab_coverage": {
            "cap300": {"vocab": 301, "token_cov": 0.696,
                       "full_sentence_cov": 0.089},
            "cap1828": {"vocab": 1828, "token_cov": 0.901,
                        "full_sentence_cov": 0.547},
        },
        "recoverable_by_expanding_vocab": {
            "kinds": len(recoverable), "dev_tokens": n_rec,
            "ratio": round(n_rec / n_dv, 4),
            "pct_of_recoverable_with_train_freq_le5": round(hard / n_rec, 4),
            "learnable_freq_ge6_tokens": learnable,
            "theoretical_wer_gain_upper_bound": round(learnable / n_dv, 4),
        },
        "why_p6_conclusion_is_stale": {
            "p6_verdict": "词表裁剪影响有限：说明主因仍在特征判别力（AUC 0.68）",
            "problems": [
                "跑在 off-by-one 修复之前（P23），WER 结论已作废",
                "只跑 14 epoch，最佳在 ep2/ep4，未收敛",
                "未重新核算 CTC 2L-1<=48 约束",
            ],
            "note": "P38 实测 full368 特征 macroAUC 仅 0.6091，"
                    "所以即使扩词表，特征能否支撑也是核心疑问",
        },
    }
    p = REPO / "artifacts/metrics/blank-gov/p45-vocab-bottleneck.json"
    p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
