#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P221 / G0-lite：dev 混淆对在 train 标注里的频次与共现分布（零训练，纯 CPU）。

第三轮方案 G1 想用「train 集上挖出的混淆对」做 hard negatives，
但 train WER 仅 1.03%（强过拟合），train 上可能根本不犯错 ⇒ G1 前提存疑。
本项不做 train 前向（需 GPU/20min），先用最便宜的方式给前提定界：
把 P219 的 693 个 dev 混淆对放回 train 标注，看它们的频次与同句共现。

预注册（08:20）：只作前提描述，不判 PASS/FAIL、不进 test。
若混淆对绝大多数是「双方 train 频次都很低 + 几乎不共现」，
则 train 混淆表大概率采不到这些对 ⇒ G1 需改为外部/跨折挖掘，不能直接照搬。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
DESK = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402
import DataProcessMoudle as DPM  # noqa: E402
from p197_beam_lm import greedy_fast  # noqa: E402
from p219_subpair_semantics import dp_pairs  # noqa: E402

M = REPO / "artifacts/metrics/blank-gov"


def main():
    labels_tr = P78.read_csv_split("train.csv")
    labels_dv = P78.read_csv_split("dev.csv")
    word2idx, n, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")

    sent_tokens = []
    freq = Counter()
    for _sid, (_p, seq) in labels_tr.items():
        toks = list(DPM.PreWords(str(seq).split("/")))
        sent_tokens.append(set(toks))
        freq.update(toks)
    print("train sentences=%d  distinct=%d" % (len(sent_tokens), len(freq)), flush=True)

    d = torch.load(M / "p188_logp_p184.pt", map_location="cpu", weights_only=False)
    pairs = []
    for s in d["items"]:
        if s not in labels_dv:
            continue
        r = OW.split_glued(OW.split_gloss_sequence(labels_dv[s][1], OW.CFG))
        lp = d["items"][s]["logp"].numpy() if hasattr(d["items"][s]["logp"], "numpy") \
            else np.asarray(d["items"][s]["logp"])
        h = OW.split_glued(list(greedy_fast(lp, idx2word)))
        ps, _ = dp_pairs(r, h)
        pairs.extend(ps)
    print("dev sub pairs=%d (unique=%d)" % (len(pairs), len(set(pairs))), flush=True)

    ref_f, hyp_f, co = [], [], 0
    for a, b in pairs:
        fa, fb = freq.get(a, 0), freq.get(b, 0)
        ref_f.append(fa)
        hyp_f.append(fb)
    uniq = set(pairs)
    for a, b in uniq:
        if any((a in s and b in s) for s in sent_tokens):
            co += 1
    rng = np.random.default_rng(42)
    vocab = [t for t, c in freq.items() if c > 0]
    rnd_pairs = [(vocab[int(i)], vocab[int(j)]) for i, j in
                 rng.integers(0, len(vocab), size=(500, 2))]
    rnd_co = sum(1 for a, b in rnd_pairs
                 if any((a in s and b in s) for s in sent_tokens))

    rf, hf = np.asarray(ref_f), np.asarray(hyp_f)
    out = {
        "tag": "p221", "kind": "prerequisite_diagnostic_no_verdict",
        "train_sentences": len(sent_tokens),
        "n_dev_sub_pairs": len(pairs), "n_unique_pairs": len(uniq),
        "ref_train_freq_median": float(np.median(rf)),
        "hyp_train_freq_median": float(np.median(hf)),
        "ref_freq_le10_share": round(float((rf <= 10).mean()), 4),
        "hyp_freq_le10_share": round(float((hf <= 10).mean()), 4),
        "both_freq_le10_share": round(float(((rf <= 10) & (hf <= 10)).mean()), 4),
        "pair_cooccur_in_train_sentence": co,
        "pair_cooccur_rate": round(co / max(len(uniq), 1), 4),
        "random_pair_cooccur_rate": round(rnd_co / max(len(rnd_pairs), 1), 4),
        "top_pairs": [{"ref": a, "hyp": b, "count": c,
                       "ref_freq": freq.get(a, 0), "hyp_freq": freq.get(b, 0)}
                      for (a, b), c in Counter(pairs).most_common(10)],
    }
    (M / "p221-g0-prior.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "top_pairs"},
                     ensure_ascii=False, indent=2), flush=True)

    verdict = ("混淆对以低频、几乎不共现为主 ⇒ train 混淆表大概率采不到这些对，"
               "G1 不能直接在 train 上照搬，需交叉验证/留出折挖掘或改用外部最小对立体标注。"
               if out["both_freq_le10_share"] > 0.5 or out["pair_cooccur_rate"] < 0.1
               else "混淆对在 train 中有可观共现 ⇒ G1 的 train 挖掘前提可能成立，"
                    "但仍需实际跑 train 前向确认（本项未做）。")
    txt = """# P221 / G0-lite 混淆对的 train 先验分布 — 前提诊断

> 生成：%s｜零训练纯 CPU｜**不判 PASS/FAIL、不进 test**

## 1. 为什么做
第三轮方案 G1 想用「train 集挖出的混淆对」做 hard negatives，但 train WER 仅 1.03%%
（强过拟合），train 上可能根本不犯错。本项不放 train 前向（需 GPU），
先把 dev 的 %d 个替换对放回 train 标注，给定性边界。

## 2. 结果

| 指标 | 值 |
|---|---|
| train 句数 / 去重词数 | %d / %d |
| dev 替换对（去重） | %d（共 %d 次） |
| ref 的 train 频次中位数 | %.1f |
| hyp 的 train 频次中位数 | %.1f |
| ref 频次 ≤10 占比 | %.1f%% |
| hyp 频次 ≤10 占比 | %.1f%% |
| **双方频次都 ≤10 占比** | **%.1f%%** |
| 混淆对在 train 同句共现 | %d / %d = **%.2f%%** |
| 随机词对共现率（对照） | %.2f%% |

高频替换对的 train 频次：
%s

## 3. 对 G1 的结论

**%s**

## 4. 诚实边界
- 本项**未跑 train 前向**，因此不能回答「train 上实际会混淆哪些词」，
  只能回答「dev 混淆对在 train 标注里有多稀疏」。G1 的最终前提仍需 train 后验（约 20 min GPU）。
- 共现按「同句是否同时出现」计算，非真实最小对立体标注。
- 文本/频次统计不能替代训练信号；不进 test、不判 PASS/FAIL。

## 5. 状态
未跑 test、未 commit、未 push。数据来源 `artifacts/metrics/blank-gov/p221-g0-prior.json`。
""" % (
        datetime.now().strftime("%Y-%m-%d %H:%M"), len(pairs),
        len(sent_tokens), len(freq), len(uniq), len(pairs),
        float(np.median(rf)), float(np.median(hf)),
        100 * float((rf <= 10).mean()), 100 * float((hf <= 10).mean()),
        100 * float(((rf <= 10) & (hf <= 10)).mean()),
        co, len(uniq), 100.0 * co / max(len(uniq), 1),
        100.0 * rnd_co / max(len(rnd_pairs), 1),
        "\n".join("- `%s`→`%s` ×%d（train 频次 %d / %d）"
                  % (x["ref"], x["hyp"], x["count"], x["ref_freq"], x["hyp_freq"])
                  for x in out["top_pairs"]),
        verdict,
    )
    for p in (REPO / "docs/experiments/stage-P221-g0-prior.md",
              DESK / "docs/experiments/stage-P221-g0-prior.md"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> stage-P221-g0-prior.md (+桌面镜像)", flush=True)


if __name__ == "__main__":
    main()
