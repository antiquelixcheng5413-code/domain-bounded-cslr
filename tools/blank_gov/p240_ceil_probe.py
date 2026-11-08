#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P240 / M0+M1：不可约误差与组合泛化分层诊断（零成本 CPU，只读 train/dev 标注）。

- M0：dev 中 train freq==0 的 gloss / token 占比（OOV 不可约误差前提）
- M1：dev 未见 gloss bigram 占比，以及「含未见 bigram / 全已见」句子的分层 WER
只做 dev，不碰 test、不跑模型、不改任何判据。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
DESK = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78       # noqa: E402
import DataProcessMoudle as DPM              # noqa: E402

M = REPO / "artifacts/metrics/blank-gov"
TAG = "p240"


def toks(seq):
    return [t for t in DPM.PreWords(str(seq).split("/")) if t]


def main():
    tr = P78.read_csv_split("train.csv")
    dv = P78.read_csv_split("dev.csv")

    freq = Counter(t for _p, g in tr.values() for t in toks(g))
    tr_bi, tr_sents = Counter(), set()
    for _p, g in tr.values():
        s = toks(g)
        tr_sents.add(tuple(s))
        tr_bi.update(zip(s, s[1:]))

    dev_sents = [(sid, toks(g)) for sid, (_p, g) in dv.items()]
    dev_tokens = [t for _sid, s in dev_sents for t in s]
    distinct = set(dev_tokens)
    oov = {t for t in distinct if freq.get(t, 0) == 0}
    oov_tok = sum(1 for t in dev_tokens if t in oov)

    dev_bi = Counter()
    for _sid, s in dev_sents:
        dev_bi.update(zip(s, s[1:]))
    seen_bi = sum(c for b, c in dev_bi.items() if b in tr_bi)
    n_bi = sum(dev_bi.values())
    seen_sent = sum(1 for _sid, s in dev_sents if tuple(s) in tr_sents)

    base = json.loads((M / "p184-base-uwer.json").read_text(encoding="utf-8"))
    by_sid = {x["sid"]: x for x in base["per_item"]}

    def group(rows):
        if not rows:
            return {"n": 0, "WER_pct": 0.0, "low_bucket_WER_pct": 0.0,
                    "ins": 0, "sub": 0, "del": 0}
        e = sum(r["edits"] for r in rows)
        n = sum(r["ref_n"] for r in rows)
        lo_n = sum(r.get("low_n", 0) for r in rows)
        lo_e = sum(r.get("low_err", 0) for r in rows)
        return {"n": len(rows), "WER_pct": round(100 * e / max(n, 1), 2),
                "low_bucket_WER_pct": round(100 * lo_e / max(lo_n, 1), 2),
                "ins": sum(r["ins"] for r in rows),
                "sub": sum(r["sub"] for r in rows),
                "del": sum(r["del"] for r in rows)}

    g_unseen, g_seen, g_oov, g_no_oov = [], [], [], []
    n_unseen_bi_tokens = 0
    for sid, s in dev_sents:
        r = by_sid.get(sid)
        unseen_pairs = [(a, b) for a, b in zip(s, s[1:]) if (a, b) not in tr_bi]
        if unseen_pairs:
            n_unseen_bi_tokens += len(unseen_pairs)
        if r is None:
            continue          # read_csv_split 的表头 / 收据未覆盖行
        (g_unseen if unseen_pairs else g_seen).append(r)
        (g_oov if any(t in oov for t in s) else g_no_oov).append(r)

    out = {
        "tag": TAG, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "只读 train/dev 标注 + p184-base 收据；未跑 test、未跑模型、未改判据",
        "vocab": {"train_tokens": sum(freq.values()),
                  "train_distinct_gloss": len(freq),
                  "dev_sentences_labels": len(dev_sents),
                  "dev_sentences_matched_receipt": len(by_sid),
                  "dev_tokens": len(dev_tokens),
                  "dev_distinct_gloss": len(distinct)},
        "M0_oov": {
            "dev_oov_gloss": len(oov),
            "dev_oov_token": oov_tok,
            "dev_oov_token_pct": round(100 * oov_tok / max(len(dev_tokens), 1), 2),
            "dev_token_freq_le_1_pct": round(100 * sum(1 for t in dev_tokens if freq.get(t, 0) <= 1) / len(dev_tokens), 2),
            "dev_token_freq_le_10_pct": round(100 * sum(1 for t in dev_tokens if freq.get(t, 0) <= 10) / len(dev_tokens), 2),
            "verdict": "OOV_PREMISE_FALSE" if oov_tok == 0 else "OOV_EXISTS",
        },
        "M1_compositional": {
            "dev_bigram_total": n_bi,
            "dev_bigram_seen_in_train": seen_bi,
            "dev_bigram_unseen_pct": round(100 * (n_bi - seen_bi) / max(n_bi, 1), 2),
            "dev_sentences_exact_in_train": seen_sent,
            "dev_sentence_exact_match_pct": round(100 * seen_sent / max(len(dev_sents), 1), 2),
            "sentences_with_unseen_bigram": len(g_unseen),
            "sentences_all_bigrams_seen": len(g_seen),
        },
        "stratified_WER": {
            "all": group(list(by_sid.values())),
            "has_unseen_bigram": group(g_unseen),
            "all_bigrams_seen": group(g_seen),
            "has_oov_gloss": group(g_oov),
            "no_oov_gloss": group(g_no_oov),
        },
    }
    (M / (TAG + "-ceil-probe.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

    v, c, w = out["vocab"], out["M1_compositional"], out["stratified_WER"]
    o = out["M0_oov"]
    txt = """# P240 / M0+M1 不可约误差与组合泛化诊断 — 阶段报告

> 日期：%s｜零成本 CPU、只读 `train.csv` / `dev.csv` 标注与 `p184-base-uwer.json`。
> **未跑 test、未跑模型、未改任何判据**；探索性诊断，不判 PASS/FAIL。

## 1. 为什么要做
2026 年系统性检索确认：CSLR 文献里**没有**「长尾 × CTC 峰值 × 未见 gloss」的分层误差诊断，
而 Isharah 的 unseen-sentence 任务 WER 50.72%% vs signer-independent 16.35%%（差 34pp）
提示「OOV / 组合泛化」可能构成硬天花板。若本地成立，则过去 15+ 个判负实验
都可能只是在优化一个被数据本身限死的指标。

## 2. M0 — 未见 gloss（OOV）前提：**不成立**

| 项 | 值 |
|---|---|
| train token / 不同 gloss | %d / %d |
| dev 句（标注 / 收据匹配） | %d / %d |
| dev token / 不同 gloss | %d / %d |
| **dev 中 train freq==0 的 gloss / token** | **%d 类 / %d token（%.2f%%）** |
| dev token 的 train 频次 ≤1 / ≤10 | %.2f%% / %.2f%% |

⇒ **OOV = 0%%**，与 `OFFICIAL_EVAL_PROTOCOL.md`「dev OOV = 0%%」及
`stage-P184` §「无 `<unk>`，词表覆盖没问题」一致。
**闭集词表外的不可约误差在本地不是瓶颈，该方向一票否决。**
本地 VAC 时代（`p8_error_attribution.py` / `p45c_error_attribution.py`、
`docs/blank_governance/实验成果汇总.md`）也做过 OOV 拆解，本诊断是在 TFNet 主线上的复核。

## 3. M1 — 组合泛化：近一半 bigram 训练没见过

| 项 | 值 |
|---|---|
| dev gloss bigram 总数 | %d |
| 在 train 见过的 bigram | %d（**%.2f%%**） |
| **未见 bigram 占比** | **%.2f%%** |
| dev 整句在 train 精确出现 | %d / %d（%.2f%%） |
| 含未见 bigram 的句子 / 全已见句子 | %d / %d |

⇒ 词表全闭合，但**近一半相邻 gloss 组合是训练没见过的**，整句几乎全新（重复率 %.2f%%）。
这与文献 unseen-sentence 协议的高 WER 同型，也解释了 train WER 1.03%% vs dev 34.34%%
的 33.3pp 间隙不只是「过拟合」，还包含真实的**组合泛化**压力。

## 4. 分层 WER（p184 基线，dev 515 句，语料级）

| 分组 | 句数 | WER | 低频桶 WER | ins/sub/del |
|---|---|---|---|---|
| 全量 | %d | %.2f | %.2f | %d/%d/%d |
| **含未见 bigram** | %d | **%.2f** | %.2f | %d/%d/%d |
| 全部 bigram 已见 | %d | **%.2f** | %.2f | %d/%d/%d |

（含 OOV gloss 的句子 %d 句，WER %.2f；不含 OOV %d 句，WER %.2f。）

## 5. 结论与边界
- **OOV 不是瓶颈**（0%%）；**组合泛化压力真实存在**（未见 bigram %.2f%%）。
- 但「新组合」这条路本地**已经从两侧试过并判负**：
  训练侧 K1 语料合成（生成 1500 条新 gloss 句，2 seed 稳健 FAIL）、
  解码侧 P197 bigram LM / D1 Qwen2.5-1.5B 重排（均 FAIL）。
  ⇒ 本诊断**不产生新的可执行方向**，只是把「为什么一直卡在 34%%」定位得更准。
- ⚠️ 诚实边界：分层 WER 是**观察性**的，含未见 bigram 的句子可能同时更长、
  低频词更多，不能把差值全归因于「组合泛化因果效应」；本报告不做因果断言。
- 未跑 test、未改任何收据与判据。
""" % (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        v["train_tokens"], v["train_distinct_gloss"],
        v["dev_sentences_labels"], v["dev_sentences_matched_receipt"],
        v["dev_tokens"], v["dev_distinct_gloss"],
        o["dev_oov_gloss"], o["dev_oov_token"], o["dev_oov_token_pct"],
        o["dev_token_freq_le_1_pct"], o["dev_token_freq_le_10_pct"],
        c["dev_bigram_total"], c["dev_bigram_seen_in_train"],
        round(100 - c["dev_bigram_unseen_pct"], 2),
        c["dev_bigram_unseen_pct"],
        c["dev_sentences_exact_in_train"], v["dev_sentences_labels"],
        c["dev_sentence_exact_match_pct"],
        w["has_unseen_bigram"]["n"], w["all_bigrams_seen"]["n"],
        c["dev_sentence_exact_match_pct"],
        w["all"]["n"], w["all"]["WER_pct"], w["all"]["low_bucket_WER_pct"],
        w["all"]["ins"], w["all"]["sub"], w["all"]["del"],
        w["has_unseen_bigram"]["n"], w["has_unseen_bigram"]["WER_pct"],
        w["has_unseen_bigram"]["low_bucket_WER_pct"],
        w["has_unseen_bigram"]["ins"], w["has_unseen_bigram"]["sub"],
        w["has_unseen_bigram"]["del"],
        w["all_bigrams_seen"]["n"], w["all_bigrams_seen"]["WER_pct"],
        w["all_bigrams_seen"]["low_bucket_WER_pct"],
        w["all_bigrams_seen"]["ins"], w["all_bigrams_seen"]["sub"],
        w["all_bigrams_seen"]["del"],
        w["has_oov_gloss"]["n"], w["has_oov_gloss"]["WER_pct"],
        w["no_oov_gloss"]["n"], w["no_oov_gloss"]["WER_pct"],
        c["dev_bigram_unseen_pct"],
    )
    for base_dir in (REPO, DESK):
        p = base_dir / "docs/experiments/stage-P240-irreducible-error.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> stage-P240-irreducible-error.md (+桌面镜像)", flush=True)


if __name__ == "__main__":
    main()
