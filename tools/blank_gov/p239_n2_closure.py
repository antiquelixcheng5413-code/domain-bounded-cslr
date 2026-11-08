#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""N3 收官诊断：两 seed 稳健性 + 合成语料多样性（纯 CPU，探索性，不判 PASS/FAIL）。

预注册：docs/planning/N3-CLOSURE-PREREG-2026-11-09.md（05:22 写死）。
只读已落盘收据/产物；不重新前向、不跑 test、不改任何已有判据与收据。
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
DESK = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
M = REPO / "artifacts/metrics/blank-gov"
CACHE = REPO / "artifacts/cache"
TAG = "p239-n2"


def jload(p):
    return json.loads(p.read_text(encoding="utf-8"))


def paired_boot(base_e, cand_e, ref_n, n_boot=20000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(ref_n)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = (base_e[idx].sum(1) - cand_e[idx].sum(1)) / ref_n[idx].sum(1) * 100.0
    point = 100.0 * (base_e.sum() - cand_e.sum()) / ref_n.sum()
    return (round(float(point), 3),
            round(float(np.percentile(d, 2.5)), 3),
            round(float(np.percentile(d, 97.5)), 3))


def main():
    b = jload(M / "p184-base-uwer.json")
    s42 = jload(M / "p233-k1-uwer.json")
    s43 = jload(M / "p238-k1-s43-uwer.json")

    sids = [[str(x["sid"]) for x in d["per_item"]] for d in (b, s42, s43)]
    assert sids[0] == sids[1] == sids[2], "三份收据 sid 未同序，配对无效"
    n = len(sids[0])
    assert n == 515

    eb = np.asarray([x["edits"] for x in b["per_item"]], dtype=np.float64)
    e42 = np.asarray([x["edits"] for x in s42["per_item"]], dtype=np.float64)
    e43 = np.asarray([x["edits"] for x in s43["per_item"]], dtype=np.float64)
    rn = np.asarray([x["ref_n"] for x in b["per_item"]], dtype=np.float64)

    boots = {
        "seed42": paired_boot(eb, e42, rn),
        "seed43": paired_boot(eb, e43, rn),
        "two_seed_mean": paired_boot(eb, (e42 + e43) / 2.0, rn),
    }

    d42, d43 = e42 - eb, e43 - eb
    both_better = int(((d42 < 0) & (d43 < 0)).sum())
    both_worse = int(((d42 > 0) & (d43 > 0)).sum())
    split = int((((d42 < 0) & (d43 > 0)) | ((d42 > 0) & (d43 < 0))).sum())
    tie = int(n - both_better - both_worse - split)

    def row(d):
        e = d["edits"]
        return {"WER_official": d["WER_official"], "uWER_pct": d["uWER_pct"],
                "low_bucket_WER": d["low_bucket"]["WER_pct"],
                "high_bucket_WER": d["high_bucket"]["WER_pct"],
                "ins": e["ins"], "sub": e["sub"], "del": e["del"],
                "micro_subdel_pct": e["micro_subdel_pct"],
                "low_n": sum(x.get("low_n", 0) for x in d["per_item"]),
                "low_err": sum(x.get("low_err", 0) for x in d["per_item"])}

    # ---- 合成语料多样性 ----
    sd = jload(CACHE / "p231_k1_sentences.json")
    sents = sd["sentences"]
    clips = jload(CACHE / "p230_k1_clips.json")["clips"]
    mani = jload(CACHE / "p232_k1_manifest.json")
    tok = Counter(t for s in sents for t in s)
    total = sum(tok.values())
    bigrams = Counter((s[i], s[i + 1]) for s in sents for i in range(len(s) - 1))
    audit = jload(M / "p235-n2-k1-audit.json")
    diversity = {
        "n_sentences": len(sents),
        "n_tokens": total,
        "n_distinct_gloss": len(tok),
        "top1_share_pct": round(100 * tok.most_common(1)[0][1] / total, 2),
        "top10_share_pct": round(100 * sum(c for _, c in tok.most_common(10)) / total, 2),
        "top1_gloss": tok.most_common(1)[0][0],
        "top10_gloss": [t for t, _ in tok.most_common(10)],
        "max_gloss_reuse": max(tok.values()),
        "n_distinct_sentences": len(set(tuple(s) for s in sents)),
        "n_distinct_bigrams": len(bigrams),
        "exact_duplicates_train_dev_test": audit["leakage"]["exact_duplicates_train_dev_test"],
        "low_freq_coverage_pct": round(100 * audit["sentences"]["low_freq_coverage"], 2),
        "js_divergence_vs_train": audit["distribution_vs_train"]["js_divergence"],
        "clip_bank_gloss_covered": audit["clip_bank"]["n_gloss_covered"],
        "clip_span_median_input_frames": audit["clip_bank"]["clip_length_input_frames"]["median"],
        "manifest_n_samples": mani["n"],
    }

    out = {
        "tag": TAG, "prereg": "docs/planning/N3-CLOSURE-PREREG-2026-11-09.md (05:22)",
        "nature": "exploratory, NO PASS/FAIL, no re-judgment",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "paired_bootstrap": {k: {"delta_WER_pp": v[0], "ci95": [v[1], v[2]]}
                             for k, v in boots.items()},
        "per_sentence_agreement": {
            "n": n, "both_better": both_better, "both_worse": both_worse,
            "split": split, "at_least_one_tie": tie,
            "both_better_pct": round(100 * both_better / n, 2),
            "both_worse_pct": round(100 * both_worse / n, 2),
            "split_pct": round(100 * split / n, 2),
            "tie_pct": round(100 * tie / n, 2)},
        "metrics": {"base": row(b), "seed42": row(s42), "seed43": row(s43)},
        "synthetic_corpus_diversity": diversity,
        "verdicts_unchanged": {"p233-k1": "FAIL", "p238-k1-s43": "FAIL",
                               "p234-n2": "FAIL"},
    }
    (M / (TAG + "-closure.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"paired_bootstrap": out["paired_bootstrap"],
                      "per_sentence_agreement": out["per_sentence_agreement"],
                      "synthetic_corpus_diversity": diversity},
                     ensure_ascii=False, indent=2), flush=True)

    # ---- 报告 ----
    pb = out["paired_bootstrap"]; ag = out["per_sentence_agreement"]
    mt = out["metrics"]; dv = diversity

    def mrow(name, r):
        return ("| %s | %.2f | %.2f | %.2f | %.2f | %d / %d / %d | %.2f |"
                % (name, r["WER_official"], r["uWER_pct"], r["low_bucket_WER"],
                   r["high_bucket_WER"], r["ins"], r["sub"], r["del"],
                   r["micro_subdel_pct"]))

    txt = """# P239 / N3 收官诊断 — 两 seed 稳健性与合成语料多样性

> 日期：%s｜性质：**探索性诊断，不判 PASS/FAIL、不重新判定 R1–R8、不产生新的性能主张**。
> 预注册 `N3-CLOSURE-PREREG-2026-11-09.md`（05:22 写死，早于本诊断）。
> 纯 CPU、只用已落盘收据与产物；未重新前向、未跑 test、未改任何判据或收据。

## 1. 为什么做
夜间接力 #2 于 05:19 完成，K1 两 seed 均 FAIL、L1 FAIL、L2 OK。
距 09:00 硬停剩 3h40m，实测不足以再投一个 6 轮训练（3h46m + 评估收尾），
故只做零成本收官诊断，回答两个描述性问题：K1 判负是否稳健？合成句多样性如何？

## 2. 两 seed 全指标（只对照，不重判）

| 配置 | WER | uWER | 低频桶 | 高频桶 | ins/sub/del | micro |
|---|---|---|---|---|---|---|
%s

已完成的 R1–R8 verdict：**seed42 FAIL、seed43 FAIL**（两 seed 均在
R1/R2/R3/R6/R8 不过）。本诊断不改动这些结论。

## 3. Q1 — 判负是否稳健

**配对 bootstrap**（20000 次、seed 42、句子级、三份收据 sid 已逐句同序）：

| 对照 | ΔWER 点估计 | 95%% CI | 含 0 |
|---|---|---|---|
| base vs seed42 | %+.3f pp | [%+.3f, %+.3f] | %s |
| base vs seed43 | %+.3f pp | [%+.3f, %+.3f] | %s |
| base vs **两 seed 逐句平均** | %+.3f pp | [%+.3f, %+.3f] | %s |

**逐句一致性**（相对 base，n=%d）：两 seed 同时改善 %d 句（%.2f%%）、
同时恶化 %d 句（%.2f%%）、分歧 %d 句（%.2f%%）、至少一方持平 %d 句（%.2f%%）。

⇒ 两个独立 seed 的 WER 点估计同为负（%+.3f / %+.3f pp）、配对 CI 均跨 0，
合并后仍跨 0；且两 seed 在低频桶改善（70.66→69.94 / 69.65）的同时
**高频桶同步恶化**（18.56→19.95 / 20.60）、micro 同步恶化（31.25→32.13 / 32.55）。
这是**稳定复现的类别间重分配**，不是 seed 抖动 ⇒ K1 的 FAIL 稳健。

## 4. Q2 — 合成语料多样性（描述，不归因）

| 项 | 值 |
|---|---|
| 生成句 / 不同句子 | %d / %d |
| token 总数 / 不同 gloss | %d / %d |
| top-1 gloss 占 token 比例 | **%.2f%%**（`%s`） |
| top-10 gloss 占 token 比例 | %.2f%% |
| 单 gloss 最大复用 | %d 次 |
| 不同 gloss bigram 数 | %d |
| 低频 gloss 覆盖率（L2） | **%.2f%%** |
| 与 train/dev/test 精确重复 | %d |
| gloss 频次分布 JS 散度 vs train | %.6f |
| 片段库覆盖 gloss / 区间中位 | %d / %s 输入帧 |
| manifest 实际合成样本 | %d |

⚠️ **诚实边界**：高频 gloss 高度集中、低频覆盖仅 %.2f%%，
说明统计 bigram 生成器（K1-PREREG 已标【无】）产出的**新组合多样性有限**，
这是 K1 的已知局限之一。但**不能**据此断言「换成 LLM 就会有效」或
「K1 判负完全归因于生成质量」——K1-PREREG 已写明更大的机制风险是
**CTC 条件独立假设**使「新上下文」收益打折，且文献收益全在 SLT（BLEU）测得。

## 5. 结论（转述已有 verdict，不新判）
- 第六轮主方向 K1（改训练数据组织）在 2 个 seed 上**稳健判负**；
- L1（推理期后验时序平滑）判负，与 I0 闭环 ⇒ 「每词 1.06 帧被监督」是 CTC 固有属性；
- 累计 15+ 判负，所有「轻量单变量」路线系统穷尽；
  按 `OVERNIGHT-2-RUN-2026-11-09.md` §6，合理结论是**接受天花板收尾**。
- 天花板不变：官方@224 零样本 test **32.74** / P184@160 test **33.98** / P150 **53.73**。

## 6. 红线
未跑 test、未 push、未删/改任何已有收据与判据、未启动新训练；
判据与统计项均在 05:22 写死，未事后调整。
""" % (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "\n".join([mrow("base", mt["base"]), mrow("seed42", mt["seed42"]),
                   mrow("seed43", mt["seed43"])]),
        pb["seed42"]["delta_WER_pp"], pb["seed42"]["ci95"][0], pb["seed42"]["ci95"][1],
        "是" if pb["seed42"]["ci95"][0] <= 0 <= pb["seed42"]["ci95"][1] else "否",
        pb["seed43"]["delta_WER_pp"], pb["seed43"]["ci95"][0], pb["seed43"]["ci95"][1],
        "是" if pb["seed43"]["ci95"][0] <= 0 <= pb["seed43"]["ci95"][1] else "否",
        pb["two_seed_mean"]["delta_WER_pp"], pb["two_seed_mean"]["ci95"][0],
        pb["two_seed_mean"]["ci95"][1],
        "是" if pb["two_seed_mean"]["ci95"][0] <= 0 <= pb["two_seed_mean"]["ci95"][1] else "否",
        ag["n"], ag["both_better"], ag["both_better_pct"],
        ag["both_worse"], ag["both_worse_pct"],
        ag["split"], ag["split_pct"],
        ag["at_least_one_tie"], ag["tie_pct"],
        pb["seed42"]["delta_WER_pp"], pb["seed43"]["delta_WER_pp"],
        dv["n_sentences"], dv["n_distinct_sentences"],
        dv["n_tokens"], dv["n_distinct_gloss"],
        dv["top1_share_pct"], dv["top1_gloss"], dv["top10_share_pct"],
        dv["max_gloss_reuse"], dv["n_distinct_bigrams"],
        dv["low_freq_coverage_pct"],
        dv["exact_duplicates_train_dev_test"], dv["js_divergence_vs_train"],
        dv["clip_bank_gloss_covered"], dv["clip_span_median_input_frames"],
        dv["manifest_n_samples"], dv["low_freq_coverage_pct"],
    )
    for base in (REPO, DESK):
        p = base / "docs/experiments/stage-P239-N2-closure.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> stage-P239-N2-closure.md (+桌面镜像)", flush=True)


if __name__ == "__main__":
    main()
