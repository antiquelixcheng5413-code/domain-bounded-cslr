#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""N2 / L2：K1 语料质量与对齐边界审计（零成本 CPU 诊断，不判 PASS/FAIL）。

只为 K1 的 verdict 提供机制解释；**不得据此改 K1 判据**。
见 docs/planning/N2-PREREG-2026-11-08.md §3（20:55 写死）。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
DESK = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78       # noqa: E402
import DataProcessMoudle as DPM              # noqa: E402

CACHE = REPO / "artifacts/cache"
M = REPO / "artifacts/metrics/blank-gov"
TAG = "p235-n2"


def js(p, q):
    keys = set(p) | set(q)
    a = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64)
    b = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64)
    a /= max(a.sum(), 1e-12); b /= max(b.sum(), 1e-12)
    m = 0.5 * (a + b)
    def kl(x, y):
        mask = x > 0
        return float((x[mask] * np.log(x[mask] / np.maximum(y[mask], 1e-12))).sum())
    return round(0.5 * kl(a, m) + 0.5 * kl(b, m), 6)


def pct(xs):
    if not xs:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "p90": 0, "max": 0,
                "mean": 0}
    a = np.asarray(xs, dtype=float)
    return {"min": round(float(a.min()), 2),
            "p25": round(float(np.percentile(a, 25)), 2),
            "median": round(float(np.median(a)), 2),
            "p75": round(float(np.percentile(a, 75)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
            "max": round(float(a.max()), 2),
            "mean": round(float(a.mean()), 2)}


def canon(seq):
    return "/".join(DPM.PreWords(str(seq).split("/")))


def main():
    clips_p = CACHE / "p230_k1_clips.json"
    sents_p = CACHE / "p231_k1_sentences.json"
    mani_p = CACHE / "p232_k1_manifest.json"
    missing = [str(p.name) for p in (clips_p, sents_p, mani_p) if not p.exists()]
    if missing:
        print("缺少产物，跳过审计:", missing, flush=True)
        (M / (TAG + "-k1-audit.json")).write_text(
            json.dumps({"status": "SKIPPED", "missing": missing},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        return

    clips = json.loads(clips_p.read_text(encoding="utf-8"))["clips"]
    sd = json.loads(sents_p.read_text(encoding="utf-8"))
    sents = sd["sentences"]
    mani = json.loads(mani_p.read_text(encoding="utf-8"))

    labels = {name: P78.read_csv_split(name)
              for name in ("train.csv", "dev.csv", "test.csv")}
    freq = Counter()
    for _sid, (_p, seq) in labels["train.csv"].items():
        for t in DPM.PreWords(str(seq).split("/")):
            freq[t] += 1

    # ① 生成句统计
    lens = [len(s) for s in sents]
    low_per_sent = [sum(1 for t in s if freq.get(t, 0) <= 10) for s in sents]
    gen_tokens = Counter(t for s in sents for t in s)
    low_gloss_in_bank = set(t for t in clips if freq.get(t, 0) <= 10)
    low_used = set(t for t in gen_tokens if freq.get(t, 0) <= 10)

    # ② 分布对比
    train_tokens = Counter(t for _sid, (_p, seq) in labels["train.csv"].items()
                           for t in DPM.PreWords(str(seq).split("/")))
    gen_dist = {t: c / max(sum(gen_tokens.values()), 1) for t, c in gen_tokens.items()}
    tr_dist = {t: c / max(sum(train_tokens.values()), 1) for t, c in train_tokens.items()}
    top20_gen = set(t for t, _ in gen_tokens.most_common(20))
    top20_tr = set(t for t, _ in train_tokens.most_common(20))
    jacc = len(top20_gen & top20_tr) / max(len(top20_gen | top20_tr), 1)

    # ③ 精确重复（须为 0）
    existing = set(canon(g) for name in labels
                   for _p, g in labels[name].values())
    dup = sum(1 for s in sents if "/".join(s) in existing)

    # ④ 片段库
    n_per_gloss = [len(v) for v in clips.values()]
    clip_frames = [c[2] - c[1] for v in clips.values() for c in v]
    n_gloss_covered = len(clips)

    # ⑤ manifest
    used_gloss = Counter(t for _sid, (_tr, seq) in mani["manifest"].items()
                         for t in str(seq).split("/"))

    out = {
        "tag": TAG, "status": "OK",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sentences": {
            "n": len(sents), "target": sd.get("target"), "tries": sd.get("tries"),
            "length": pct(lens), "low_freq_per_sentence": pct(low_per_sent),
            "n_distinct_gloss": len(gen_tokens),
            "low_freq_gloss_in_clip_bank": len(low_gloss_in_bank),
            "low_freq_gloss_used": len(low_used),
            "low_freq_coverage": round(len(low_used) / max(len(low_gloss_in_bank), 1), 4),
            "sample_sentences": ["/".join(s) for s in sents[:5]],
        },
        "distribution_vs_train": {
            "js_divergence": js(gen_dist, tr_dist),
            "top20_jaccard": round(jacc, 4),
            "top20_gen": [t for t, _ in gen_tokens.most_common(20)],
            "top20_train": [t for t, _ in train_tokens.most_common(20)],
        },
        "leakage": {"exact_duplicates_train_dev_test": dup},
        "clip_bank": {
            "n_gloss_covered": n_gloss_covered,
            "n_clips_total": sum(n_per_gloss),
            "clips_per_gloss": pct(n_per_gloss),
            "clip_length_input_frames": pct(clip_frames),
        },
        "manifest": {
            "n_samples": mani.get("n"), "n_missing_clip": mani.get("n_missing_clip"),
            "n_frames_linked": mani.get("n_frames_linked"), "seed": mani.get("seed"),
            "n_gloss_used": len(used_gloss),
            "max_reuse_of_one_gloss": max(used_gloss.values()) if used_gloss else 0,
        },
    }
    (M / (TAG + "-k1-audit.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("sentences", "distribution_vs_train",
                                          "leakage", "clip_bank", "manifest")},
                     ensure_ascii=False, indent=2)[:2200], flush=True)

    s = out["sentences"]; d = out["distribution_vs_train"]
    cb = out["clip_bank"]; mf = out["manifest"]
    txt = """# P235 / N2-L2 K1 语料质量与对齐边界审计 — 阶段报告（探索性诊断）

> 日期：%s｜性质：**只给 K1 的 verdict 提供机制解释，不判 PASS/FAIL、不改任何判据**。
> 预注册 `N2-PREREG-2026-11-08.md` §3（20:55 写死）。纯 CPU、不占 GPU、不跑 test。

## 1. 生成句（%d 条，目标 %s，尝试 %s）
- 长度 min/p25/中位/p75/max = %s / %s / **%s** / %s / %s；
  每条低频词数中位 **%s**。
- 不同 gloss 数 **%d**；片段库中低频 gloss **%d** 个，生成句用到 **%d** 个
  ⇒ 低频覆盖 **%.2f%%**。
- 与 train/dev/test 的**精确重复 = %d**（红线须为 0）。
- 样例（统计 bigram 的质量局限）：
%s

## 2. 分布对比（生成句 vs train 原句）
- gloss 频次分布 **JS 散度 = %.6f**；top-20 高频 gloss 的 Jaccard = **%.4f**。
- 生成句 top-20：%s
- train 原句 top-20：%s
- 低频优先采样会使分布右移（JS 偏大是预期的）；这一项只描述，不用于判定 K1 好坏。

## 3. 片段库（CTC Viterbi 强制对齐）
- 覆盖 gloss **%d** 个、片段总数 **%d**；每 gloss 片段数中位 **%s**（p90 **%s**）。
- 片段长度（输入帧）min/p25/中位/p75/p90/max =
  %s / %s / **%s** / %s / %s / %s。
- 与 J1（dev 上区间中位 31.5 输入帧、99.26%% gloss ≥8 帧）交叉核对：
  train 侧中位 **%s** 帧，%s。

## 4. 合成 manifest
- 实际样本 **%s** 条（缺失 %s）、硬链接帧 **%s**、seed %s；
  用到 gloss %d 个、单个 gloss 最大复用 %d 次。
- 全部片段只来自 train、只进 train；dev/test 目录零改动。

## 5. 边界（诚实）
- 本审计**不能**回答「语料合成是否有效」，只能回答「合成语料长什么样」。
- 生成器是统计 bigram 而非文献 LLM（K1-PREREG 已标【无】）；若 K1 判负，
  无法区分「机制无效」与「生成质量不足」，本审计的分布/覆盖数字仅作机制线索。
- 未跑 test、未改任何权重与判据。
""" % (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        s["n"], s["target"], s["tries"],
        s["length"]["min"], s["length"]["p25"], s["length"]["median"],
        s["length"]["p75"], s["length"]["max"],
        s["low_freq_per_sentence"]["median"],
        s["n_distinct_gloss"], s["low_freq_gloss_in_clip_bank"],
        s["low_freq_gloss_used"], 100 * s["low_freq_coverage"],
        out["leakage"]["exact_duplicates_train_dev_test"],
        "\n".join("  - `%s`" % x for x in s["sample_sentences"]),
        d["js_divergence"], d["top20_jaccard"],
        "/".join(d["top20_gen"][:10]) + " …",
        "/".join(d["top20_train"][:10]) + " …",
        cb["n_gloss_covered"], cb["n_clips_total"],
        cb["clips_per_gloss"]["median"], cb["clips_per_gloss"]["p90"],
        cb["clip_length_input_frames"]["min"], cb["clip_length_input_frames"]["p25"],
        cb["clip_length_input_frames"]["median"], cb["clip_length_input_frames"]["p75"],
        cb["clip_length_input_frames"]["p90"], cb["clip_length_input_frames"]["max"],
        cb["clip_length_input_frames"]["median"],
        "与 J1 口径一致" if 24 <= cb["clip_length_input_frames"]["median"] <= 40
        else "与 J1 有偏差，需在汇总里注明",
        mf["n_samples"], mf["n_missing_clip"], mf["n_frames_linked"], mf["seed"],
        mf["n_gloss_used"], mf["max_reuse_of_one_gloss"],
    )
    for base in (REPO, DESK):
        p = base / "docs/experiments/stage-P235-N2-K1-corpus-audit.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> stage-P235-N2-K1-corpus-audit.md (+桌面镜像)", flush=True)


if __name__ == "__main__":
    main()
