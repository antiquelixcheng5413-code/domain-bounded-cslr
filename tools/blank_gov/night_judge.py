#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""夜间链统一判定器（预注册规则，所有实验共用，禁止事后调线）。

规则（写死于 2026-11-07 23:30，实验启动前）：
  主看尾部（D4 挖出的真瓶颈）：
    R1 uWER 改善 >= 2.0pp
    R2 低频桶 WER 改善 >= 2.0pp
  防止拆东墙补西墙 / 假改善：
    R3 高频桶恶化 <= 1.0pp
    R4 WER_official 退化 <= 1.0pp
    R5 插入数相对基线增幅 <= 10%（E0 教训：uWER 看不见插入）
    R6 micro sub+del 恶化 <= 0.30pp（D4 教训）
    R7 训练类：6 轮 best dev <= 34.80；零训练实验跳过 R7
  另报（不参与 PASS）：配对 bootstrap 95% CI、oracle/非blank 帧占比
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
M = REPO / "artifacts/metrics/blank-gov"


def load(tag):
    p = M / ("%s-uwer.json" % tag)
    return json.loads(p.read_text(encoding="utf-8"))


def paired_bootstrap(base, cand, n_boot=20000, seed=42):
    """句子级配对重采样，返回 ΔWER(base-cand) 的点估计与 95% CI。"""
    b = np.asarray([x["edits"] for x in base["per_item"]], dtype=np.float64)
    c = np.asarray([x["edits"] for x in cand["per_item"]], dtype=np.float64)
    n = np.asarray([x["ref_n"] for x in base["per_item"]], dtype=np.float64)
    assert len(b) == len(c) == len(n), "paired sample mismatch"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    db = (b[idx].sum(1) - c[idx].sum(1)) / n[idx].sum(1) * 100.0
    point = 100.0 * (b.sum() - c.sum()) / n.sum()
    return round(float(point), 3), round(float(np.percentile(db, 2.5)), 3), \
        round(float(np.percentile(db, 97.5)), 3)


def curve(log_path):
    p = Path(log_path)
    if not log_path or not p.is_file():
        return []
    txt = p.read_text(encoding="utf-8", errors="replace")
    return [(int(a), float(b), float(c)) for a, b, c in re.findall(
        r"ep(\d{3}) ctc\s+([\d.]+)\s+\*\*WER_official\s+([\d.]+)%", txt)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="p184-base")
    ap.add_argument("--log", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--kind", default="train", choices=["train", "decode"])
    a = ap.parse_args()

    b, c = load(a.base), load(a.tag)
    cv = curve(a.log)
    best = min((x[2] for x in cv), default=None)

    d_wer = round(b["WER_official"] - c["WER_official"], 3)
    d_uwer = round(b["uWER_pct"] - c["uWER_pct"], 3)
    d_low = round(b["low_bucket"]["WER_pct"] - c["low_bucket"]["WER_pct"], 3)
    d_high = round(c["high_bucket"]["WER_pct"] - b["high_bucket"]["WER_pct"], 3)
    d_micro = round(c["edits"]["micro_subdel_pct"] - b["edits"]["micro_subdel_pct"], 3)
    ins_b = b["edits"]["ins"]
    ins_c = c["edits"]["ins"]
    ins_ratio = round((ins_c - ins_b) / max(ins_b, 1), 4)

    point, lo, hi = paired_bootstrap(b, c)

    checks = {
        "R1_uWER_improve_ge_2pp": bool(d_uwer >= 2.0),
        "R2_low_bucket_improve_ge_2pp": bool(d_low >= 2.0),
        "R3_high_bucket_worsen_le_1pp": bool(d_high <= 1.0),
        "R4_WER_official_degrade_le_1pp": bool(d_wer >= -1.0),
        "R5_ins_increase_le_10pct": bool(ins_ratio <= 0.10),
        "R6_micro_worsen_le_0.30pp": bool(d_micro <= 0.30),
    }
    if a.kind == "train":
        checks["R7_train_best_dev_le_34.80"] = bool(best is not None and best <= 34.80)
    # R8：低频桶单独配对 bootstrap（G1 方案额外要求；仅当评估器带 low_n 字段时启用）
    lowfreq_ci = None
    if ("low_n" in b["per_item"][0] and "low_n" in c["per_item"][0]):
        lb = np.asarray([x["low_err"] for x in b["per_item"]], dtype=np.float64)
        lc = np.asarray([x["low_err"] for x in c["per_item"]], dtype=np.float64)
        ln = np.asarray([x["low_n"] for x in b["per_item"]], dtype=np.float64)
        m = ln > 0
        if int(m.sum()) > 10:
            rng = np.random.default_rng(42)
            ii = rng.integers(0, int(m.sum()), size=(20000, int(m.sum())))
            dd = ((lb[m][ii] - lc[m][ii]).sum(1)
                  / ln[m][ii].sum(1).clip(min=1.0)) * 100.0
            lo_l, hi_l = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
            lowfreq_ci = {"delta_pp": round(float(100.0 * (lb[m].sum() - lc[m].sum())
                                                  / max(ln[m].sum(), 1)), 3),
                          "ci95": [round(lo_l, 3), round(hi_l, 3)],
                          "n_sentences": int(m.sum())}
            checks["R8_lowfreq_CI_gt_0"] = bool(lo_l > 0)
    passed = all(checks.values())

    out = {
        "tag": a.tag,
        "title": a.title,
        "kind": a.kind,
        "base_tag": a.base,
        "base": {k: v for k, v in b.items() if k != "per_item"},
        "cand": {k: v for k, v in c.items() if k != "per_item"},
        "train_curve": [{"ep": e, "ctc": ct, "wer": w} for e, ct, w in cv],
        "train_best_dev_wer": best,
        "delta": {
            "WER_official_improve_pp": d_wer,
            "uWER_improve_pp": d_uwer,
            "low_bucket_improve_pp": d_low,
            "high_bucket_worsen_pp": d_high,
            "micro_subdel_worsen_pp": d_micro,
            "ins_relative_increase": ins_ratio,
            "ins_base": ins_b, "ins_cand": ins_c,
        },
        "paired_bootstrap": {"n_boot": 20000, "delta_WER_pp": point,
                             "ci95": [lo, hi], "ci_contains_zero": bool(lo <= 0 <= hi)},
        "lowfreq_bootstrap": lowfreq_ci,
        "checks": checks,
        "pre_registered": {
            "fixed_at": "2026-11-07 23:30 CST (before any of tonight's experiments)",
            "rule": "R1..R6 (+R7 for train) all must hold; tail metrics are primary",
            "note": "不得事后调整；仅总量改善不算正向（D4/E0 教训）",
        },
        "verdict": "PASS" if passed else "FAIL",
    }
    p = M / ("%s-judge.json" % a.tag)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("tag", "delta", "checks", "verdict")},
                     ensure_ascii=False, indent=2), flush=True)
    print("收据 -> %s" % p, flush=True)


if __name__ == "__main__":
    main()
