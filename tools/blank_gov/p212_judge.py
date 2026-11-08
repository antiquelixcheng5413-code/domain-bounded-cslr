#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E1 判定：按预注册判据，主看 uWER 与低频桶，防 D4 式重分配。"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
M = REPO / "artifacts/metrics/blank-gov"


def load(tag):
    return json.loads((M / ("%s-uwer.json" % tag)).read_text(encoding="utf-8"))


def curve():
    txt = (REPO / "artifacts/logs_P212_e1_train.log").read_text(
        encoding="utf-8", errors="replace")
    return [(int(a), float(b), float(c)) for a, b, c in re.findall(
        r"ep(\d{3}) ctc\s+([\d.]+)\s+\*\*WER_official\s+([\d.]+)%", txt)]


def main():
    b, c = load("p184-base"), load("p212-e1")
    cv = curve()
    best = min((x[2] for x in cv), default=None)
    d_wer = b["WER_official"] - c["WER_official"]
    d_uwer = b["uWER_pct"] - c["uWER_pct"]
    d_low = b["low_bucket"]["WER_pct"] - c["low_bucket"]["WER_pct"]
    d_high = c["high_bucket"]["WER_pct"] - b["high_bucket"]["WER_pct"]
    # 预注册：uWER >=2pp 且低频桶 >=2pp；高频桶不得恶化 >1pp；
    # 总量退化 <=1.0pp（E1 允许以小幅总量换尾部，但必须显式报出）
    passed = (d_uwer >= 2.0 and d_low >= 2.0 and d_high <= 1.0
              and d_wer >= -1.0 and best is not None and best <= 34.80)
    out = {
        "tag": "p212-e1-csgc",
        "base": b, "cand": c,
        "train_curve": [{"ep": e, "ctc": ct, "wer": w} for e, ct, w in cv],
        "train_best_dev_wer": best,
        "delta": {
            "WER_official": round(d_wer, 2),
            "uWER_pct": round(d_uwer, 2),
            "low_bucket_WER_pct": round(d_low, 2),
            "high_bucket_WER_pct_worsen": round(d_high, 2),
        },
        "pre_registered": {
            "rule": "uWER改善>=2pp 且 低频桶改善>=2pp 且 高频桶恶化<=1pp "
                    "且 WER_official退化<=1.0pp 且 dev<=34.80",
            "note": "主看 uWER/低频桶；仅总量改善不算正向（D4 教训）",
        },
        "verdict": "PASS" if passed else "FAIL",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    p = M / "p212-e1-judge.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("收据 ->", p)


if __name__ == "__main__":
    main()
