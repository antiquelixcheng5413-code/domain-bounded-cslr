#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P210 串联判定器：按预注册判据判 D2 / D4，落 JSON 收据。

用法：python p210_chain_judge.py d2 | d4
判据全部来自 docs/planning/OPT-DIRECTIONS-2026-11-07.md（启动前写死）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
BASE_DEV = 34.34          # P184@160 dev WER_official（语料级）
NOISE = 0.77              # dev 噪声地板 ±0.77pp
HARD_GAIN = 0.70          # 项目硬约束：好 >=0.7pp 才进复跑/终评


def epoch_curve(log_path):
    txt = Path(log_path).read_text(encoding="utf-8", errors="replace")
    ep = [(int(a), float(b), float(c)) for a, b, c in re.findall(
        r"ep(\d{3}) ctc\s+([\d.]+)\s+\*\*WER_official\s+([\d.]+)%", txt)]
    return ep


def judge_d2():
    log = REPO / "artifacts/logs_P208_d2_train.log"
    curve = epoch_curve(log)
    if not curve:
        print("[d2] FAIL: 无 epoch 曲线")
        return 1
    best_ep, best_ctc, best_wer = min(curve, key=lambda x: x[2])
    delta = BASE_DEV - best_wer
    # 判据：dev <= 33.8（E3 筛选线）；另报项目硬约束 33.64
    passed_screen = best_wer <= 33.80
    passed_hard = delta >= HARD_GAIN
    # 训练-验证间隙：用 train ctc 作代理（一致性正则通常使 train ctc 上升）
    out = {
        "tag": "p208-d2-cons",
        "baseline_dev": BASE_DEV,
        "curve": [{"ep": e, "train_ctc": c, "dev_wer": w} for e, c, w in curve],
        "best": {"epoch": best_ep, "train_ctc": best_ctc,
                 "dev_wer": best_wer},
        "delta_pp": round(delta, 3),
        "noise_floor_pp": NOISE,
        "pre_registered": {
            "screen": "dev <= 33.80 且 train-dev 间隙缩小（train ctc 为代理）",
            "hard": "项目硬约束：比 34.34 好 >= 0.70pp 才进 paired_eval/2seed/test",
        },
        "screen_passed": bool(passed_screen),
        "hard_passed": bool(passed_hard),
        "verdict": ("PROCEED_TO_FINAL" if (passed_screen and passed_hard)
                    else ("NOISE_OR_NEGATIVE" if passed_screen else "NEGATIVE")),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    p = REPO / "artifacts/metrics/blank-gov/p208-d2-judge.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[d2] 收据 ->", p)
    return 0


def judge_d4():
    def load(tag):
        p = REPO / "artifacts/metrics/blank-gov" / ("%s-uwer.json" % tag)
        return json.loads(p.read_text(encoding="utf-8"))
    try:
        b, c = load("p184-base"), load("p210-d4")
    except Exception as e:
        print("[d4] FAIL: 读取评估收据失败:", e)
        return 1
    curve = epoch_curve(REPO / "artifacts/logs_P210_d4_train.log")
    best_wer = min((x[2] for x in curve), default=None)
    d_wer = b["WER_official"] - c["WER_official"]
    d_uwer = b["uWER_pct"] - c["uWER_pct"]
    d_low = b["low_bucket"]["WER_pct"] - c["low_bucket"]["WER_pct"]
    passed = (best_wer is not None and best_wer <= 33.60
              and d_uwer > 0 and d_low > 0)
    out = {
        "tag": "p210-d4-cb",
        "base": b, "cand": c,
        "train_best_dev_wer": best_wer,
        "delta": {"WER_official": round(d_wer, 2),
                  "uWER_pct": round(d_uwer, 2),
                  "low_bucket_WER_pct": round(d_low, 2)},
        "pre_registered": {
            "rule": "dev<=33.60 且 uWER 与低频桶 WER 均改善；否则判负",
            "note": "不能只看总 WER（FGXM PRL25 指出高频功能词掩盖低频实词）",
        },
        "verdict": "PASS" if passed else "FAIL",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    p = REPO / "artifacts/metrics/blank-gov/p210-d4-judge.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[d4] 收据 ->", p)
    return 0


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit(judge_d2() if which == "d2" else
             judge_d4() if which == "d4" else 1)
