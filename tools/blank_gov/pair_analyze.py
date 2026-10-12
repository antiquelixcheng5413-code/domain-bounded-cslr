"""epoch 级配对分析（通用）—— 禁止只报 best WER（P83/P85/P89 教训）。

用法：
    python pair_analyze.py --a p124d-bf16-wh03 --b p119-hflip --lo 13 --hi 20

输出：mean ± std(ddof=1)、胜率、信噪比、逐 epoch 对照表。
噪声基准 1.556 pp（P101b-ctrl 同配置重复实测）。

判读：
    SNR < 1        ⇒ 不可判（单次实验检测不到）
    1 ≤ SNR < 2    ⇒ 强提示，非已证实
    SNR ≥ 2        ⇒ 可判
⚠️ 还必须看胜率：mean 达标但胜率 5/8 这种不算干净。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
MET = REPO / "artifacts/metrics/blank-gov"
NOISE = 1.556          # pp，同配置重复的 epoch 级 std（P101b-ctrl 实测）


def load(tag: str):
    p = MET / ("%s-official-tfnet.json" % tag)
    if not p.exists():
        raise SystemExit("收据不存在：%s" % p)
    d = json.load(open(p))
    return ({r["epoch"]: r["dev_wer_official"] for r in d["history"]}, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="新配置")
    ap.add_argument("--b", required=True, help="对照配置")
    ap.add_argument("--lo", type=int, default=13)
    ap.add_argument("--hi", type=int, default=20)
    ap.add_argument("--out", default="")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    ha, da = load(args.a)
    hb, db = load(args.b)
    eps = [e for e in range(args.lo, args.hi + 1) if e in ha and e in hb]
    if len(eps) < 2:
        raise SystemExit("可配对轮数不足（%d），无法计算 std" % len(eps))

    diffs = [ha[e] - hb[e] for e in eps]
    n = len(diffs)
    mean = sum(diffs) / n
    std = (sum((x - mean) ** 2 for x in diffs) / (n - 1)) ** 0.5
    win = sum(1 for x in diffs if x < 0)          # A 更好（WER 更低）的轮数
    snr = abs(mean) / NOISE

    if snr < 1.0:
        verdict = "不可判"
    elif snr < 2.0:
        verdict = "强提示"
    else:
        verdict = "可判" if win >= n * 0.75 else "强提示（胜率不足）"

    print("=" * 64)
    print("配对分析：%s  vs  %s   （ep%d–%d，n=%d）"
          % (args.a, args.b, args.lo, args.hi, n))
    print("=" * 64)
    print("%-6s %10s %10s %10s" % ("epoch", args.a[:9], args.b[:9], "A−B"))
    for e in eps:
        print("ep%-4d %10.2f %10.2f %+10.2f" % (e, ha[e], hb[e], ha[e] - hb[e]))
    print("-" * 64)
    print("mean        %+.3f pp" % mean)
    print("std(ddof=1) %.3f pp" % std)
    print("胜率        %d/%d" % (win, n))
    print("信噪比      %.2f  （噪声基准 %.3f pp）" % (snr, NOISE))
    print("判定        %s" % verdict)
    print("-" * 64)
    print("best:  %s %.2f%%   %s %.2f%%"
          % (args.a[:9], min(ha.values()), args.b[:9], min(hb.values())))

    rec = {
        "pair": {"a": args.a, "b": args.b},
        "epoch_range": [args.lo, args.hi],
        "n_pairs": n,
        "mean_pp": round(mean, 4),
        "std_pp": round(std, 4),
        "win": "%d/%d" % (win, n),
        "snr": round(snr, 3),
        "noise_baseline_pp": NOISE,
        "verdict": verdict,
        "best_a": round(min(ha.values()), 2),
        "best_b": round(min(hb.values()), 2),
        "per_epoch": {str(e): [round(ha[e], 2), round(hb[e], 2),
                               round(ha[e] - hb[e], 2)] for e in eps},
        "config_a": da.get("config", {}),
        "config_b": db.get("config", {}),
        "note": args.note,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = args.out or ("pair-%s-vs-%s.json" % (args.a, args.b))
    dst = MET / out
    dst.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print("收据 -> %s" % dst)


if __name__ == "__main__":
    main()
