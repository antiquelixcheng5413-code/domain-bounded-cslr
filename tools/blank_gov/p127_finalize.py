"""P127 收尾：从 last.pt 重建收据（被止损杀死 ⇒ p78 没写收据）+ 与 P124e 配对。

为什么需要：早停/止损都是外部 SIGTERM，p78 只在正常跑完时写收据
⇒ 与 p123_watch_stop.py 同样的处理方式（MEMORY 21）。

输出：
  artifacts/metrics/blank-gov/p127-res224-official-tfnet.json  （重建收据）
  artifacts/metrics/blank-gov/pair-p127-res224-vs-p124e-wh0-bf16.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
CK = REPO / "artifacts/checkpoints/p127-res224-last.pt"
BASE = "p124e-wh0-bf16"
NOISE = 1.556          # 本项目实测的单次实验 epoch 级噪声 std（MEMORY 4）


def main() -> int:
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    hist = ck["history"]
    h127 = {r["epoch"]: r["dev_wer_official"] for r in hist}
    best_ep = min(h127, key=h127.get)

    base_p = REPO / "artifacts/metrics/blank-gov" / ("%s-official-tfnet.json" % BASE)
    hb = {r["epoch"]: r["dev_wer_official"]
          for r in json.load(open(base_p))["history"]}

    # ---- 重建收据 ----
    receipt = {
        "tag": "p127-res224",
        "config": {k: ck["config"].get(k) for k in
                   ["epochs", "total_epochs", "img_size", "amp_bf16", "no_hflip",
                    "word_head", "warmup_ep", "seed", "official_pad", "batch",
                    "hidden", "lr", "mem_budget"]},
        "history": hist,
        "best_wer_official": h127[best_ep],
        "best_epoch": best_ep,
        "epochs_completed": len(hist),
        "_stopped_early": {
            "by": "manual (中期判定止损)",
            "reason": "判据区间 ep13-15 差距已达 +13.29pp（SNR 8.54），"
                      "且单调发散，继续跑不会改变结论 ⇒ 止损省 5 轮 ×42min",
            "epochs_run": len(hist), "epochs_planned": 20,
            "note": "权重完好（last.pt 423MB）；收据由 p127_finalize.py 重建",
        },
    }
    out = REPO / "artifacts/metrics/blank-gov/p127-res224-official-tfnet.json"
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("[P127] 收据重建 -> %s" % out.name)

    # ---- 配对 ----
    rows = [(e, hb[e], h127[e], h127[e] - hb[e])
            for e in sorted(h127) if e in hb]
    print("\nep   P124e(160)  P127(224)   差(224-160)")
    for e, a, b, d in rows:
        print("%3d   %8.2f  %8.2f   %+7.2f" % (e, a, b, d))

    blocks = {}
    for lo, hi, name in [(1, 15, "ep01-15（全部）"), (13, 15, "ep13-15（判据区间）")]:
        ds = [d for e, _a, _b, d in rows if lo <= e <= hi]
        if not ds:
            continue
        m = st.mean(ds)
        sd = st.stdev(ds) if len(ds) > 1 else 0.0
        win = sum(1 for d in ds if d < 0)          # 224 更低的轮数
        blocks[name] = {
            "n": len(ds), "mean_pp": round(m, 3), "std_pp": round(sd, 3),
            "win_224": win, "snr": round(abs(m) / NOISE, 2),
        }
        print("\n%s: n=%d mean %+.3f std %.3f  224胜 %d/%d  SNR %.2f"
              % (name, len(ds), m, sd, win, len(ds), abs(m) / NOISE))

    main_b = blocks["ep13-15（判据区间）"]
    m, snr, win, n = main_b["mean_pp"], main_b["snr"], main_b["win_224"], main_b["n"]
    if m >= 1.556:
        verdict = ("224 有害：判据区间 mean %+.2f pp（SNR %.2f，224 胜 %d/%d）"
                   "⇒ **160 保留**，分辨率方向关闭" % (m, snr, win, n))
    elif m <= -1.556 and snr >= 2 and win >= 6:
        verdict = "224 有益 ⇒ 切换 224"
    else:
        verdict = "不可判，需继续跑满 20 轮"
    print("\n判定：%s" % verdict)

    pair = {
        "exp": "p127-res224", "base": BASE,
        "variable": "img_size 160 → 224（唯一变量）",
        "noise_std_pp": NOISE,
        "per_epoch": [{"epoch": e, "base_160": a, "exp_224": b, "diff": round(d, 2)}
                      for e, a, b, d in rows],
        "blocks": blocks,
        "verdict": verdict,
        "note": "P127 于 ep15 被止损（差距单调发散，继续跑不改变结论）",
    }
    pout = (REPO / "artifacts/metrics/blank-gov"
            / "pair-p127-res224-vs-p124e-wh0-bf16.json")
    pout.write_text(json.dumps(pair, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("配对收据 -> %s" % pout.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
