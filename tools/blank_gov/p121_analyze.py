"""P121 结果分析 —— 配对统计（不是比 best WER）。

═══════════════════════════════════════════════════════════════
🔴 必须先说清的设计缺陷（2026-10-11 发现）
═══════════════════════════════════════════════════════════════
P121 的启动命令带了 `--amp-bf16`，但三个对照组（P93 / P100 / P101b-CTRL）
全部是 **fp32**（已核对其收据 `config.amp_bf16 = false`）。

且 `--warmup-ep 3` 在 `word_head=0.0` 时**完全惰性**：
  p78 脚本 line 795: `if a.word_head > 0 and ep > a.warmup_ep:`  ← 只门控词级头

⇒ **P121 原定的「拆 warmup 净效应」目标落空**（warmup 没有作用对象）。

重新定性后，P121 实际能回答的只有一问：

| 对比 | 变量 | 是否有效 |
|---|---|---|
| P121 vs P93 | **仅 bf16**（其余全同：word_head 0、no_hflip、20 轮、160res、seed42） | ✅ 有效 |
| P121 vs CTRL | word_head 0→0.3 **且** warmup 0→3 **且** fp32→bf16 | ❌ 三方混淆，不可用于归因 |
| CTRL vs P93 | word_head+warmup 组合（fp32 同口径） | ✅ 有效（即 P101 的 −2.240） |
| CTRL vs P100 | 同配置重复 | ✅ 噪声复核 |

═══════════════════════════════════════════════════════════════
判据（启动前写死，与 P108 一致）
═══════════════════════════════════════════════════════════════
噪声基准 std = **1.556 pp**
- epoch 级配对 |mean| < 1.556 ⇒ 不可判
- 1.556 ~ 3.1 ⇒ 强提示
- > 3.1 ⇒ 可判
必须同时报 mean±std + 胜率 + SNR；**禁止只比 best WER**。
⚠️ std 一律用 **ddof=1**（样本标准差），并同时打印 ddof=0 供交叉核对。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
MET = REPO / "artifacts/metrics/blank-gov"
NOISE_STD = 1.556

TAGS = ["p121-disamb", "p93-20ep", "p101b-ctrl", "p100-whalign"]


def load(tag):
    p = MET / ("%s-official-tfnet.json" % tag)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    hist = {h["epoch"]: h["dev_wer_official"] for h in d.get("history", [])}
    c = d.get("config", {})
    return {"tag": tag, "hist": hist, "best": d.get("best_wer_official"),
            "cfg": {k: c.get(k) for k in
                    ["epochs", "total_epochs", "word_head", "warmup_ep",
                     "no_hflip", "amp_bf16", "img_size", "seed"]}}


def paired(a, b, lo, hi):
    """a - b 的 epoch 级配对差（负 = a 更好）。"""
    ds = []
    for e in range(lo, hi + 1):
        if e in a["hist"] and e in b["hist"]:
            ds.append(a["hist"][e] - b["hist"][e])
    if not ds:
        return None
    n = len(ds)
    mean = sum(ds) / n
    var1 = sum((x - mean) ** 2 for x in ds) / (n - 1) if n > 1 else 0.0
    var0 = sum((x - mean) ** 2 for x in ds) / n if n else 0.0
    wins = sum(1 for x in ds if x < 0)
    return {"n": n, "mean": mean, "std_ddof1": var1 ** 0.5,
            "std_ddof0": var0 ** 0.5, "wins": wins,
            "snr": abs(mean) / NOISE_STD,
            "deltas": [round(x, 2) for x in ds]}


def verdict(s):
    if s is None:
        return "无数据"
    m = abs(s["mean"])
    if m > 3.1:
        return "可判"
    if m >= NOISE_STD:
        return "强提示"
    return "噪声内（不可判）"


def show(name, s, note=""):
    if s is None:
        print("  %-28s 无数据" % name)
        return None
    print("  %-28s mean %+.3f  std(ddof1) %.3f [ddof0 %.3f]  胜 %d/%d  SNR %.2f  ⇒ %s"
          % (name, s["mean"], s["std_ddof1"], s["std_ddof0"], s["wins"],
             s["n"], s["snr"], verdict(s)))
    if note:
        print("      ↳ %s" % note)
    return verdict(s)


def main():
    data = {t: load(t) for t in TAGS}
    print("=" * 78)
    print("P121 配对分析（%s）" % time.strftime("%Y-%m-%d %H:%M"))
    print("=" * 78)
    for t in TAGS:
        d = data[t]
        if d is None:
            print("  %-14s 收据缺失" % t)
            continue
        print("  %-14s best %.2f%%  epochs %d  cfg=%s"
              % (t, d["best"], len(d["hist"]), d["cfg"]))

    if data["p121-disamb"] is None:
        print("\n⚠️ P121 收据还没出，等它跑完再执行本脚本")
        return 1

    print("\n── 唯一有效的对比：bf16 净效应（P121 vs P93，其余变量全同）──")
    r_full = paired(data["p121-disamb"], data["p93-20ep"], 1, 20)
    r_late = paired(data["p121-disamb"], data["p93-20ep"], 13, 20)
    v_full = show("ep01-20", r_full)
    v_late = show("ep13-20（lr 衰减后）", r_late,
                  "P101 的口径：衰减后区间更稳定")

    print("\n── 三方混淆，不可用于归因（仅供参考）──")
    r_conf = paired(data["p121-disamb"], data["p101b-ctrl"], 13, 20)
    show("P121 vs CTRL ep13-20", r_conf,
         "同时变了 word_head 0→0.3、warmup 0→3、fp32→bf16")

    print("\n── 附带复核（fp32 同口径）──")
    r_cw = paired(data["p101b-ctrl"], data["p93-20ep"], 13, 20)
    show("CTRL vs P93 ep13-20", r_cw, "= word_head + warmup 的**组合**效应")
    r_cc = paired(data["p101b-ctrl"], data["p100-whalign"], 13, 20)
    show("CTRL vs P100 ep13-20", r_cc, "同配置重复 ⇒ 噪声复核")

    out = {
        "experiment": "P121",
        "date": time.strftime("%Y-%m-%d"),
        "design_flaw": ("P121 带了 --amp-bf16，但 P93/P100/CTRL 全是 fp32；"
                        "且 --warmup-ep 3 在 word_head=0 时惰性（p78 line 795）"
                        "⇒ 原定「拆 warmup 净效应」落空，实际只能测 bf16"),
        "noise_ref_pp": NOISE_STD,
        "std_convention": "ddof=1（同时记录 ddof=0）",
        "runs": {t: {"best": (data[t] or {}).get("best"),
                     "cfg": (data[t] or {}).get("cfg")} for t in TAGS},
        "bf16_effect_ep01_20": r_full,
        "bf16_effect_ep13_20": r_late,
        "bf16_verdict": {"ep01_20": v_full, "ep13_20": v_late},
        "confounded_p121_vs_ctrl": r_conf,
        "ctrl_vs_p93_combined": r_cw,
        "ctrl_vs_p100_repeat": r_cc,
        "implication": ("若 bf16 判「无差异/噪声内」⇒ 后续可用 bf16 换速度；"
                        "若 bf16 显著变好 ⇒ 值得重跑关键配置；"
                        "若显著变差 ⇒ P123(fp32) 的选择是对的"),
    }
    dst = MET / "p121-analysis.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("\n收据 -> %s" % dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
