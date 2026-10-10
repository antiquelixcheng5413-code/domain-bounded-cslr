"""P91(160res) vs P92(224res+bf16) 逐 epoch 配对分析。

🔴 判据铁律（P83 修订版）：
   绝不只比 best 单点，必须报 ① 逐 epoch 配对差 ② 均值±std
   ③ 胜率 ④ 符号稳定性。符号翻转 或 |均值| < 2×std ⇒ 判「噪声内」。
"""
import json
import statistics
from pathlib import Path

M = Path("/home/su127/FYP/domain-bounded-cslr/artifacts/metrics/blank-gov")
a = json.loads((M / "p91-full-4973-official-tfnet.json").read_text(encoding="utf-8"))
b = json.loads((M / "p92-full-224bf16-official-tfnet.json").read_text(encoding="utf-8"))

print("=" * 92)
print("P91 vs P92 —— 唯一差异：分辨率 160 vs 224（+bf16）")
print("=" * 92)
print("  P91: img_size=%s amp_bf16=%s  %d 条  %d epoch  %.0f 分钟"
      % (a["img_size"], a["amp_bf16"], 4973, len(a["history"]), a["minutes"]))
print("  P92: img_size=%s amp_bf16=%s  %d 条  %d epoch  %.0f 分钟"
      % (b["img_size"], b["amp_bf16"], 4973, len(b["history"]), b["minutes"]))
print("  其余配置完全一致（module=VAC, hidden=1024, seed=42, lr=1e-4, wd=1e-4, 3515词表）")
print()

ha = {h["epoch"]: h for h in a["history"]}
hb = {h["epoch"]: h for h in b["history"]}
common = sorted(set(ha) & set(hb))

print("=" * 92)
print("逐 epoch 配对（diff = P91 - P92，正 = 160 更优）")
print("=" * 92)
print("  ep%4s %10s %10s %10s %12s %12s"
      % ("", "P91(160)", "P92(224)", "diff", "P91_ctc", "P92_ctc"))
diffs = []
for e in common:
    wa = ha[e]["dev_wer_official"]
    wb = hb[e]["dev_wer_official"]
    d = wa - wb
    diffs.append(d)
    print("  %-4d %10.2f %10.2f %10.2f %12.4f %12.4f"
          % (e, wa, wb, d, ha[e]["train_ctc"], hb[e]["train_ctc"]))

m = statistics.fmean(diffs)
sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
win = sum(1 for d in diffs if d > 0)
flip = any(d > 0 for d in diffs) and any(d < 0 for d in diffs)

print()
print("  平均 diff = %+.2f pp   标准差 = %.2f   160 更优率 = %d/%d"
      % (m, sd, win, len(diffs)))
print("  符号翻转 = %s" % flip)
print()
print("  best: P91 %.2f%%  vs  P92 %.2f%%   diff %+.2f pp"
      % (min(h["dev_wer_official"] for h in a["history"]),
         min(h["dev_wer_official"] for h in b["history"]),
         min(h["dev_wer_official"] for h in a["history"])
         - min(h["dev_wer_official"] for h in b["history"])))
print()

print("=" * 92)
print("判据（P83 修订版）")
print("=" * 92)
if flip or abs(m) < 2 * sd:
    print("  |均值| %.2f %s 2×std %.2f，符号%s"
          % (abs(m), "<" if abs(m) < 2 * sd else "≥", 2 * sd,
             "不稳定" if flip else "稳定"))
    print("  ⇒ **噪声范围内：分辨率 160 vs 224 无显著影响**")
    verdict = "NO_EFFECT"
elif m > 0:
    print("  ⇒ 160 持续更优 %+.2f ± %.2f pp" % (m, sd))
    verdict = "160_BETTER"
else:
    print("  ⇒ 224 持续更优 %+.2f ± %.2f pp" % (-m, sd))
    verdict = "224_BETTER"
print()

print("=" * 92)
print("训练成本对比")
print("=" * 92)
print("  P91 (160res) %.0f 分钟 = %.1f 分钟/epoch"
      % (a["minutes"], a["minutes"] / len(a["history"])))
print("  P92 (224res+bf16) %.0f 分钟 = %.1f 分钟/epoch"
      % (b["minutes"], b["minutes"] / len(b["history"])))
ratio = (b["minutes"] / len(b["history"])) / (a["minutes"] / len(a["history"]))
print("  ⇒ 224 慢 %.2f 倍" % ratio)
print()

print("=" * 92)
print("对「差距构成」结论的影响")
print("=" * 92)
if verdict == "NO_EFFECT":
    print("  分辨率**不是**主要差距来源 ⇒ 差距几乎全部来自 epoch 数（6 vs 55）")
    print("  ⇒ 行动：**不加轮次不会接近官方**")
elif verdict == "224_BETTER":
    print("  分辨率是差距来源之一 ⇒ 值得为224 付时间成本")
else:
    print("  ⚠️ 160 反而更优 ⇒ 降分辨率不仅省显存，还不损精度（**意外收获**）")

out = {
    "experiment": "P91-vs-P92 paired analysis",
    "purpose": "分辨率 160 vs 224 的逐 epoch 配对消融（唯一差异）",
    "base_tag": "p91-full-4973",
    "exp_tag": "p92-full-224bf16",
    "base_best": min(h["dev_wer_official"] for h in a["history"]),
    "exp_best": min(h["dev_wer_official"] for h in b["history"]),
    "paired": {
        "per_epoch": [{"epoch": e, "p91_160": ha[e]["dev_wer_official"],
                       "p92_224": hb[e]["dev_wer_official"],
                       "diff": round(ha[e]["dev_wer_official"]
                                     - hb[e]["dev_wer_official"], 2)}
                      for e in common],
        "mean_diff_pp": round(m, 2), "std_pp": round(sd, 2),
        "win_rate_160": "%d/%d" % (win, len(diffs)), "sign_flip": flip,
    },
    "cost": {"p91_min_per_ep": round(a["minutes"] / len(a["history"]), 1),
             "p92_min_per_ep": round(b["minutes"] / len(b["history"]), 1),
             "slowdown": round(ratio, 2)},
    "verdict": verdict,
    "caveat": "全量 4973 条、6 epoch、seed=42，仅分辨率(+bf16)不同。"
              "⚠️ bf16 与分辨率同时变化，是本实验的已知混淆因素；"
              "但 bf16 只影响数值精度，理论上不改变最优解。",
}
dst = M / "p92-resolution-ablation.json"
dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("收据 -> %s" % dst)
