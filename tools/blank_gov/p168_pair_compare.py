"""P168：P167(词级头消融, --word-head 0) vs P150(词级头 0.3) 逐轮配对分析。

沿用项目判据：
- 只用 ep32+（P123 的 ep1-31 是坏解码截断虚高值；这里两边都是新解码，
  但为与历史口径一致，仍以 ep32+ 为主判据，同时报告 ep1-55 全段作参考）
- 报告 mean / std / 胜率 / 配对 t / SNR
- 噪声基准 1.556pp（保守分母）
"""
import json, math
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
MD = REPO / "artifacts/metrics/blank-gov"

P167 = MD / "p167-vac160-rgb-wh0-official-tfnet.json"
P150 = MD / "p150-rgbfix-official-tfnet.json"

NOISE = 1.556  # 保守噪声基准


def load(p, tag):
    d = json.load(open(p, encoding="utf-8"))
    hist = d["history"]
    curve = {}
    for it in hist:
        e = it.get("epoch")
        w = it.get("dev_wer_official")
        if w is not None:
            curve[int(e)] = float(w)
    # 已知的 key 差异：P150 收据没有 best_epoch 字段，但 P167 也没有
    best = d.get("best_wer_official")
    print("[%s] history=%d  best_wer=%s  word_head_w=%s  img=%s  bf16=%s"
          % (tag, len(hist), best, d.get("word_head_w"), d.get("img_size"), d.get("amp_bf16")))
    return curve, best, d


c167, b167, d167 = load(P167, "P167 wh=0")
c150, b150, d150 = load(P150, "P150 wh=0.3")

# 配置一致性核对（差异只应是 word_head_w）
print("\n=== 配置一致性核对（除 word-head 外应全部相同）===")
keys = ["module", "img_size", "amp_bf16", "word_boundary", "official_pad",
        "warmup_ep", "lr_milestone_ep", "lr_gamma", "mem_budget_GB", "vocab", "n_params_M"]
for k in keys:
    a, b = d167.get(k), d150.get(k)
    flag = "OK " if a == b else "*** DIFF ***"
    print("  %-20s P167=%-10s P150=%-10s %s" % (k, a, b, flag))
print("  %-20s P167=%-10s P150=%-10s %s"
      % ("word_head_w", d167.get("word_head_w"), d150.get("word_head_w"), "*** 这是被操纵的变量 ***"))


def paired(ca, cb, lo, hi, name):
    eps = sorted(set(ca) & set(cb))
    eps = [e for e in eps if lo <= e <= hi]
    if not eps:
        print("\n[%s] 无可比轮次" % name)
        return
    diffs = [ca[e] - cb[e] for e in eps]  # P167 - P150，负 = P167 更好
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    win = sum(1 for x in diffs if x < 0)  # P167 胜
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    snr_cons = abs(mean) / NOISE
    snr_pair = abs(mean) / sd if sd > 0 else float("inf")
    print("\n=== %s（ep%d–ep%d，%d 轮）===" % (name, lo, hi, n))
    print("  配对均值 Δ(P167 − P150) = %+.3f pp   （负 = P167 更好）" % mean)
    print("  std = %.3f   胜率 P167 = %d/%d" % (sd, win, n))
    print("  配对 t ≈ %.2f" % t)
    print("  SNR = %.2f（保守噪声 %.3f） | %.2f（配对 std）" % (snr_cons, NOISE, snr_pair))
    if snr_cons >= 2:
        verdict = "✅ 可判（保守 SNR ≥ 2）"
    elif snr_cons >= 1:
        verdict = "🟡 强提示（1 ≤ SNR < 2）"
    else:
        verdict = "⚪ 不可判（SNR < 1）"
    print("  判定：%s  ⇒ %s" % (verdict, "去掉词级头更好" if mean < 0 else "保留词级头更好"))
    print("\n  逐轮明细：")
    for e in eps:
        print("    ep%02d  P167=%6.2f  P150=%6.2f  Δ=%+6.2f" % (e, ca[e], cb[e], ca[e] - cb[e]))


print("\n############################################################")
print("#  best WER 对比（禁止只比 best，仅作参考）")
print("############################################################")
print("  P167 (wh=0)   best = %.2f" % b167)
print("  P150 (wh=0.3) best = %.2f" % b150)
print("  Δ = %+.2f pp" % (b167 - b150))

paired(c167, c150, 32, 55, "主判据：ep32+（lr 衰减后定型区）")
paired(c167, c150, 1, 55, "参考：全段 ep1–55")
paired(c167, c150, 1, 35, "参考：ep1–35（lr 首次衰减前）")
paired(c167, c150, 36, 55, "参考：ep36+（第二次 lr 衰减后）")
