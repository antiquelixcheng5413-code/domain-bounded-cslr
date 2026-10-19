#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P162：P150（VAC+160+RGB） vs P123（VAC+160+BGR）——通道修正到底值多少 pp。

铁律：禁止只比 best WER；必须 mean ± std + 胜率 + SNR；且**只能用 ep32 以后**
     （P123 的 ep1-31 是坏解码截断的虚高值，ep31→ep32 有 −15.70pp 跳变）。
铁律：读收据前先核对字段名（本脚本先 dump 字段再取值，避免 KeyError）。
"""
import json
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
NOISE = 1.556
MIN_EP = 32
FA = "p150-rgbfix-official-tfnet.json"
FB = "p123-official55bf16-official-tfnet.json"


def load(name):
    return json.loads((REPO / "artifacts/metrics/blank-gov" / name).read_text(encoding="utf-8"))


a, b = load(FA), load(FB)

print("=" * 66)
print("字段核对（先看清再取值）")
print("=" * 66)
for tag, d in (("P150", a), ("P123", b)):
    print("\n[%s] keys:" % tag)
    print("   ", list(d.keys()))
    for k in sorted(d.keys()):
        if any(s in k for s in ("best", "epoch", "wer")):
            v = d[k]
            if isinstance(v, list):
                v = "list(%d)" % len(v)
            print("    %-22s = %s" % (k, v))


def best_of(d):
    for k in ("best_wer_official", "best_wer", "bestWER"):
        if k in d:
            return d[k]
    return None


def best_ep_of(d):
    for k in ("best_epoch", "best_ep", "epoch_best"):
        if k in d:
            return d[k]
    return "?"


def hist(d):
    return {h["epoch"]: h for h in d.get("history", [])}


ha, hb = hist(a), hist(b)
ba, bb = best_of(a), best_of(b)

print()
print("=" * 66)
print("各自最佳")
print("=" * 66)
print("  P150 (RGB) : best %s%%  @ ep%s   轮数 %d" % (ba, best_ep_of(a), len(ha)))
print("  P123 (BGR) : best %s%%  @ ep%s   轮数 %d" % (bb, best_ep_of(b), len(hb)))
if ba is not None and bb is not None:
    print("  ⇒ best 之差 : %+.2f pp" % (ba - bb))

print()
print("配置一致性核对（必须同架构/同分辨率）")
for k in ("module", "img_size", "amp_bf16", "official_pad", "word_head_w",
          "lr_milestones", "lr_gamma", "vocab"):
    va, vb = a.get(k), b.get(k)
    mark = "OK " if va == vb else "!! "
    print("  %s %-14s P150=%-12s P123=%s" % (mark, k, va, vb))

# ---- 逐轮配对（ep32+）----
eps = sorted(set(ha) & set(hb))
eps = [e for e in eps if e >= MIN_EP]
print()
print("=" * 66)
print("逐轮配对（仅 ep%d 以后，共 %d 轮可比）" % (MIN_EP, len(eps)))
print("=" * 66)
print("  %-6s %11s %11s %11s" % ("epoch", "P150(RGB)", "P123(BGR)", "delta"))
ds = []
for e in eps:
    wa = ha[e]["dev_wer_official"]
    wb = hb[e]["dev_wer_official"]
    ds.append(wa - wb)
    print("  ep%-3d %11.2f %11.2f %+11.2f" % (e, wa, wb, wa - wb))

if ds:
    n = len(ds)
    mean = sum(ds) / n
    var = sum((x - mean) ** 2 for x in ds) / (n - 1) if n > 1 else 0.0
    std = var ** 0.5
    wins = sum(1 for x in ds if x < 0)
    snr = abs(mean) / NOISE
    print()
    print("=" * 66)
    print("  配对均值 delta = %+.3f pp   （负 = RGB 更好）" % mean)
    print("  std            = %.3f pp" % std)
    print("  RGB 胜率       = %d/%d = %.0f%%" % (wins, n, 100.0 * wins / n))
    print("  SNR            = |%.3f| / %.3f = %.2f" % (mean, NOISE, snr))
    if snr < 1:
        v = "不可判 (<1)"
    elif snr < 2:
        v = "强提示 (1~2) —— 方向可信，未达可判"
    else:
        v = "可判 (>=2)" + ("且胜率>=75%" if wins / n >= 0.75 else "但胜率<75%，谨慎")
    print("  判定           : %s" % v)
    print("=" * 66)

    if ba is not None and bb is not None:
        gain = bb - ba
        total = bb - 33.92
        print()
        print("对 22pp 缺口的意义")
        print("  P150(RGB) %.2f%%  vs 官方权重 33.92%%  ⇒ 仍差 %.2f pp" % (ba, ba - 33.92))
        print("  通道修正拿回 %.2f pp，占总缺口 %.2f pp 的 %.0f%%"
              % (gain, total, 100.0 * gain / total))
        print("  ⇒ 绝大部分缺口**不是**通道造成的。")
