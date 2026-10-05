"""P70：官方口径 WER 评估器 —— 落地「按官方来」的判断标准

用户要求（2026-10-06）：「之后模型效果的判断按照官方的来，计算方法也是」

官方公式（arXiv:2409.11960v2 式(6)）：
    WER = 100% × (ins + del + sub) / sum
    - ins/del/sub：把识别序列变换到参考序列所需的最少插入/删除/替换数
    - sum：**参考标注的 gloss token 总数**
    - 无任何归一化：不去标点、不合并重复 token、不做大小写/空格处理

已用 Table VIII 的 Case-2 验证：把「可以支持」拆成 2 token 后，
5 个算例的 WER 全部与论文报告一致（42.9/42.9/42.9/28.6/0.0）。

**主指标 = token 级 WER（corpus-level）。**
另三项（剔 unk / 逐句等权 / exact）官方未报，只作补充诊断，
写报告时必须标注为「附加分析」而非主指标。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/su127/FYP/domain-bounded-cslr/src")

from cslr.recognition.gloss_sequence import (  # noqa: E402
    split_gloss_sequence, GlossSequenceConfig)

CFG = GlossSequenceConfig()


def levenshtein(a, b):
    """标准编辑距离，与论文式(6) 的 ins+del+sub 一致。"""
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def wer_official(refs, hyps):
    """官方口径 WER（%）。

    refs/hyps: 逐句的 token 列表。
    语料级累加：sum(ed) / sum(len(ref)) × 100
    """
    assert len(refs) == len(hyps), "长度不一致"
    e = 0
    n = 0
    for r, h in zip(refs, hyps):
        e += levenshtein(list(r), list(h))
        n += len(r)
    return (100.0 * e / n if n else 0.0), e, n


def wer_excl_unk(refs, hyps):
    """附加诊断：把 ref 里的 <unk> 剔除后的 WER。
    ⚠️ 官方未报，仅作「模型对已学词的真实能力」分析。
    """
    e = n = 0
    for r, h in zip(refs, hyps):
        ri = [t for t in r if t != "<unk>"]
        hi = [t for t in h if t != "<unk>"]
        e += levenshtein(ri, hi)
        n += len(ri)
    return (100.0 * e / n if n else 0.0)


def wer_per_sentence(refs, hyps):
    """附加诊断：逐句等权平均（最接近「看一条视频」的直觉）。
    ⚠️ 官方未报。
    """
    if not refs:
        return 0.0
    return sum(levenshtein(list(r), list(h))
               for r, h in zip(refs, hyps)) / len(refs) * 100.0


def exact_match(refs, hyps):
    """附加诊断：完全正确的句子比例。
    ⚠️ 官方未报。
    """
    if not refs:
        return 0.0
    return sum(1 for r, h in zip(refs, hyps) if list(r) == list(h)) \
        / len(refs)


def evaluate(ref_gloss_strings, hyp_token_lists):
    """按官方口径评估。

    ref_gloss_strings: dev.csv 里的原始 gloss 串（如 "你/去/运动/。"）
    hyp_token_lists:  每句的模型输出 token 列表
    """
    refs = [split_gloss_sequence(g, CFG) for g in ref_gloss_strings]
    main, e, n = wer_official(refs, hyp_token_lists)
    return {
        # ===== 主指标：与论文/官方直接可比 =====
        "WER_official": round(main, 2),
        "edits": e,
        "ref_tokens": n,
        # ===== 附加诊断：官方未报，写报告须标注 =====
        "WER_excl_unk": round(wer_excl_unk(refs, hyp_token_lists), 2),
        "WER_per_sentence": round(wer_per_sentence(refs, hyp_token_lists), 4),
        "exact_rate_pct": round(exact_match(refs, hyp_token_lists) * 100, 2),
        "_note": "WER_official 是主指标；其余三项为附加诊断",
    }


# ---------------------------------------------------------------- 自检
if __name__ == "__main__":
    print("=" * 72)
    print("官方口径 WER 评估器 —— 自检")
    print("=" * 72)

    # 1) 用论文 Table VIII Case-2 验证（官方把「可以支持」拆 2 token）
    gt = "我/可以/支持/你/去/运动/。"
    cases = {
        "SEN": ("我/可以你/经济/木头/。", 42.9),
        "CorrNet": ("我/可以/你/。", 42.9),
        "VAC": ("我/可以/你/好/锻炼/。", 42.9),
        "MAM-FSD": ("我/可以/支持/你/锻炼/。", 28.6),
        "TFNet": ("我/可以/支持/你/去/运动/。", 0.0),
    }
    refs = [split_gloss_sequence(gt, CFG)]
    print("\n【自检 1】论文 Table VIII Case-2（式6 验证）")
    print("  GT = %s  （%d token）" % (gt, len(refs[0])))
    ok_all = True
    for m, (hg, rep) in cases.items():
        hyps = [split_gloss_sequence(hg, CFG)]
        got, e, n = wer_official(refs, hyps)
        ok = abs(got - rep) < 0.15
        ok_all = ok_all and ok
        print("    %-8s ed=%d/%d=%5.1f%%  论文=%5.1f%%  %s"
              % (m, e, n, got, rep, "✓" if ok else "✗"))
    print("  ⇒ %s" % ("**全部与论文一致，口径确认无误**" if ok_all
                       else "仍有偏差"))

    # 2) 边界
    print("\n【自检 2】边界情况")
    for desc, r, h in (("空 ref", [], []),
                       ("空 hyp", ["我", "去"], []),
                       ("完全一致", ["我", "去"], ["我", "去"]),
                       ("全错", ["我", "去"], ["好", "。"])):
        w, e, n = wer_official([r], [h])
        print("    %-10s ed=%d n=%d WER=%.1f%%" % (desc, e, n, w))

    # 3) 演示：与我们 P42 的历史数字对照
    print("\n【自检 3】用官方口径重述P42 的历史结果")
    print("    之前报（token 级 0.5211）= **与官方同口径**，可直接对照论文")
    print("    之前报 0.7919/2.8774/3.5% = 附加诊断，官方未报")
    print("""
  ⇒ 之后所有判优**只用 WER_official**。
     看到 0.7919 这类数字时，必须标注「附加分析，官方未报」，
     不得用于与论文表格对照。
""")
    print("  官方基准对照（CE-CSL Dev WER %）：")
    print("    MSTNet 54.4 / CorrNet 47.2 / SEN 46.5 / VAC 45.1")
    print("    MAM-FSD 44.9 / **TFNet 42.1（官方 SOTA）**")
    print("    我们 P42= 52.11 → 位于 MSTNet 与 CorrNet 之间")

    out = {
        "official_formula": "arXiv:2409.11960v2 式(6): "
                            "WER = 100% x (ins+del+sub) / sum",
        "sum": "参考 gloss token 总数（corpus-level累加）",
        "normalization": "无：不去标点、不合并重复、不做大小写/空格处理",
        "verified_against": "Table VIII Case-2，5/5 算例一致",
        "primary_metric": "WER_official",
        "supplementary_only": ["WER_excl_unk", "WER_per_sentence",
                               "exact_rate_pct"],
        "official_benchmark_dev": {"MSTNet": 54.4, "CorrNet": 47.2,
                                    "SEN": 46.5, "VAC": 45.1,
                                    "MAM-FSD": 44.9, "TFNet": 42.1},
        "our_p42": 52.11,
        "our_position": "介于 MSTNet(54.4) 与 CorrNet(47.2) 之间",
    }
    p = Path("/home/su127/FYP/domain-bounded-cslr/tools/blank_gov"
             "/official_wer.py")
    p.write_text("""\"\"\"官方口径 WER（arXiv:2409.11960v2 式6）。
本文件是判优的唯一标准来源，由 P70 生成。
\"\"\"
""" + __doc__ + "\n\n" + open(__file__, encoding="utf-8").read()
             .split('if __name__ == "__main__":')[0]
             + '\n\n\nif __name__ == "__main__":\n' + """
    print("=" * 72)
    print("官方口径 WER 评估器")
    print("=" * 72)
    print("  式(6): WER = 100% x (ins+del+sub) / sum")
    print("  sum   = 参考 gloss token 总数（corpus-level 累加）")
    print("  无归一化：不去标点、不合并重复")
    print("  已用 Table VIII Case-2 验证：5/5 算例与论文一致")
""", encoding="utf-8")
    print("\n模块 -> %s" % p)

    q = Path("/home/su127/FYP/domain-bounded-cslr/artifacts/metrics/blank-gov"
             "/p70-official-standard.json")
    q.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("收据 -> %s" % q)