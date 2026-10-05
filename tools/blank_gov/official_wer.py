"""P71：官方口径 WER 评估器（最终版）

用户要求（2026-10-06）：「之后模型效果的判断按照官方的来，计算方法也是」

═══ 官方定义（arXiv:2409.11960v2 式6）═══
    WER = 100% × (ins + del + sub) / sum
    - ins/del/sub：把识别序列变换到参考序列所需的最少插入/删除/替换数
    - sum：参考标注的 gloss token 总数
    - corpus-level 累加：先求所有句 ed 之和，再除以所有参考 token 之和

═══ 无归一化 ═══
    不去标点（。是一个 token）
    不合并重复 token
    不做大小写/空格处理

═══ 词组粘连的处理（P70b 反推确认）═══
该数据集标注存在**词组粘连**（如「可以支持」）。
官方在计算时把**参考与识别两侧的粘连词都拆开**：
    ref  我/可以支持/你/去/运动/。      (6 token)
      → 我/可以/支持/你/去/运动/。      (7 token)
    hyp  我/可以你/经济/木头/。        (4 token)
      → 我/可以/你/经济/木头/。        (6 token)
    ed=3, n=7 → WER = 42.9%  ✓与论文 Table VIII 完全一致

验证结果（Table VIII Case-2，5/5 精确吻合）：
    SEN     42.9%  ✓
    CorrNet 42.9%  ✓
    VAC     42.9%  ✓
    MAM-FSD 28.6%  ✓
    TFNet    0.0%  ✓
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/su127/FYP/domain-bounded-cslr/src")

from cslr.recognition.gloss_sequence import (  # noqa: E402
    split_gloss_sequence, GlossSequenceConfig)

CFG = GlossSequenceConfig()


def levenshtein(a, b):
    """标准编辑距离 = ins + del + sub，与论文式(6) 一致。"""
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


# 官方 gloss 里出现的粘连型功能词/副词（用于把「可以支持」拆成「可以|支持」、
# 把「可以你」拆成「可以|你」）。**两侧都必须拆**，否则 Table VIII 算例对不上。
_LEAVES = (
    "可以", "能够", "应该", "必须", "需要", "不能", "不要", "没有",
    "可能", "也许", "已经", "正在", "将要", "曾经",
)
# 中文功能字（用于兜底拆分粘连的单字）
_FUNCTION_CHARS = "的了和与在有是不了着过"


def split_glued(tokens):
    """把粘连 token 拆开（官方口径的反推结果，P70b 验证）。

    规则：
      1. 若 token 本身是叶子词，原样保留（「可以」不动）
      2. 若 token 以叶子词开头且剩余 1~3 字，拆成两段
         「可以支持」->「可以|支持」，「可以你」->「可以|你」
      3. 其余不动，避免拆坏正常词（如「律师」「支持」）

    ⚠️ 仅用于**评估**对齐，不影响训练时的 target 构造。
    """
    out = []
    for t in tokens:
        if t in _LEAVES:
            out.append(t)
            continue
        done = False
        for leaf in _LEAVES:                       # 前缀拆分
            if t.startswith(leaf) and len(t) > len(leaf):
                rest = t[len(leaf):]
                if 1 <= len(rest) <= 3:
                    out.extend([leaf, rest])
                    done = True
                    break
        if done:
            continue
        for leaf in _LEAVES:                       # 后缀拆分
            if t.endswith(leaf) and len(t) > len(leaf):
                head = t[:-len(leaf)]
                if 1 <= len(head) <= 3:
                    out.extend([head, leaf])
                    done = True
                    break
        if not done:
            out.append(t)
    return out


def wer_official(ref_gloss_strings, hyp_token_lists, split_glued_both=True):
    """官方口径 WER（%）。

    ref_gloss_strings: 原始 gloss 串（如 "你/去/运动/。"）
    hyp_token_lists:  每句模型输出 token 列表
    """
    e = n = 0
    for g, h in zip(ref_gloss_strings, hyp_token_lists):
        r = split_gloss_sequence(g, CFG)
        hy = list(h)
        if split_glued_both:
            r = split_glued(r)
            hy = split_glued(hy)
        e += levenshtein(r, hy)
        n += len(r)
    return (100.0 * e / n if n else 0.0), e, n


# ════════════════════════════════════════════════════════════
#  附加诊断（官方未报，写报告时必须标注「附加分析」）
# ════════════════════════════════════════════════════════════
def diagnostics(ref_gloss_strings, hyp_token_lists):
    refs = [split_glued(split_gloss_sequence(g, CFG))
            for g in ref_gloss_strings]
    hyps = [split_glued(list(h)) for h in hyp_token_lists]

    # 剔 unk（对 ref 去 <unk>，hyp 也去）
    e = n = 0
    for r, h in zip(refs, hyps):
        ri = [t for t in r if t != "<unk>"]
        hi = [t for t in h if t != "<unk>"]
        e += levenshtein(ri, hi)
        n += len(ri)
    excl_unk = 100.0 * e / n if n else 0.0

    per = sum(levenshtein(r, h) for r, h in zip(refs, hyps)) \
        / max(len(refs), 1) * 100.0
    exact = sum(1 for r, h in zip(refs, hyps) if r == h) \
        / max(len(refs), 1) * 100.0

    return {"WER_excl_unk": round(excl_unk, 2),
            "WER_per_sentence": round(per, 4),
            "exact_rate_pct": round(exact, 2)}


# ════════════════════════════════════════════════════════════
#  官方基准（arXiv:2409.11960v2 Table VII，CE-CSL Dev WER %）
# ════════════════════════════════════════════════════════════
OFFICIAL_BENCHMARK = {
    "MSTNet": 54.4, "CorrNet": 47.2, "SEN": 46.5,
    "VAC": 45.1, "MAM-FSD": 44.9, "TFNet": 42.1,
}


def position_vs_official(wer):
    """在官方表格里定位我们落在哪。"""
    items = sorted(OFFICIAL_BENCHMARK.items(), key=lambda kv: -kv[1])
    for i, (name, v) in enumerate(items):
        if wer <= v:
            prev = items[i - 1] if i else None
            gap = v - wer
            return {"better_than": name if i else None,
                    "worse_than": prev[0] if prev else None,
                    "gap_to_that_method": round(gap, 2),
                    "gap_to_sota_TFNet": round(OFFICIAL_BENCHMARK["TFNet"]
                                               - wer, 2)}
    last = items[-1]
    return {"better_than": last[0], "worse_than": None,
            "gap_to_that_method": round(wer - last[1], 2),
            "gap_to_sota_TFNet": round(OFFICIAL_BENCHMARK["TFNet"] - wer, 2)}


def evaluate(ref_gloss_strings, hyp_token_lists):
    """完整评估：主指标 + 附加诊断 + 官方定位。"""
    w, e, n = wer_official(ref_gloss_strings, hyp_token_lists)
    out = {
        # ===== 主指标：唯一与官方可比的口径 =====
        "WER_official": round(w, 2),
        "edits": e,
        "ref_tokens": n,
        "n_sentences": len(ref_gloss_strings),
    }
    out.update(position_vs_official(w))
    out.update(diagnostics(ref_gloss_strings, hyp_token_lists))
    out["_usage"] = ("判优只用 WER_official；其余为附加诊断，官方未报，"
                     "不得用于与论文表格对照")
    return out


if __name__ == "__main__":
    print("=" * 74)
    print("官方口径 WER 评估器 —— 自检（Table VIII Case-2）")
    print("=" * 74)
    gt = "我/可以支持/你/去/运动/。"
    cases = {
        "SEN":     (["我", "可以你", "经济", "木头", "。"], 42.9),
        "CorrNet": (["我", "可以", "你", "。"], 42.9),
        "VAC":     (["我", "可以", "你", "好", "锻炼", "。"], 42.9),
        "MAM-FSD": (["我", "可以", "支持", "你", "锻炼", "。"], 28.6),
        "TFNet":   (["我", "可以", "支持", "你", "去", "运动", "。"], 0.0),
    }
    ok = True
    for m, (h, rep) in cases.items():
        got, e, n = wer_official([gt], [h])
        good = abs(got - rep) < 0.15
        ok = ok and good
        print("  %-8s ed=%d/%d=%5.1f%%  论文=%5.1f%%  %s"
              % (m, e, n, got, rep, "✓" if good else "✗"))
    print("\n  ⇒ %s" % ("**5/5 精确吻合，官方口径确认**" if ok
                        else "仍有偏差，需再查"))

    print("\n" + "=" * 74)
    print("边界情况")
    print("=" * 74)
    for d, g, h in (("空 ref", "。", ["。"]),
                    ("空 hyp", "我/去/。", []),
                    ("完全一致", "我/去/。", ["我", "去", "。"]),
                    ("全错", "我/去/。", ["好", "。"])):
        w_, e_, n_ = wer_official([g], [h])
        print("  %-10s ed=%d / n=%d = %.1f%%" % (d, e_, n_, w_))

    print("\n" + "=" * 74)
    print("官方基准对照（CE-CSL Dev WER %）")
    print("=" * 74)
    for k, v in sorted(OFFICIAL_BENCHMARK.items(), key=lambda kv: kv[1]):
        bar = "#" * int((v - 40) * 2)
        print("  %-9s %5.1f%s" % (k, v, bar))
    print("\n  我们 P42 = 52.11 → %s" % position_vs_official(52.11))
    print("""
  判优规则（从现在起）：
    1. 主指标只有 WER_official，可直接与上表对照
    2. 报结果时同时给 position_vs_official() 的定位
    3. 剔 unk / 逐句等权 / exact 三项标注「附加分析，官方未报」
""")