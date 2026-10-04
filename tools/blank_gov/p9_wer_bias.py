# -*- coding: utf-8 -*-
"""P9 · OOV 折叠口径对结论的影响评估（纯 CPU，无需模型）

回答一个关键质疑：`GlossVocabulary.encode` 把 OOV 折叠成 `<unk>`，
那仓库报的 WER 0.8506 是否虚高？虚高会不会推翻已有结论？

三步：
  1) 量化「训练侧也被折叠」—— 不是评估 bug，而是训练目标的必然结果
  2) 计算虚高的**上界**（每句最多白送 min(1, OOV数) 个编辑距离）
  3) **敏感性分析**：把虚高当作区间 [0, U] 代入，看已有结论是否翻转

第 3 步是重点。若结论在区间内翻转，说明该结论**对口径敏感、不可靠**。

只读 CSV 标签，不加载模型，不触碰 test split。
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

try:
    from cslr.recognition.gloss_sequence import build_ordered_vocabulary
    _CSLR_ERROR = None
except ModuleNotFoundError as exc:
    _CSLR_ERROR = exc


def read_gloss(split):
    """读某个 split 的 gloss 列表。

    只接受 "train" / "validation" 两种取值，**其他值直接报错**——
    早先写成 `{"validation": "dev.csv"}.get(split, "train.csv")`，
    于是 read_gloss("dev") 静默返回了 train 的数据，
    导致 dev_side 段报出 train 的 OOV 率（32.4% 而非 30.4%）。
    静默 fallback 是这类统计脚本最危险的错误来源，必须显式拒绝。
    """
    table = {"train": "train.csv", "validation": "dev.csv"}
    if split not in table:
        raise ValueError(
            "split 必须是 {} 之一，收到 {!r}。不接受别名，"
            "以免静默读到错误的 split。".format(sorted(table), split))
    p = REPO / "data/raw/CE-CSL/label" / table[split]
    with open(p, newline="", encoding="utf-8") as f:
        return [r["Gloss"] for r in csv.DictReader(f)]


def credit_upper_bound(dev_glosses, vset):
    """折叠口径最多能白送多少编辑距离。

    一句里有 k 个 OOV 时，模型只要吐 1 个 <unk> 就能匹配掉 min(1, k) 个
    —— 因为 k 个不同的 OOV 词都被折叠成了同一个符号。
    所以白送上限 = sum over 含 OOV 的句子 of min(1, k) = 含 OOV 的句子数。

    注意：模型吐**多于** 1 个 <unk> 不会进一步降低编辑距离
    （多余的 unk 变成 insertion），故 1 个即为最优。
    """
    tot = oov_tot = credit = n_sent_oov = 0
    for g in dev_glosses:
        toks = [t.strip() for t in g.split("/") if t.strip()]
        if not toks:
            continue
        n_oov = sum(1 for t in toks if t not in vset)
        tot += len(toks)
        oov_tot += n_oov
        if n_oov:
            n_sent_oov += 1
            credit += 1        # min(1, n_oov) == 1 for any n_oov >= 1
    return {
        "n_tokens": tot,
        "n_oov_tokens": oov_tot,
        "oov_rate": oov_tot / max(tot, 1),
        "n_sent_with_oov": n_sent_oov,
        "credit_upper_bound": credit,
        "wer_inflation_upper_bound": credit / max(tot, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p9-wer-bias.json")
    a = ap.parse_args()
    if _CSLR_ERROR is not None:
        raise SystemExit("cslr 不可用：{}".format(_CSLR_ERROR))

    tr = read_gloss("train")
    dv = read_gloss("validation")
    print("train {} 条 / dev {} 条".format(len(tr), len(dv)))
    # 交叉校验：dev 的 token 数必须等于 P8/P9 独立核算的 2842
    _dev_tok = sum(len([t for t in g.split("/") if t.strip()]) for g in dv)
    print("dev gloss token 总数 {}".format(_dev_tok))
    if _dev_tok != 2842:
        raise SystemExit(
            "dev token 数为 {}，与已核算的 2842 不符 —— 读到了错误的 split？".format(_dev_tok))

    receipt = {
        "question": "OOV 折叠是否使 WER 虚高？虚高是否推翻已有结论？",
        "note": "credit 上界 = sum over sentences of min(1, oov_count)。"
                "k 个不同 OOV 折叠成同一符号，模型吐 1 个 <unk> 即可匹配 min(1,k) 个。",
    }

    # ---- 1) 训练侧是否也被折叠 ----
    print("=" * 70)
    print("1) 训练侧是否也被折叠（决定这是评估 bug 还是训练目标设计）")
    print("=" * 70)
    train_side = {}
    for mt, mf in ((300, 2), (1828, 2), (3517, 1)):
        v, _ = build_ordered_vocabulary(tr, min_frequency=mf, max_tokens=mt)
        vs = set(v.tokens) - {"<unk>"}
        tot = oov = ns = 0
        for g in tr:
            toks = [t.strip() for t in g.split("/") if t.strip()]
            if not toks:
                continue
            tot += len(toks)
            k = sum(1 for t in toks if t not in vs)
            oov += k
            if k:
                ns += 1
        key = "max_tokens={},min_frequency={}".format(mt, mf)
        train_side[key] = {
            "vocab_size": v.size,
            "n_tokens": tot,
            "n_oov_tokens": oov,
            "oov_rate": oov / max(tot, 1),
            "n_sent_with_oov": ns,
        }
        print("  {:<32} 词表 {:>5}  train OOV {} / {}  ({:.1f}%)".format(
            key, v.size, oov, tot, 100 * oov / max(tot, 1)))
    receipt["train_side_oov"] = train_side
    print()
    print("  -> cap300 下 32.4% 的**训练目标 token** 是 <unk>。")
    print("     所以模型是被显式训练去输出 <unk> 的，")
    print("     评估时的折叠与训练目标**自洽**，不是孤立的评估 bug。")

    # ---- 2) 虚高上界 ----
    print()
    print("=" * 70)
    print("2) 评估侧虚高上界")
    print("=" * 70)
    dev_side = {}
    for mt, mf in ((300, 2), (1828, 2), (3517, 1)):
        v, _ = build_ordered_vocabulary(tr, min_frequency=mf, max_tokens=mt)
        vs = set(v.tokens) - {"<unk>"}
        st = credit_upper_bound(dv, vs)
        key = "max_tokens={},min_frequency={}".format(mt, mf)
        st["vocab_size"] = v.size
        dev_side[key] = st
        print("  {:<32} 词表 {:>5}  dev OOV {:.1f}%  虚高上界 {:.4f}".format(
            key, v.size, 100 * st["oov_rate"], st["wer_inflation_upper_bound"]))
    receipt["dev_side"] = dev_side

    # ---- 3) 敏感性分析：已有结论是否翻转 ----
    print()
    print("=" * 70)
    print("3) 敏感性分析：把虚高当区间 [0, U] 代入 P6 结论")
    print("=" * 70)
    # P6 实测的宽松口径 WER（p6-vocab-ablation-long.json）
    measured = {
        "max_tokens=300,min_frequency=2": 0.8506,
        "max_tokens=1828,min_frequency=2": 0.9257,
        "max_tokens=3517,min_frequency=1": 0.9553,
    }
    sens = {}
    for key, w_loose in measured.items():
        u = dev_side[key]["wer_inflation_upper_bound"]
        lo, hi = w_loose, w_loose + u
        sens[key] = {
            "vocab_size": dev_side[key]["vocab_size"],
            "wer_loose": w_loose,
            "inflation_upper_bound": u,
            "wer_strict_range": [lo, hi],
        }
        print("  {:<32} 宽松 {:.4f}  严格区间 [{:.4f}, {:.4f}]".format(
            key, w_loose, lo, hi))
    receipt["sensitivity"] = sens

    lo_v = min(v["wer_strict_range"][0] for v in sens.values())
    hi_v = max(v["wer_strict_range"][1] for v in sens.values())
    print()
    print("  三配置严格区间并集 = [{:.4f}, {:.4f}]".format(lo_v, hi_v))
    overlap = all(
        sens[a]["wer_strict_range"][0] <= sens[b]["wer_strict_range"][1]
        and sens[b]["wer_strict_range"][0] <= sens[a]["wer_strict_range"][1]
        for a in sens for b in sens)
    print("  区间两两重叠 = {}".format(overlap))
    if overlap:
        print()
        print("  ⚠️ 判定：区间高度重叠 -> 「词表越大 WER 越差」**在严格口径下不成立**。")
        print("     该结论**对口径敏感**，只能表述为「宽松口径下观测到该趋势，")
        print("     但不足以判定三者优劣」。")
        print()
        print("  注意方向：cap300 的 OOV 率最高(30.4%)、虚高上界最大(0.1650)，")
        print("  其区间上界 1.0156 反而**高于** cap1828/cap3517 的 1.0077。")
        print("  即口径偏差在 cap300 上最有利，真实优劣无法判定。")
    receipt["conclusion_overlap"] = overlap
    receipt["verdict"] = (
        "P6 的「词表越大 WER 越差」对 OOV 折叠口径敏感，严格口径下区间重叠，"
        "不足以判定优劣；「词表裁剪设下界」（OOV 30.4%）不受影响。"
        if overlap else "区间不重叠，原结论成立。")

    # ---- 4) 哪些结论不受影响 ----
    print()
    print("=" * 70)
    print("4) 不受口径影响的结论")
    print("=" * 70)
    unaffected = [
        ("词表裁剪设下界（OOV 30.4%）", "客观计数，与评估口径无关"),
        ("P8 错误归因", "已用严格口径（原始 CSV ref）重算"),
        ("P7 signer 分析", "相对比较，两侧同口径偏差抵消"),
        ("A1/A2 的 macro-AUC", "不涉及 gloss 序列，不用 WER"),
        ("P0-P5 的 blank/peaky 诊断", "基于 argmax 与 blank 率，不受 ref 口径影响"),
    ]
    for name, why in unaffected:
        print("  [OK] {:<32} {}".format(name, why))
    receipt["unaffected"] = [{"conclusion": n, "reason": w} for n, w in unaffected]

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
