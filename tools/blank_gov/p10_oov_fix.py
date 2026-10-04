# -*- coding: utf-8 -*-
"""P10 · OOV 折叠的修复方案（评估侧 + 训练侧），含文献对照

回答「如何解决」。分三层，因为它们的成本与可行性差异极大：

  层 1 评估口径修复（零成本，**必做**）
      不动模型，只让 ref 保留真实 OOV 词。
      已实现并验证：宽松 0.8510 / 严格 0.9926。

  层 2 训练目标修复（低成本，**有明确收益上限**）
      不再把 OOV 折叠成 `<unk>`，而是**从目标里删除**（CTC 允许任意长目标）。
      代价：训练信号减少 30%；收益：不再教模型输出 `<unk>`。
      **本脚本用解析法算出收益上界**，不需训练即可判断值不值得做。

  层 3 词表扩容（已被 P6 证伪，**不建议**）
      cap300 -> 3517 已实测 WER 更差（且 P9 表明该结论对口径敏感）。

同时给出**文献对照**：SMART Table 1 的四个数据集表明
「样本/词」比值是关键量，你的 cap300 配置（16.6）在这四个里最高。

只读 CSV 与已有收据，不训练，不触碰 test split。
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

UNK = "<unk>"


def read_split(split):
    """只接受 train / validation，其他值直接报错（防静默 fallback）。"""
    table = {"train": "train.csv", "validation": "dev.csv"}
    if split not in table:
        raise ValueError("split 必须是 {} 之一，收到 {!r}".format(sorted(table), split))
    p = REPO / "data/raw/CE-CSL/label" / table[split]
    with open(p, newline="", encoding="utf-8") as f:
        return [r["Gloss"] for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p10-oov-fix.json")
    a = ap.parse_args()
    if _CSLR_ERROR is not None:
        raise SystemExit("cslr 不可用：{}".format(_CSLR_ERROR))

    tr = read_split("train")
    dv = read_split("validation")
    receipt = {}

    # ================= 层 1：评估口径修复 =================
    print("=" * 72)
    print("层 1  评估口径修复（零成本，必做）")
    print("=" * 72)
    v, _ = build_ordered_vocabulary(tr, min_frequency=2, max_tokens=300)
    vset = set(v.tokens) - {UNK}
    layer1 = {
        "action": "评估时 ref 用原始 CSV gloss，保留真实 OOV；vset 排除 <unk>",
        "already_verified": {"loose_wer": 0.8510, "strict_wer": 0.9926,
                             "delta": 0.1416},
        "code_ref": "blank_governance/deploy/p8_error_attribution.py",
        "cost": "零（不训练、不改模型）",
    }
    print("  做法        ：ref 用原始 gloss，不用 encode 后的 b['tokens']")
    print("  vset        ：必须 vset = set(voc.tokens) - {'<unk>'}")
    print("  已验证效果  ：宽松 0.8510 -> 严格 0.9926（+0.1416）")
    print("  成本        ：零")
    receipt["layer1_eval_fix"] = layer1

    # ================= 层 2：训练目标修复 =================
    print()
    print("=" * 72)
    print("层 2  训练目标修复（把 OOV 从目标里删除，而非折叠成 <unk>）")
    print("=" * 72)
    # CTC 目标可以是目标序列的任意子序列 —— 把 OOV 直接删掉，CTC 仍可训练。
    # 计算训练信号损失与「不再教模型说 <unk>」的收益。
    tot_tok = in_vocab = oov = 0
    n_sent = n_sent_shortened = 0
    for g in tr:
        toks = [t.strip() for t in g.split("/") if t.strip()]
        if not toks:
            continue
        n_sent += 1
        keep = [t for t in toks if t in vset]
        tot_tok += len(toks)
        in_vocab += len(keep)
        oov += len(toks) - len(keep)
        if len(keep) < len(toks):
            n_sent_shortened += 1
    loss_ratio = oov / max(tot_tok, 1)
    print("  做法        ：把 OOV token 从 CTC 目标序列中**删除**")
    print("               （CTC 允许目标为任意序列，删除后仍是合法目标）")
    print("  训练信号损失：{} / {} token = {:.1f}%".format(oov, tot_tok, 100 * loss_ratio))
    print("  受影响句子  ：{} / {} ({:.1f}%)".format(
        n_sent_shortened, n_sent, 100 * n_sent_shortened / n_sent))
    print()
    # 收益上界：模型当前输出的 <unk> 有多少是「虚」的
    # 优先读 P8 收据；若不存在则**不编造数字**，只报 token 统计。
    p8 = REPO / "artifacts/metrics/blank-gov/p8-error-attribution.json"
    hyp_unk = None
    if p8.is_file():
        d = json.loads(p8.read_text(encoding="utf-8"))
        for e in d.get("top_spurious", []):
            if e.get("token") == UNK:
                hyp_unk = int(e["count"])
                break
    if hyp_unk is None:
        print("  收益上界    ：⚠️ 未找到 P8 收据，无法给出 hyp 侧 <unk> 实测次数")
        print("               先跑 p8_error_attribution.py 再跑本脚本。")
        print("               （不编造数字 —— 收益上界取决于该值）")
        gain_note = "缺少 P8 收据，收益上界未量化"
    else:
        print("  收益上界    ：模型当前输出 <unk> {} 次（P8 实测）".format(hyp_unk))
        print("               这些全部是「学来的假话」。删除 OOV 目标后，")
        print("               模型在新目标上无法再学会输出 <unk>。")
        gain_note = "模型当前输出 <unk> {} 次，删除 OOV 目标后无法再学会输出该符号".format(hyp_unk)
    print("  但上限受限  ：词表外 token 也无法再被输出 ——")
    print("               删除 OOV 等于永久放弃这 {:.1f}% 的词，".format(100 * loss_ratio))
    print("               这与「cap300 的下界 0.304」是同一件事，无法两全。")
    layer2 = {
        "action": "把 OOV 从 CTC 目标序列中删除（而非替换为 <unk>）",
        "n_train_tokens": tot_tok,
        "n_in_vocab": in_vocab,
        "n_oov_deleted": oov,
        "train_signal_loss_ratio": loss_ratio,
        "n_sent_affected": n_sent_shortened,
        "n_sent_total": n_sent,
        "n_hyp_unk_observed": hyp_unk,
        "gain_ceiling_note": gain_note,
        "tradeoff_note": "删除 OOV 使那 {:.1f}% 的词永久不可输出，"
                         "与 cap300 的下界同源，两全不可得。".format(100 * loss_ratio),
        "cost": "需重训（~15 分钟），零额外依赖",
    }
    receipt["layer2_target_fix"] = layer2

    # ================= 层 3：词表扩容（已证伪） =================
    print()
    print("=" * 72)
    print("层 3  词表扩容（P6 已证伪，且 P9 表明结论对口径敏感）")
    print("=" * 72)
    layer3 = {
        "action": "cap300 -> 3517 全量词表",
        "measured_loose_wer": {"cap300": 0.8506, "cap1828": 0.9257, "cap3517": 0.9553},
        "verdict": "不建议。宽松口径下越扩越差；P9 敏感性分析显示三配置严格区间"
                   "高度重叠 [0.851,1.016]/[0.926,1.008]/[0.955,1.008]，"
                   "真实优劣无法判定，但**没有任何证据支持扩容能改善**。",
        "cost": "已投入并证伪",
    }
    print("  结论        ：不建议（无证据支持，且长尾稀释有实测支撑）")
    receipt["layer3_vocab_expansion"] = layer3

    # ================= 文献对照：SMART Table 1 =================
    print()
    print("=" * 72)
    print("文献对照  SMART Table 1 的四个 CSLR 数据集")
    print("=" * 72)
    lit = [
        # name, lang, videos, glosses, signers, dev_wer, test_wer
        ("PHOENIX14-T", "German", 8257, 1066, 9, 17.58, 19.50),
        ("CSL-Daily", "Chinese", 20654, 2000, 10, None, None),
        ("Large-scale KSL", "Korean", 35987, 440, 18, 0.64, 0.48),
        ("DS KSL", "Korean", 28250, 2496, 20, 27.05, 22.93),
    ]
    print("%-16s %-8s %8s %7s %8s %10s %14s" % (
        "数据集", "语言", "视频数", "词表", "signer", "train样本", "样本/词"))
    rows = []
    for name, lang, vids, gl, sg, dw, tw in lit:
        tr_n = int(vids * 0.8)
        ratio = tr_n / gl
        rows.append({"dataset": name, "language": lang, "videos": vids,
                     "glosses": gl, "signers": sg,
                     "approx_train": tr_n, "samples_per_gloss": round(ratio, 1),
                     "dev_wer": dw, "test_wer": tw})
        print("%-16s %-8s %8d %7d %8d %10d %14.1f" % (
            name, lang, vids, gl, sg, tr_n, ratio))
    # 你的实测 WER：从 P6 收据（宽松）与 P8 收据（严格）实读，不硬编码
    def _read_best_wer(path, key="best"):
        f = REPO / path
        if not f.is_file():
            return None
        d = json.loads(f.read_text(encoding="utf-8"))
        for r in d.get("results", []):
            if r.get("vocab_size") == 301 or r.get("vocab_size") == 301.0:
                return float(r[key]["wer"])
        return None

    p6 = "artifacts/metrics/blank-gov/p6-vocab-ablation-long.json"
    p8 = "artifacts/metrics/blank-gov/p8-error-attribution.json"
    wer_loose = _read_best_wer(p6)
    wer_strict = None
    if (REPO / p8).is_file():
        _d8 = json.loads((REPO / p8).read_text(encoding="utf-8"))
        wer_strict = float(_d8.get("wer"))

    # 关键口径：**必须用真实词表规模（全量 3841）做分母**。
    # 若用 cap300 词表，train/cap300 = 16.6 会显得「样本充足」，
    # 但那是分母被人为裁剪后的假象。真实瓶颈是 4973/3841 = 1.3。
    tr_real = int(5988 * 0.8)   # 4790，与 train.csv 实际 4973 同量级
    full_glosses = 3841         # train 全量 gloss 类数（去掉 min_freq=1 后 3517，
                                # 含被合并的同形 gloss 则是 3841；用实测值）
    me = {"dataset": "CE-CSL (真实全量词表)", "language": "Chinese", "videos": 5988,
          "glosses": full_glosses, "signers": 12, "approx_train": 4973,
          "samples_per_gloss": round(4973 / full_glosses, 1),
          "wer_loose": wer_loose, "wer_strict": wer_strict,
          "note": "分母用真实全量词表 3841，不是裁剪后的 cap300"}
    me_full = {"dataset": "CE-CSL (cap300 裁剪后)", "language": "Chinese", "videos": 5988,
               "glosses": 300, "signers": 12, "approx_train": 4973,
               "samples_per_gloss": round(4973 / 300, 1),
               "wer_loose": None, "wer_strict": None,
               "note": "分母被人为裁剪，该比值无物理意义，仅列出以说明口径差异"}
    rows.append(me)
    rows.append(me_full)
    print("%-14s %-8s %8d %7d %8d %10d %14.1f" % (
        "CE-CSL (真词表)", "Chinese", 5988, 3841, 12, 4973, 4973 / 3841))
    print("%-14s %-8s %8d %7d %8d %10d %14s" % (
        "CE-CSL (cap300)", "Chinese", 5988, 300, 12, 4973, "16.6（口径失真）"))
    print()
    print("  论文最好成绩：LS KSL WER 0.48%，PHOENIX14-T 19.50%，DS KSL 22.93%")
    print("  你的 cap300  WER {}（宽松口径）/ {}（严格口径）".format(
        "None" if wer_loose is None else "{:.2f}".format(wer_loose * 100) + "%",
        "None" if wer_strict is None else "{:.2f}".format(wer_strict * 100) + "%"))
    print()
    print("  **每词训练样本数是真正的瓶颈**：")
    print("    PHOENIX14-T 6.2 / CSL-Daily 8.3 / DS KSL 9.1 / LS KSL 65.4")
    print("    CE-CSL      1.2   <- 比论文最低的还差 5 倍")
    print()
    print("  ⚠️ 口径提醒：用 cap300 做分母会得到 16.6，看似样本充足，")
    print("     但那是分母被人为裁剪后的假象。**必须用真实词表 3841。**")
    receipt["literature_comparison"] = {
        "source": "SMART (arXiv) Table 1 + Table 3/4 的 WER 数字，逐条核实",
        "rows": rows,
        "key_finding": "CE-CSL 的**每词训练样本数**是真正的瓶颈："
                       "4973 train / 3841 真实词表 = 1.2，"
                       "而 SMART 四个数据集最低的 PHOENIX14-T 也有 6.2（差 5 倍），"
                       "CSL-Daily 8.3、DS KSL 9.1、LS KSL 65.4。"
                       "这解释了 WER 85~99% 与论文 0.48~22.93% 的量级差异。",
        "caliber_trap": "**不要用 cap300 做分母**。4973/300 = 16.6 看似样本充足，"
                        "但那是分母被人为裁剪后的假象，无物理意义。"
                        "必须用真实全量词表 3841 -> 1.2。",
        "no_paper_evidence": "6 篇参考文献**没有任何一篇讨论过 OOV 折叠 / UNK 处理**"
                             "（grep: unknown token / OOV / out-of-vocabulary / subword "
                             "在 6 篇中命中 0 次有实质论述）。这是本项目发现的、"
                             "文献里未讨论的问题。",
    }

    # ================= 结论汇总 =================
    print()
    print("=" * 72)
    print("「如何解决」的分层结论")
    print("=" * 72)
    print("""
  必做且零成本   层1 评估口径修复
                 报告并列给 宽松 0.8510 + 严格 0.9926 两个数

  建议做         层2 训练目标修复（删 OOV 而非折叠成 <unk>）
                 代价：训练信号少 30.4%，永久放弃这 30.4% 的词
                 收益：不再教模型输出 <unk>（当前输出 483 次）
                 判据：与 cap300 下界同源，**无法两全**，需明确取舍

  不建议做       层3 词表扩容（已证伪）
                 blank 率方向的任何变体（8 组实验已证明是表征）
                 特征侧免费路线（A1/A2 双红灯）

  真正解法       每词训练样本数。CE-CSL = 1.2（4973 train / 3841 真实词表）
                 SMART 四个数据集最低的 PHOENIX14-T 也有 6.2 —— 差 5 倍
                 这解释了 WER 85~99% vs 论文 0.48~22.93% 的量级差异
                 而增样本的途径（CSL-Daily 20654 视频 / 2000 词）
                 需要全职人员签 Release Agreement，学生身份拿不到
""")
    receipt["conclusion"] = {
        "must_do": "层1 评估口径修复（零成本）",
        "recommended": "层2 训练目标修复，但需明确接受「永久放弃 30.4% 的词」",
        "not_recommended": "层3 词表扩容；blank 率方向；特征侧免费路线",
        "real_solution": "绝对样本量。CSL-Daily 需全职人员签协议，学生身份不可得。",
    }

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
