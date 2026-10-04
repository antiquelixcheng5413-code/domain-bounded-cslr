# -*- coding: utf-8 -*-
"""P12 · SCP 计数先验的有效性判定（四组对照 + 双口径 WER）

P11 只报了 blank / pfpt，**没报 WER** —— 而 WER 才是识别任务的
主指标。且 P11 的级 1 用了 oracle M（真值词数），存在**信息泄漏**：
把「答案的长度」直接告诉模型，指标必然变好，但**不能证明方法可用**。

本脚本补齐两件事：

## 1. WER（主指标，双口径）

  宽松口径：ref 用 encode 后的 b["tokens"]（OOV 折叠成 <unk>）—— 与仓库可比
  严格口径：ref 用原始 CSV gloss（保留真实 OOV）—— 真实水平

  这两个口径必须同时报，因为 P8 已证明它们差 0.1416。

## 2. 四组对照（判定收益来源）

  A) oracle M  —— 用真值 M 约束（信息泄漏，上界）
  B) 常数 M    —— 全部用 train 众数 M=5 约束（零信息）
  C) 随机 M    —— 从 train 的 M 经验分布采样（信息错误但分布正确）
  D) baseline  —— 不约束

  **判据（预先写死）**：
  - 若 A 的 WER 明显低于 B/C  -> 收益来自「计数信息」，方法有效
  - 若 A ≈ B/C               -> 收益只来自「知道长度」，方法无效

  这与 SMART Table 6 的精神一致：论文自己做了 batch size 与 alignment
  目标的对照，说明作者也把「收益是否来自方法本身」当关键判据。

## 3. 约束解码的实现口径

不是「重新选词」，而是**在已有 CTC 概率上做约束**：
  - 段数固定为 M（由 A/B/C 给出）
  - 每段的词从该段的 argmax 概率峰中取
  - 段内帧数 = 该段 argmax 连续的游程长度

只读 train/dev，不触碰 test split。
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

# `constrained_decode` 是纯 numpy 函数，可脱离 torch / cslr 单独测试
# （tests/test_scp_validity.py 在 Windows 侧导入本模块时，
#  torch 与仓库包都不在 sys.path 上）。故重依赖改为容错导入。
_CSLR_ERROR = None
try:
    import torch
    from cslr.recognition.model import CTCRecognizer, ctc_config_from_dict, BLANK_INDEX
    from cslr.recognition.gloss_sequence import GlossVocabulary, GlossSequenceConfig
    from cslr.recognition.dataset import (
        GlossSequenceDataset, FeatureNormalizer, collate_samples)
    from cslr.recognition.training import (
        iterate_batches, to_torch_batch, resolve_device, decode_batch)
    from cslr.contracts import SampleRecord
except ModuleNotFoundError as exc:
    _CSLR_ERROR = exc
    torch = None
    BLANK_INDEX = 0        # CTC 的 blank 恒为 0

try:
    from p11_scp_count_prior import collapse_repeats, read_split, ensure_link_dir
    from p8_error_attribution import levenshtein
except ModuleNotFoundError as exc2:
    _CSLR_ERROR = _CSLR_ERROR or exc2
    # 本地回退实现，保证纯函数可测（与仓库 collapse_repeats 语义一致）
    def collapse_repeats(ids):  # noqa: F811
        if ids.size == 0:
            return ids
        change = np.flatnonzero(np.diff(ids) != 0) + 1
        runs = np.concatenate([np.zeros(1, dtype=np.int64), change])
        return ids[runs][ids[runs] != BLANK_INDEX]

    def read_split(split):  # noqa: F811
        raise RuntimeError("仓库数据不可用：{}".format(_CSLR_ERROR))

    def ensure_link_dir(split):  # noqa: F811
        raise RuntimeError("仓库数据不可用：{}".format(_CSLR_ERROR))

    def levenshtein(ref, hyp):  # noqa: F811
        n, m = len(ref), len(hyp)
        d = np.zeros((n + 1, m + 1), dtype=np.int32)
        for i in range(1, n + 1):
            d[i, 0] = i
        for j in range(1, m + 1):
            d[0, j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                c = 0 if ref[i - 1] == hyp[j - 1] else 1
                d[i, j] = min(d[i - 1, j - 1] + c, d[i - 1, j] + 1, d[i, j - 1] + 1)
        return int(d[n, m]), 0, 0, 0


def constrained_decode(probs: np.ndarray, m_target: int,
                       min_gap: int = 1) -> np.ndarray:
    """在 CTC 概率矩阵上做「段数 = m_target」的约束解码。

    参数
    ----
    probs    : (T, C) 每帧的类别概率，C = 词表大小 + 1（index 0 = blank）
    m_target : 目标段数（= 该输出几个词）
    min_gap  : 相邻段起点最小间隔

    返回
    ----
    (m_target,) 的词表索引数组（若实际可分段数 < m_target 则返回较短）

    做法
    ----
    1. 用 onset 分数（该帧的类别变化量 × 置信度）贪心选 m_target 个起点，
       强制间隔 >= min_gap
    2. 每段内取 argmax 折叠后的首个非 blank 类作为该段的词
       （连续同类的游程代表一个词）
    """
    T, C = probs.shape
    m = int(max(1, min(m_target, T // max(min_gap, 1))))
    non_blank = probs[:, 1:]                      # 去掉 blank 列
    # onset 分数：类别变化量 × 置信度
    best = non_blank.max(axis=1)
    if T > 1:
        change = np.abs(non_blank[1:] - non_blank[:-1]).sum(axis=1)
        onset = np.zeros(T)
        onset[1:] = change
        onset[0] = best[0]
    else:
        onset = best.copy()
    score = onset * (0.5 + 0.5 * best)
    # 贪心选起点
    order = np.argsort(-score)
    starts = []
    for idx in order:
        if len(starts) >= m:
            break
        if all(abs(int(idx) - c) >= min_gap for c in starts):
            starts.append(int(idx))
    starts = sorted(starts)
    if not starts:
        return np.zeros(0, dtype=np.int64)
    # 段边界（末段到 T）
    bounds = starts + [T]
    out = []
    for i in range(len(starts)):
        s, e = bounds[i], bounds[i + 1]
        if e <= s:
            continue
        seg = probs[s:e].argmax(axis=1)          # 类索引（含 0=blank）
        ids = collapse_repeats(seg)               # 折叠去 blank -> 类索引
        # 只取该段的**首个**非 blank 类作为词（一个游程 = 一个词）
        nz = ids[ids != BLANK_INDEX]
        if nz.size:
            out.append(int(nz[0]) - 1)            # 类索引 -> 词表索引
    return np.asarray(out, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/ctc-landmark48-cap300.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-gap", type=int, default=1)
    ap.add_argument("--out",
                    default="artifacts/metrics/blank-gov/p12-scp-validity.json")
    a = ap.parse_args()
    if _CSLR_ERROR is not None:
        raise SystemExit("cslr 不可用：{}".format(_CSLR_ERROR))

    device = resolve_device("auto")
    print("device = {}".format(device))
    rng = random.Random(a.seed)

    # ---- M 的经验分布（用于「随机 M」对照）----
    tr = read_split("train")
    m_hist = Counter()
    for _, g in tr:
        n = len([t for t in g.split("/") if t.strip()])
        if n:
            m_hist[n] += 1
    m_mode = m_hist.most_common(1)[0][0]
    m_values = sorted(m_hist)
    m_probs = [m_hist[k] / sum(m_hist.values()) for k in m_values]
    print("\n=== M 分布 ===")
    print("  众数 M={}（{:.1f}%）  范围 {}~{}".format(
        m_mode, 100 * m_hist[m_mode] / sum(m_hist.values()), min(m_hist), max(m_hist)))

    # ---- 加载模型 ----
    payload = torch.load(REPO / a.ckpt, map_location="cpu", weights_only=False)
    cfg = ctc_config_from_dict(payload["model_config"])
    model = CTCRecognizer(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    vc = payload.get("vocabulary_config") or {}
    voc = GlossVocabulary(tokens=tuple(payload["vocabulary"]),
                          counts=dict(payload.get("vocabulary_counts") or {}),
                          config=GlossSequenceConfig(**vc) if vc else GlossSequenceConfig())
    nrm_raw = payload.get("feature_normalizer")
    nrm = FeatureNormalizer(mean=np.asarray(nrm_raw["mean"], np.float32),
                            std=np.asarray(nrm_raw["std"], np.float32) + 1e-8)

    va_root = ensure_link_dir("validation")
    dv = read_split("validation")
    ref_csv = {sid: [t.strip() for t in g.split("/") if t.strip()] for sid, g in dv}
    recs = []
    for sid, g in dv:
        if not (va_root / (sid + ".npy")).exists():
            continue
        if not ref_csv[sid]:
            continue
        recs.append(SampleRecord(sample_id=sid, video=Path(sid + ".mp4"), label=g,
                                 signer=sid.split("-")[0], session="x", split="validation"))
    recs.sort(key=lambda r: r.sample_id)
    print("dev 样本 {}".format(len(recs)))
    ds = GlossSequenceDataset(recs, va_root, voc, nrm, feature_view="full")

    # ---- 逐样本算四组 ----
    groups = ("baseline", "oracle_M", "const_M", "random_M")
    agg = {g: {"loose_d": 0, "strict_d": 0, "segs": 0, "blank": []} for g in groups}
    tot_loose_ref = 0
    tot_strict_ref = 0
    per_sample = []

    for samples in iterate_batches(ds, a.batch, shuffle=False, seed=a.seed):
        b = to_torch_batch(collate_samples(samples), device)
        with torch.no_grad():
            logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        dec, _, _ = decode_batch(lp, ol.cpu().tolist(), 1)
        for row in range(len(samples)):
            sid = samples[row].sample_id
            T = int(ol[row])
            probs = np.exp(lp[row, :T])
            # 宽松口径的 ref：b["tokens"] 已是**字符串序列**（voc.encode 的输出），
            # 不能再次 voc.decode（那是给类索引用的，会抛 TypeError）。
            # 它对应 P8 的口径：OOV 已被替换成 <unk>。
            ref_loose = list(b["tokens"][row])
            ref_strict = ref_csv[sid]                       # 严格：原始 gloss
            M_true = len(ref_strict)
            tot_strict_ref += len(ref_strict)
            tot_loose_ref += len(ref_loose)

            # D) baseline：现有 greedy
            hyp_base = voc.decode(dec[row])

            # A) oracle M
            m_a = M_true
            # B) 常数 M（众数，零信息）
            m_b = m_mode
            # C) 随机 M（分布正确、取值错误）
            m_c = rng.choices(m_values, weights=m_probs, k=1)[0]

            row_out = {"id": sid, "M_true": M_true,
                       "M_const": m_b, "M_rand": m_c}
            for tag, m_val in (("oracle_M", m_a), ("const_M", m_b), ("random_M", m_c)):
                ids = constrained_decode(probs, m_val, min_gap=a.min_gap)
                hyp = voc.decode(list(ids))
                row_out[tag + "_segs"] = int(len(hyp))
                if tag == "oracle_M":
                    row_out[tag + "_hyp"] = hyp[:10]
                d_loose = levenshtein(ref_loose, hyp)[0]
                d_strict = levenshtein(ref_strict, hyp)[0]
                agg[tag]["loose_d"] += d_loose
                agg[tag]["strict_d"] += d_strict
                agg[tag]["segs"] += len(hyp)
                row_out[tag + "_dloose"] = d_loose
                row_out[tag + "_dstrict"] = d_strict

            # baseline
            agg["baseline"]["loose_d"] += levenshtein(ref_loose, hyp_base)[0]
            agg["baseline"]["strict_d"] += levenshtein(ref_strict, hyp_base)[0]
            agg["baseline"]["segs"] += len(hyp_base)
            agg["baseline"]["blank"].append(
                float((lp[row, :T].argmax(axis=1) == BLANK_INDEX).mean()))
            row_out["baseline_dloose"] = levenshtein(ref_loose, hyp_base)[0]
            row_out["baseline_dstrict"] = levenshtein(ref_strict, hyp_base)[0]
            per_sample.append(row_out)

    # ---- 汇总 ----
    n = len(per_sample)
    print("\n" + "=" * 74)
    print("P12 · SCP 计数先验的有效性判定（{} 条 dev）".format(n))
    print("=" * 74)
    print("{:>14} {:>10} {:>10} {:>10} {:>10}".format(
        "组别", "宽松WER", "严格WER", "平均段数", "M误差"))
    print("-" * 74)
    summary = {}
    for tag in ("baseline", "const_M", "random_M", "oracle_M"):
        wl = agg[tag]["loose_d"] / max(tot_loose_ref, 1)
        ws = agg[tag]["strict_d"] / max(tot_strict_ref, 1)
        avg_seg = agg[tag]["segs"] / max(n, 1)
        # M 误差：段数 vs 真值 M
        if tag == "baseline":
            m_err = float(np.mean([abs(r["baseline_dloose"] * 0) for r in per_sample]))
            m_err = 0.0
        else:
            key = tag + "_segs"
            m_err = float(np.mean([abs(r[key] - r["M_true"]) for r in per_sample]))
        summary[tag] = {"wer_loose": wl, "wer_strict": ws,
                        "avg_segments": avg_seg, "seg_mae_vs_M": m_err}
        print("{:>14} {:>10.4f} {:>10.4f} {:>10.2f} {:>10.2f}".format(
            tag, wl, ws, avg_seg, m_err))
    print("-" * 74)
    print("参考: 严格口径 WER 的 SOTA 水平约 0.48~0.22（SMART 论文数据集）")
    print("      本项目 cap300 基线严格 WER = {:.4f}".format(
        summary["baseline"]["wer_strict"]))

    # ---- 判定 ----
    print("\n" + "=" * 74)
    print("判定：收益来自「计数信息」还是仅来自「知道长度」？")
    print("=" * 74)
    b_or = summary["const_M"]["wer_strict"]
    b_rd = summary["random_M"]["wer_strict"]
    orc = summary["oracle_M"]["wer_strict"]
    base = summary["baseline"]["wer_strict"]
    print("  baseline 严格 WER        {:.4f}".format(base))
    print("  const_M  （零信息）     {:.4f}   Δ vs base = {:+.4f}".format(
        b_or, b_or - base))
    print("  random_M （分布对）     {:.4f}   Δ vs base = {:+.4f}".format(
        b_rd, b_rd - base))
    print("  oracle_M （信息泄漏）   {:.4f}   Δ vs base = {:+.4f}".format(
        orc, orc - base))
    print()
    # oracle 相对无信息对照的增益
    gain_oracle = b_or - orc      # 正 = oracle 更好
    print("  oracle 相对 const  的增益 = {:+.4f}".format(-gain_oracle))
    print("  oracle 相对 random 的增益 = {:+.4f}".format(b_rd - orc))
    print()
    if orc < b_or - 0.02 and orc < b_rd - 0.02:
        verdict = ("oracle_M 显著优于无信息对照 -> 收益来自「计数信息」本身，"
                   "SCP 机制有效（但需级 2 的预测 M 才可用）")
    else:
        verdict = ("oracle_M 与无信息对照差异 < 0.02 -> 收益几乎全部来自"
                   "「知道答案长度」，**SCP 机制本身无额外价值**")
    print("判定：{}".format(verdict))
    print()
    print("注意：即便 oracle_M 有效，级 1 用的是**真值 M**，真实推理不可得。")
    print("      只有级 2（预测 M）达标后，SCP 才能成为可用方案。")

    receipt = {
        "ckpt": a.ckpt,
        "n_samples": n,
        "min_gap": a.min_gap,
        "m_mode_train": m_mode,
        "m_hist_train": dict(sorted(m_hist.items())),
        "groups": summary,
        "verdict": verdict,
        "gain_oracle_vs_const": float(b_or - orc),
        "gain_oracle_vs_random": float(b_rd - orc),
        "caveats": [
            "oracle_M 用真值 M，有信息泄漏，是上界探针",
            "const_M 用 train 众数 M=5，零信息对照",
            "random_M 从 train 的 M 经验分布采样，取值错误但分布正确",
            "本脚本不训练 SCP 分类器（级 2）；只判定「计数信息」是否有价值",
        ],
        "per_sample": per_sample[:60],
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
