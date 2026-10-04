"""
blank 率与 peak alignment 诊断。

为什么先做这个：97% blank 有两种完全不同的成因，干预完全不同。
  (a) 词间过渡型 blank：blank 集中在 gloss 之间，分布合理
      -> 真解法在特征判别力（AUC 0.69-0.71），加密集监督收益有限
  (b) peaky 病理型 blank：每个 gloss 只被少数帧预测，帧级定位能力极差
      -> 正是 SMART Table 3 里 CSLR-Only 的 F1@50=7.37 那个病灶
      -> 密集帧级监督是正解

论文依据：SMART (arXiv) Sec 3.2 指出 CTC 产生 peak alignment，
每个 gloss 仅由少数帧预测，其余帧全归 blank，弱时序监督限制
密集帧级表示学习。但 SMART 全篇**未给出任何 blank 率数字**，
所以这个度量是本项目自建的，不是论文报告值。

本模块只依赖 numpy，不需要模型权重、不需要 GPU。
"""

from __future__ import annotations

import numpy as np

BLANK = 0  # CTC blank 恒为 0 类（约定，见下）
EPS = 1e-8


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def blank_ratio(
    logits: np.ndarray,
    blank_index: int = BLANK,
    mode: str = "argmax",
) -> float:
    """blank 率。

    logits: (T, C) 未归一化
    mode:  "argmax"  → 贪心解码口径，与 WER 评测一致（论文用这个）
           "thresh"  → blank 概率 > 0.5 的帧占比，口径更严

    两种口径都要看：若 argmax 97% 但 thresh 只有 60%，
    说明概率质量尚可，peak 位置也还行；若两者都 97%，
    模型是彻底放弃了帧级区分能力。
    """
    p = _softmax(logits, axis=-1)
    if mode == "argmax":
        return float((np.argmax(p, axis=-1) == blank_index).mean())
    elif mode == "thresh":
        return float((p[:, blank_index] > 0.5).mean())
    raise ValueError(mode)


def peak_segments(
    logits: np.ndarray,
    blank_index: int = BLANK,
) -> list[tuple[int, int, int]]:
    """切出非 blank 连续段（贪心解码下的 peak）。

    返回 [(start, end_exclusive, label), ...]
    """
    ids = np.argmax(logits, axis=-1)
    segs: list[tuple[int, int, int]] = []
    i = 0
    t = ids.shape[0]
    while i < t:
        if ids[i] == blank_index:
            i += 1
            continue
        j = i
        while j < t and ids[j] != blank_index:
            j += 1
        # 贪心解码会合并连续重复标签，取段内众数
        vals, counts = np.unique(ids[i:j], return_counts=True)
        segs.append((i, j, int(vals[counts.argmax()])))
        i = j
    return segs


def blank_runs(
    logits: np.ndarray,
    blank_index: int = BLANK,
) -> np.ndarray:
    """连续 blank 段的长度分布。"""
    ids = np.argmax(logits, axis=-1)
    runs: list[int] = []
    cur = 0
    for v in ids:
        if v == blank_index:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return np.asarray(runs, dtype=np.int64)


def peak_alignment_report(
    logits_batch: list[np.ndarray],
    n_gloss_batch: list[int],
    blank_index: int = BLANK,
) -> dict:
    """核心诊断报告。判据全部来自 SMART 论文的定性描述 + F1@50=7.37 反推。

    关键指标 peak_frames_per_token：
      论文说每个 gloss 仅由少数帧预测。若这个值接近 1-2，
      即确诊 peaky 病理（case b）。
      正常情况下一个 gloss 应占据其时长的大部分帧。

    peak_count_ratio = peak 段数 / 真值 gloss 数：
      远小于 1 → 欠预测（漏词，WER 必然高）
      远大于 1 → 过度分割（一个词被拆成多段）
  """
    all_blank_ratio_argmax = []
    all_blank_ratio_thresh = []
    peak_lens: list[int] = []
    peak_counts: list[int] = []
    blank_run_lens: list[int] = []
    ratios: list[float] = []
    per_sample: list[dict] = []

    for logits, n_gloss in zip(logits_batch, n_gloss_batch):
        if logits.shape[0] == 0:
            continue
        ba = blank_ratio(logits, blank_index, "argmax")
        bt = blank_ratio(logits, blank_index, "thresh")
        segs = peak_segments(logits, blank_index)
        runs = blank_runs(logits, blank_index)
        n_peak = len(segs)
        lens = [e - s for s, e, _ in segs]
        n_g = max(int(n_gloss), 1)

        all_blank_ratio_argmax.append(ba)
        all_blank_ratio_thresh.append(bt)
        peak_lens.extend(lens)
        peak_counts.append(n_peak)
        blank_run_lens.extend(runs.tolist())
        ratios.append(n_peak / n_g)

        per_sample.append(
            {
                "T": int(logits.shape[0]),
                "n_gloss": n_g,
                "n_peak": n_peak,
                "peak_count_ratio": n_peak / n_g,
                "blank_ratio_argmax": ba,
                "mean_peak_len": float(np.mean(lens)) if lens else 0.0,
                "max_peak_len": int(max(lens)) if lens else 0,
            }
        )

    peak_lens_arr = np.asarray(peak_lens, dtype=np.float64)
    runs_arr = np.asarray(blank_run_lens, dtype=np.float64)
    n_tokens = max(sum(n_gloss_batch), 1)
    report = {
        "n_samples": len(logits_batch),
        "blank_ratio_argmax_mean": float(np.mean(all_blank_ratio_argmax)) if all_blank_ratio_argmax else 0.0,
        "blank_ratio_thresh_mean": float(np.mean(all_blank_ratio_thresh)) if all_blank_ratio_thresh else 0.0,
        "total_gloss": n_tokens,
        "total_peak": int(sum(peak_counts)),
        "peak_count_ratio_global": float(sum(peak_counts) / n_tokens),
        "peak_frames_per_token": float(peak_lens_arr.sum() / n_tokens) if peak_lens_arr.size else 0.0,
        "peak_len_median": float(np.median(peak_lens_arr)) if peak_lens_arr.size else 0.0,
        "peak_len_p90": float(np.percentile(peak_lens_arr, 90)) if peak_lens_arr.size else 0.0,
        "blank_run_median": float(np.median(runs_arr)) if runs_arr.size else 0.0,
        "blank_run_p90": float(np.percentile(runs_arr, 90)) if runs_arr.size else 0.0,
        "per_sample": per_sample,
    }
    report["diagnosis"] = diagnose(report)
    return report


def diagnose(report: dict) -> dict:
    """给出成因判定与建议路线。

    阈值来自 SMART 的定性描述：CTC peak alignment 使每个 gloss
    仅由少数帧预测。结合 SMART Table 3 中 CSLR-Only 的
    F1@50 = 7.37（纯 CTC 模型的帧级定位水平），本项目把
    peak_frames_per_token < 3 视为确诊 peaky。
    """
    pfpt = report["peak_frames_per_token"]
    pcr = report["peak_count_ratio_global"]
    br = report["blank_ratio_argmax_mean"]
    bt = report["blank_ratio_thresh_mean"]

    peaky = pfpt < 3.0
    severe_peaky = pfpt < 2.0

    if severe_peaky:
        verdict = "peaky_pathological"
        route = [
            "确诊 peaky 病理：每个 gloss 平均仅 %.1f 帧被预测，"
            "与 SMART 报告的 CSLR-Only F1@50=7.37 同源。" % pfpt,
            "优先级 1：TS²-TFD 免训练 TFD 产出伪边界（零标注零训练，半天）。",
            "优先级 2：BIO 三类标签 + B 标签 ±1 帧膨胀 + 加权 CE 与 CTC 联合。",
            "优先级 3：landmark 差分增强（位置→速度→加速度）+ SE 通道注意力。",
            "优先级 4：CSFormer 双流 + alpha=0.7 推理期 late fusion。",
        ]
    elif peaky:
        verdict = "peaky_mild"
        route = [
            "轻度 peaky（每 gloss %.1f 帧）。密集监督有中等收益空间。" % pfpt,
            "先做优先级 2（BIO + 联合损失），成本最低。",
            "同时并行推进特征判别力（AUC 0.69-0.71 才是真天花板）。",
        ]
    else:
        verdict = "transitional_blank"
        route = [
            "blank 分布尚合理（每 gloss %.1f 帧），不属 peaky 病理。" % pfpt,
            "blank 率 %.1f%%（argmax）主要反映词间过渡，加密集监督收益有限。" % (br * 100),
            "建议：把预算投到特征判别力（计划 2.0 的 A1/A2 路线），"
            "而非 spotting 头。",
        ]

    if pcr < 0.3:
        route.append(
            "注意：peak 数仅为真值 gloss 数的 %.0f%%，存在严重漏词，"
            "WER 高的直接原因是欠预测而非定位不准。" % (pcr * 100)
        )
    elif pcr > 2.5:
        route.append(
            "注意：peak 数为真值 gloss 数的 %.1f 倍，过度分割严重，"
            "考虑加入 SignShift 式边界间距约束。" % pcr
        )

    if br - bt > 0.3:
        route.append(
            "argmax(%.1f%%) 与 thresh(%.1f%%) 口径差距大，说明概率质量尚可，"
            "仅是尖峰位置集中 —— 这种情况 late fusion 收益会比较好。" % (br * 100, bt * 100)
        )

    return {
        "verdict": verdict,
        "peaky": peaky,
        "severe": severe_peaky,
        "route": route,
    }
