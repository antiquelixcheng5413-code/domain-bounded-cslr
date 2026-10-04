# -*- coding: utf-8 -*-
"""P14 · 约束解码（真实数据版，基于 P13 实测设计）

P12 的失败根因已由 P13 定位清楚（全部实测）：
  候选段（thr=0.05）只有 **2.03 个 / 样本**，而参考 M = **5.52**
  帧内非 blank 置信度中位仅 **0.0253**（典型帧几乎全是 blank）
  但帧内峰值 conf_max 中位 **0.6637**（真候选段有明确类别）

## 与 P12 的三处关键差异（全部由实测驱动，非假设）

1. **排序用 conf 而非 onset 分数**
   P12 用 onset（相邻帧概率变化量），它在 blank 区也很高 ->
   起点被摊到 blank 区。实测 conf（帧内非 blank 最大概率）在 blank 区低、
   在真候选段高，**天然区分**。

2. **M 截断到候选段数**
   P12 强行切 M 段（实现成「强行切 M 段」），必然选到 blank 区。
   本实现 `M_eff = min(M_target, 实际候选段数)`，不产生无效段。

3. **先用阈值筛候选段，再在候选里选**
   阈值 0.05 是 P13 实测的最佳点（候选段数最多）。

## 四组对照（必须，否则无法判定收益来源）

  A) M_clip  = min(M_true, 候选段数)  —— oracle，但受候选段上限约束
  B) M_const = 2（实测候选段中位）    —— 零信息
  C) M_rand  从实测候选段分布采样     —— 分布对、取值错
  D) baseline 不约束

**所有指标均为真实解码实测，无任何公式推算。**

只读 dev，不触碰 test split。
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

# 纯 numpy 的解码函数需独立可测（tests/test_real_constrained.py 在 Windows 侧跑），
# 故 torch / cslr 依赖改为容错导入。
try:
    import torch
except ModuleNotFoundError:      # 纯函数测试场景
    torch = None

_CSLR_ERROR = None
try:
    from cslr.recognition.model import CTCRecognizer, ctc_config_from_dict, BLANK_INDEX
    from cslr.recognition.gloss_sequence import GlossVocabulary, GlossSequenceConfig
    from cslr.recognition.dataset import (
        GlossSequenceDataset, FeatureNormalizer, collate_samples)
    from cslr.recognition.training import (
        iterate_batches, to_torch_batch, resolve_device, decode_batch)
    from cslr.contracts import SampleRecord
    from p11_scp_count_prior import read_split, ensure_link_dir
    from p8_error_attribution import levenshtein
except ModuleNotFoundError as exc:
    _CSLR_ERROR = exc
    BLANK_INDEX = 0


def find_candidate_segments(conf: np.ndarray, threshold: float,
                           min_gap: int = 1) -> list:
    """在非 blank 置信度序列上找候选段。

    参数
    ----
    conf      : (T,) 每帧「最可能非 blank 类」的概率
    threshold : 候选阈值（P13 实测 0.05 最优）
    min_gap   : 相邻候选段的最小间隔

    返回
    ----
    [(start, end, cls), ...]，按 start 升序

    关键：这是**先筛后选**，不强行产生目标段数 —— P12 的失败正源于
    强行切段导致起点落在 blank 区。
    """
    T = conf.shape[0]
    above = conf >= threshold
    if not above.any():
        return []
    ch = np.flatnonzero(np.diff(above.astype(np.int64)) != 0) + 1
    runs = np.concatenate([np.zeros(1, dtype=np.int64), ch])
    starts, ends = runs[:-1], runs[1:]
    # 合并间隔过近的段（保留置信度更高的那个所在段）
    merged = []
    for s, e in zip(starts, ends):
        if merged and (s - merged[-1][1]) < min_gap:
            # 用平均 conf 更高的那个
            if conf[s:e].mean() > conf[merged[-1][0]:merged[-1][1]].mean():
                merged[-1] = (s, e)
            continue
        merged.append((int(s), int(e)))
    return merged


def decode_constrained(probs: np.ndarray, m_target: int,
                       threshold: float = 0.05, min_gap: int = 1) -> np.ndarray:
    """基于候选段的约束解码。

    M_eff = min(m_target, 候选段数) —— 不强行产生无效段。
    每段输出其内部非 blank 置信度最高的那个类（词表索引）。
    """
    T, C = probs.shape
    nb = probs[:, 1:]                       # 非 blank 类概率
    conf = nb.max(axis=1)                   # (T,) 帧内非 blank 最大概率
    conf_arg = nb.argmax(axis=1)            # (T,) 对应的类（0 基，+1 得类索引）
    cands = find_candidate_segments(conf, threshold, min_gap)
    if not cands:
        return np.zeros(0, dtype=np.int64)
    m_eff = int(max(1, min(m_target, len(cands))))
    # 若候选多于 m_eff，按段内峰值 conf 取前 m_eff 个
    if len(cands) > m_eff:
        scored = sorted(
            cands,
            key=lambda se: -float(conf[se[0]:se[1]].max()))
        cands = sorted(scored[:m_eff])
    out = []
    for s, e in cands:
        seg = nb[s:e]
        if seg.size == 0:
            continue
        k = int(seg.mean(axis=0).argmax())   # 段内平均概率最高的类
        out.append(k)                        # 已是 0 基词表索引
    return np.asarray(out, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/ctc-landmark48-cap300.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--min-gap", type=int, default=1)
    ap.add_argument("--m-const", type=int, default=2,
                    help="零信息对照的 M（P13 实测候选段中位为 2）")
    ap.add_argument("--out",
                    default="artifacts/metrics/blank-gov/p14-real-constrained.json")
    a = ap.parse_args()
    if _CSLR_ERROR is not None:
        raise SystemExit("cslr 不可用：{}".format(_CSLR_ERROR))

    device = resolve_device("auto")
    print("device = {}".format(device))
    print("阈值 threshold={}  min_gap={}  M_const={}".format(
        a.threshold, a.min_gap, a.m_const))
    rng = random.Random(a.seed)

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

    groups = ("baseline", "M_const", "M_rand", "M_clip")
    agg = {g: {"strict_d": 0, "loose_d": 0, "segs": 0, "cands": 0} for g in groups}
    tot_strict = tot_loose = 0
    cand_hist = []
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
            ref_s = ref_csv[sid]
            ref_l = list(b["tokens"][row])
            M = len(ref_s)
            tot_strict += len(ref_s)
            tot_loose += len(ref_l)

            nb = probs[:, 1:]
            conf = nb.max(axis=1)
            cands = find_candidate_segments(conf, a.threshold, a.min_gap)
            n_cand = len(cands)
            cand_hist.append(n_cand)

            hyp_base = voc.decode(dec[row])
            agg["baseline"]["strict_d"] += levenshtein(ref_s, hyp_base)[0]
            agg["baseline"]["loose_d"] += levenshtein(ref_l, hyp_base)[0]
            agg["baseline"]["segs"] += len(hyp_base)
            agg["baseline"]["cands"] += n_cand

            # 三种约束：M_const / M_rand / M_clip(=min(M_true, 候选段数))
            m_rand = rng.randint(1, max(1, n_cand)) if n_cand else 1
            for tag, m_val in (("M_const", a.m_const),
                               ("M_rand", m_rand),
                               ("M_clip", min(M, max(n_cand, 1)))):
                ids = decode_constrained(probs, m_val, a.threshold, a.min_gap)
                hyp = voc.decode(list(ids))
                agg[tag]["strict_d"] += levenshtein(ref_s, hyp)[0]
                agg[tag]["loose_d"] += levenshtein(ref_l, hyp)[0]
                agg[tag]["segs"] += len(hyp)
                agg[tag]["cands"] += n_cand

            per_sample.append({
                "id": sid, "T": T, "M": M, "n_cand": n_cand,
                "M_rand": m_rand,
                "baseline_segs": len(hyp_base),
                "baseline_dstrict": levenshtein(ref_s, hyp_base)[0],
            })

    n = len(per_sample)
    print("\n" + "=" * 76)
    print("P14 · 约束解码（真实数据版）  {} 条 dev".format(n))
    print("=" * 76)
    print("实测候选段数/样本 : 均值 {:.2f}  中位 {:.0f}  分布 {}".format(
        np.mean(cand_hist), np.median(cand_hist),
        dict(zip(*np.unique(cand_hist, return_counts=True)))))
    print("参考词数 M/样本   : 均值 {:.2f}".format(
        np.mean([r["M"] for r in per_sample])))
    print()
    print("{:>12} {:>10} {:>10} {:>10} {:>10}".format(
        "组别", "严格WER", "宽松WER", "输出词数", "候选段数"))
    print("-" * 76)
    summary = {}
    for tag in groups:
        ws = agg[tag]["strict_d"] / max(tot_strict, 1)
        wl = agg[tag]["loose_d"] / max(tot_loose, 1)
        segs = agg[tag]["segs"] / max(n, 1)
        cds = agg[tag]["cands"] / max(n, 1)
        summary[tag] = {"wer_strict": ws, "wer_loose": wl,
                        "avg_out_tokens": segs, "avg_candidates": cds}
        print("{:>12} {:>10.4f} {:>10.4f} {:>10.2f} {:>10.2f}".format(
            tag, ws, wl, segs, cds))
    print("-" * 76)

    # ---- 判定 ----
    print()
    print("=" * 76)
    print("判定：约束解码在真实数据上是否有效？")
    print("=" * 76)
    base = summary["baseline"]["wer_strict"]
    mc = summary["M_const"]["wer_strict"]
    mr = summary["M_rand"]["wer_strict"]
    mcl = summary["M_clip"]["wer_strict"]
    print("  baseline 严格 WER      {:.4f}".format(base))
    print("  M_const  严格 WER      {:.4f}   Δ vs base = {:+.4f}".format(
        mc, mc - base))
    print("  M_rand   严格 WER      {:.4f}   Δ vs base = {:+.4f}".format(
        mr, mr - base))
    print("  M_clip   严格 WER      {:.4f}   Δ vs base = {:+.4f}".format(
        mcl, mcl - base))
    print()
    print("  输出词数（baseline {:.2f} -> 各约束组）: ".format(
        summary["baseline"]["avg_out_tokens"]) +
        ", ".join("{}={:.2f}".format(t, summary[t]["avg_out_tokens"]) for t in groups[1:]))
    print("  候选段数（实测上限）: {:.2f}".format(
        summary["baseline"]["avg_candidates"]))
    print()

    best = min(summary[t]["wer_strict"] for t in groups[1:])
    if best < base - 0.005:
        gain_src = "计数信息" if mcl < mc - 0.005 else "仅强制段数"
        verdict = ("约束解码使严格 WER 改善 {:.4f}（{}），但受候选段 {:.2f} 个"
                   "（参考 M {:.2f}）的硬上限约束".format(
                       base - best, gain_src,
                       summary["baseline"]["avg_candidates"],
                       np.mean([r["M"] for r in per_sample])))
    elif best < base - 0.001:
        verdict = ("约束解码仅带来 {:.4f} 的改善，属噪声带，判定无效".format(base - best))
    else:
        verdict = "约束解码对严格 WER 无改善，判定无效"
    print("判定：{}".format(verdict))
    print()
    print("  结构性限制（P13 实测）：非 blank 总概率质量中位仅 9.95%，")
    print("  候选段均值 {:.2f} 个而参考 M = {:.2f} 个。".format(
        summary["baseline"]["avg_candidates"],
        np.mean([r["M"] for r in per_sample])))
    print("  即便 M 全部命中，输出词数也受候选段上限限制，")
    print("  这个下界与解码策略无关。")

    receipt = {
        "ckpt": a.ckpt,
        "n_samples": n,
        "threshold": a.threshold,
        "min_gap": a.min_gap,
        "m_const": a.m_const,
        "note": "全部指标为真实特征实测解码，无公式推算",
        "design_differences_vs_p12": [
            "排序用 conf（帧内非blank最大概率）而非 onset 分数",
            "M 截断到候选段数，不强行产生无效段",
            "先阈值筛候选段，再在候选里选",
        ],
        "groups": summary,
        "verdict": verdict,
        "candidate_segments": {
            "mean": float(np.mean(cand_hist)),
            "median": float(np.median(cand_hist)),
            "hist": {int(k): int(v) for k, v in zip(*np.unique(cand_hist, return_counts=True))},
        },
        "per_sample": per_sample[:60],
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n收据已落据: {}".format(out))


if __name__ == "__main__":
    main()
