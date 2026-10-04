# -*- coding: utf-8 -*-
"""P13 · 真实数据诊断：非 blank 候选段到底存不存在

P12 的失败根因是「onset 分数在 blank 区也很高」，
但**到底有多高**、真实数据里**有没有**可选的候选段，此前都是推测。

本脚本**只做诊断，不做解码**，全部指标来自真实特征：
  1. 逐帧非 blank 置信度分布（max prob over non-blank classes）
  2. 真实候选段数（连续非 blank 游程）vs 参考词数 M
  3. 若按置信度阈值筛候选，筛出的段数分布
  4. 概率质量守恒：非 blank 类总共占多少概率质量

**所有数字均为实测，无任何公式推算。**

只读 train/dev，不触碰 test split。
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

try:
    from cslr.recognition.model import CTCRecognizer, ctc_config_from_dict, BLANK_INDEX
    from cslr.recognition.gloss_sequence import GlossVocabulary, GlossSequenceConfig
    from cslr.recognition.dataset import (
        GlossSequenceDataset, FeatureNormalizer, collate_samples)
    from cslr.recognition.training import (
        iterate_batches, to_torch_batch, resolve_device, decode_batch)
    from cslr.contracts import SampleRecord
    from p11_scp_count_prior import collapse_repeats, read_split, ensure_link_dir
    _CSLR_ERROR = None
except ModuleNotFoundError as exc:
    _CSLR_ERROR = exc
    torch = None
    BLANK_INDEX = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/ctc-landmark48-cap300.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thresholds", default="0.05,0.10,0.20,0.30,0.50")
    ap.add_argument("--out",
                    default="artifacts/metrics/blank-gov/p13-real-candidate-probe.json")
    a = ap.parse_args()
    if _CSLR_ERROR is not None:
        raise SystemExit("cslr 不可用：{}".format(_CSLR_ERROR))
    thresholds = [float(x) for x in a.thresholds.split(",")]

    device = resolve_device("auto")
    print("device = {}".format(device))

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
            nb = probs[:, 1:]                       # 非 blank 类的概率
            conf = nb.max(axis=1)                  # 每帧「最可能的那个非 blank 类」的置信度
            mass = nb.sum(axis=1)                  # 非 blank 类总概率质量
            argmax_ids = probs.argmax(axis=1)      # 逐帧 argmax（类索引）

            # 真实候选段：连续的非 blank argmax 游程
            real_segs = int(collapse_repeats(argmax_ids).size)

            rec = {
                "id": sid, "T": T, "M": len(ref_csv[sid]),
                "blank_rate": float((argmax_ids == BLANK_INDEX).mean()),
                "conf_max": float(conf.max()),
                "conf_mean": float(conf.mean()),
                "conf_p90": float(np.percentile(conf, 90)),
                "conf_p99": float(np.percentile(conf, 99)),
                "nonblank_mass_mean": float(mass.mean()),
                "nonblank_mass_max": float(mass.max()),
                "real_segments": real_segs,
                "decoded_tokens": int(len(dec[row])),
                "n_frames_above_thr": {},
                "n_segs_above_thr": {},
            }
            for th in thresholds:
                above = conf >= th
                rec["n_frames_above_thr"]["{:.2f}".format(th)] = int(above.sum())
                # 以阈值切出的候选段（连续 above 的游程）
                if above.any():
                    ch = np.flatnonzero(np.diff(above.astype(np.int64)) != 0) + 1
                    runs = np.concatenate([np.zeros(1, dtype=np.int64), ch])
                    ids = above[runs]
                    rec["n_segs_above_thr"]["{:.2f}".format(th)] = int(ids.sum())
                else:
                    rec["n_segs_above_thr"]["{:.2f}".format(th)] = 0
            per_sample.append(rec)

    n = len(per_sample)
    M = np.array([r["M"] for r in per_sample], dtype=np.float64)
    real_segs = np.array([r["real_segments"] for r in per_sample], dtype=np.float64)
    conf_max = np.array([r["conf_max"] for r in per_sample])
    conf_mean = np.array([r["conf_mean"] for r in per_sample])
    conf_p99 = np.array([r["conf_p99"] for r in per_sample])
    mass_mean = np.array([r["nonblank_mass_mean"] for r in per_sample])
    blank_rate = np.array([r["blank_rate"] for r in per_sample])
    decoded = np.array([r["decoded_tokens"] for r in per_sample], dtype=np.float64)

    print("\n" + "=" * 76)
    print("P13 · 真实数据诊断（{} 条 dev，全部实测）".format(n))
    print("=" * 76)
    print("参考词数 M        : 中位 {:.0f}  均值 {:.2f}  范围 {:.0f}~{:.0f}".format(
        np.median(M), M.mean(), M.min(), M.max()))
    print("blank 率（逐帧argmax）: 均值 {:.4f}".format(blank_rate.mean()))
    print("真实候选段（连续非blank游程）:")
    print("   均值 {:.2f}  中位 {:.0f}  范围 {:.0f}~{:.0f}".format(
        real_segs.mean(), np.median(real_segs), real_segs.min(), real_segs.max()))
    print("   vs M 的比值 {:.3f}".format(real_segs.sum() / max(M.sum(), 1)))
    print("解码输出 token 数  : 均值 {:.2f}".format(decoded.mean()))
    print()
    print("非 blank 置信度（每帧最可能非blank类的概率）:")
    print("   帧内最大值 conf_max      : 中位 {:.4f}  均值 {:.4f}  最大 {:.4f}".format(
        np.median(conf_max), conf_max.mean(), conf_max.max()))
    print("   帧内均值   conf_mean     : 中位 {:.4f}  均值 {:.4f}".format(
        np.median(conf_mean), conf_mean.mean()))
    print("   帧内 p99    conf_p99     : 中位 {:.4f}  均值 {:.4f}".format(
        np.median(conf_p99), conf_p99.mean()))
    print("非 blank 总概率质量        : 中位 {:.6f}  均值 {:.6f}".format(
        np.median(mass_mean), mass_mean.mean()))
    print()
    print("按置信度阈值筛候选（每条样本的平均）：")
    print("  阈值    帧数/样本    段数/样本    段数/M")
    thr_stats = {}
    for th in thresholds:
        k = "{:.2f}".format(th)
        fr = np.mean([r["n_frames_above_thr"][k] for r in per_sample])
        sg = np.mean([r["n_segs_above_thr"][k] for r in per_sample])
        thr_stats[k] = {"mean_frames": float(fr), "mean_segs": float(sg),
                        "segs_per_M": float(sg * n / max(M.sum(), 1))}
        print("  {:<7} {:>9.2f}  {:>11.2f}  {:>8.3f}".format(k, fr, sg, sg * n / max(M.sum(), 1)))

    # ---- 判定：候选段是否够选 M 个 ----
    print()
    print("=" * 76)
    print("判定：真实数据上是否存在「可选的候选段」？")
    print("=" * 76)
    best_thr = max(thr_stats, key=lambda k: thr_stats[k]["mean_segs"])
    bs = thr_stats[best_thr]["mean_segs"]
    print("  最宽松阈值 {} 下：平均候选段 {:.2f}，参考 M 均值 {:.2f}".format(
        best_thr, bs, M.mean()))
    if bs >= 0.5 * M.mean():
        verdict = ("存在候选段（段数 >= M/2），约束解码**有材料可用**。"
                   "可行做法：按非 blank 置信度选段起点，而非用 onset 分数。")
    elif bs >= 0.2 * M.mean():
        verdict = ("候选段很少（段数约为 M 的 {:.0%}），约束解码材料紧张。"
                   "即便选中也只能覆盖少数词，WER 改善上限有限。").format(bs / M.mean())
    else:
        verdict = ("**几乎不存在候选段**（段数仅为 M 的 {:.0%}）。"
                   "约束解码在本数据上**不适用** —— 无论怎么设计选点策略，"
                   "选出的段内部 argmax 都是 blank。").format(bs / M.mean())
    print("  {}".format(verdict))
    print()
    print("  判据依据（实测）：")
    print("    conf_mean 中位 {:.4f}  -> 非 blank 类在典型帧上几乎没有概率质量".format(
        np.median(conf_mean)))
    print("    mass_mean 中位 {:.6f} -> 非 blank 总质量仅 {:.2%}".format(
        np.median(mass_mean), np.median(mass_mean)))
    print("    真实候选段均值 {:.2f} vs M 均值 {:.2f}".format(real_segs.mean(), M.mean()))

    receipt = {
        "ckpt": a.ckpt,
        "n_samples": n,
        "note": "全部指标来自真实特征实测，无公式推算",
        "reference_M": {"median": float(np.median(M)), "mean": float(M.mean()),
                        "min": float(M.min()), "max": float(M.max())},
        "blank_rate_mean": float(blank_rate.mean()),
        "real_segments": {"mean": float(real_segs.mean()),
                          "median": float(np.median(real_segs)),
                          "min": float(real_segs.min()),
                          "max": float(real_segs.max()),
                          "ratio_to_M": float(real_segs.sum() / max(M.sum(), 1))},
        "decoded_tokens_mean": float(decoded.mean()),
        "nonblank_confidence": {
            "conf_max_median": float(np.median(conf_max)),
            "conf_max_mean": float(conf_max.mean()),
            "conf_mean_median": float(np.median(conf_mean)),
            "conf_p99_median": float(np.median(conf_p99)),
        },
        "nonblank_mass_mean": float(mass_mean.mean()),
        "threshold_scan": thr_stats,
        "verdict": verdict,
        "per_sample": per_sample[:60],
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
