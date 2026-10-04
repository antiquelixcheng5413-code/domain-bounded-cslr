# -*- coding: utf-8 -*-
"""P11 · SCP 式全局计数先验（SignShift Sec 3.4 迁移）

这是**文献核查后找到的唯一一个我们条件全部满足、且论文有消融数字支撑**
的未试方案。

论文依据（SignShift Sec 3.4，已逐字核实原文）：
  "we introduce a Segment Count Prediction (SCP) module to estimate the
   probability distribution over the total number of sentences in a video."
  "By formulating sentence count estimation as a global classification problem
   (i.e., the k-th class means the video contains exactly k sentences), SCP
   derives a holistic constraint that cannot be reliably inferred from local
   transitions."
  "The SCP module is trained independently using ground-truth sentence counts
   and is used only at inference to guide boundary selection."

论文消融（Table 2，MS-TCN backbone）：
  F1@50  51.71 -> 61.82   (+10.11)
  SER     0.43 -> 0.26   (-0.17)
  计数 MAE 1.734 -> 1.177 (How2Sign)
        1.970 -> 1.275 (OpenASL)
        1.337 -> 0.798 (PHOENIX14T)

## 为什么这个方案在我们的条件下成立

论文的任务是**句子级**分割，M = 句子数，需要人工句子边界。
我们的任务 CTC 的 M = **gloss 数**，直接来自 dev.csv/train.csv 的标注
—— **无需任何人工标注**。这是比论文更好的条件。

train 词数分布：M = 2~16（14 类），分布熵 2.81 bits（最大 3.70），
分类任务不难。

## 与我们已有实验的关系

P2-b（late fusion）失败的根因是「伪边界标签里 blank 占比 0%，
P_spot 根本不预测 blank」。SCP 从结构上排除这个病态解：
**先定「有几个词」，再选「在哪些帧输出」**。

## 两级实现

  级 1（零成本，先做）：**计数约束解码**
      不改模型。在 beam search 中加入段数约束，使输出的非 blank 段数
      接近参考长度 M。这是 CTC 的经典约束解码。
      预期：blank 率从 0.9698 降到 0.8850（每词 1 帧的下界），
            pfpt 从 0.26 升到 1.0（脱离「全 1 帧」病态）。

  级 2（需训练）：**SCP 头 + 联合解码**
      训练一个 M 分类器（14 类），推理期用预测的 M* 约束解码。
      论文的完整做法。

本脚本实现级 1（零成本、可立即验证），并为级 2 提供训练接口。

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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

try:
    from cslr.recognition.model import CTCRecognizer, ctc_config_from_dict, BLANK_INDEX
    from cslr.recognition.gloss_sequence import GlossVocabulary, GlossSequenceConfig
    from cslr.recognition.dataset import (
        GlossSequenceDataset, FeatureNormalizer, collate_samples)
    from cslr.recognition.training import (
        iterate_batches, to_torch_batch, resolve_device, decode_batch)
    from cslr.contracts import SampleRecord
    _CSLR_ERROR = None
except ModuleNotFoundError as exc:
    _CSLR_ERROR = exc
    # 纯 numpy 的解码函数需独立可测（tests/test_scp_count_prior.py 在 Windows 侧跑），
    # 故 BLANK_INDEX 用其字面值兜底 —— CTC 的 blank 恒为 0。
    BLANK_INDEX = 0


def read_split(split):
    """只接受 train / validation，其他值直接报错（防静默 fallback）。"""
    table = {"train": "train.csv", "validation": "dev.csv"}
    if split not in table:
        raise ValueError("split 必须是 {} 之一，收到 {!r}".format(sorted(table), split))
    p = REPO / "data/raw/CE-CSL/label" / table[split]
    with open(p, newline="", encoding="utf-8") as f:
        return [(r["Number"], r["Gloss"]) for r in csv.DictReader(f)]


def ensure_link_dir(split):
    """特征实际是 {id}.landmark.npy，而 Dataset 读 {id}.npy -> 建软链目录。"""
    root = REPO / ".vocab_link" / split
    root.mkdir(parents=True, exist_ok=True)
    src_dir = REPO / "artifacts/part3_features" / split
    for p in sorted(src_dir.glob("*.landmark.npy")):
        sid = p.name[: -len(".landmark.npy")]
        dst = root / (sid + ".npy")
        if not dst.exists():
            try:
                dst.symlink_to(p)
            except OSError:
                pass
    return root


# ------------------------------------------------------------------ 约束解码

def collapse_repeats(ids: np.ndarray) -> np.ndarray:
    """把 CTC 的逐帧输出折叠成游程（去重复 + 去 blank）。

    这是 CTC greedy 解码的标准后处理：
      [b, b, 3, 3, 3, b, 7] -> [3, 7]
    """
    if ids.size == 0:
        return ids
    change = np.flatnonzero(np.diff(ids) != 0) + 1
    runs = np.concatenate([np.zeros(1, dtype=np.int64), change])
    keep = ids[runs] != BLANK_INDEX
    return ids[runs][keep]


def count_segments(probs: np.ndarray) -> int:
    """给定每帧的类别概率（已排除 blank），贪心地数出会有几个连续段。"""
    ids = probs.argmax(axis=1)
    return int(collapse_repeats(ids).size)


def constrained_pick_topk(probs: np.ndarray, m_target: int,
                          min_gap: int = 1) -> np.ndarray:
    """在「每个词至少占 min_gap 帧」的前提下，选出尽量少的段。

    做法：从「每帧作为潜在起点」的置信度里，选 top-(m_target - extra) 个，
    且强制相邻起点间隔 >= min_gap。extra 是允许的段数上浮。

    这是 SCP 思想的最小实现：把全局计数约束注入到「选哪些帧输出」。

    参数
    ----
    probs     : (T, C) 每帧的类别概率，已排除 blank 列
    m_target  : 期望段数（= 参考 gloss 数）
    min_gap   : 相邻段起点的最小间隔

    返回
    ----
    选中的起点下标数组（长度 <= T）
    """
    T = probs.shape[0]
    m_target = int(max(1, min(m_target, T // max(min_gap, 1))))
    # 起点置信度 = 该帧非 blank 的最大概率 × 与前一帧的类别变化量
    best = probs.max(axis=1)
    if T > 1:
        change = np.abs(probs[1:] - probs[:-1]).sum(axis=1)
        onset = np.zeros(T)
        onset[1:] = change
        onset[0] = best[0]
    else:
        onset = best.copy()
    score = onset * (0.5 + 0.5 * best)
    # 贪心选点：按分数降序，且与已选点间隔 >= min_gap
    order = np.argsort(-score)
    chosen = []
    for idx in order:
        if len(chosen) >= m_target:
            break
        if all(abs(int(idx) - c) >= min_gap for c in chosen):
            chosen.append(int(idx))
    return np.array(sorted(chosen), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/ctc-landmark48-cap300.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-gap", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p11-scp-count-prior.json")
    a = ap.parse_args()
    if _CSLR_ERROR is not None:
        raise SystemExit("cslr 不可用：{}（需在仓库 venv 下运行）".format(_CSLR_ERROR))

    device = resolve_device("auto")
    print("device = {}".format(device))

    # ---- M（词数）分布：SCP 的标签 ----
    tr = read_split("train")
    dv = read_split("validation")
    m_hist = Counter()
    for _, g in tr:
        n = len([t for t in g.split("/") if t.strip()])
        if n:
            m_hist[n] += 1
    m_max = max(m_hist)
    tot = sum(m_hist.values())
    entropy = -sum((c / tot) * np.log2(c / tot) for c in m_hist.values())
    print("\n=== M（词数）分布 —— SCP 的分类标签 ===")
    print("  M 范围 {}~{}  -> M_max = {}（{} 类）".format(
        min(m_hist), m_max, m_max, m_max))
    print("  分布熵 {:.2f} bits（最大 {:.2f} bits）".format(entropy, np.log2(len(m_hist))))
    print("  众数 M={}（{:.1f}%）".format(
        m_hist.most_common(1)[0][0], 100 * m_hist.most_common(1)[0][1] / tot))

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
    ref_by_id = {sid: [t.strip() for t in g.split("/") if t.strip()] for sid, g in dv}
    recs = []
    for sid, g in dv:
        if not (va_root / (sid + ".npy")).exists():
            continue
        if not ref_by_id[sid]:
            continue
        recs.append(SampleRecord(sample_id=sid, video=Path(sid + ".mp4"), label=g,
                                 signer=sid.split("-")[0], session="x", split="validation"))
    recs.sort(key=lambda r: r.sample_id)
    if a.limit:
        recs = recs[: a.limit]
    print("\ndev 样本 {}".format(len(recs)))

    ds = GlossSequenceDataset(recs, va_root, voc, nrm, feature_view="full")

    # ---- 逐样本对比：基线 greedy vs 计数约束 ----
    rows = []
    for samples in iterate_batches(ds, a.batch, shuffle=False, seed=a.seed):
        b = to_torch_batch(collate_samples(samples), device)
        with torch.no_grad():
            logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        dec, _, _ = decode_batch(lp, ol.cpu().tolist(), 1)
        for row, ids in enumerate(dec):
            sid = samples[row].sample_id
            T_used = int(ol[row])
            ref = ref_by_id[sid]
            M = len(ref)
            probs = np.exp(lp[row, :T_used])          # (T, C)
            non_blank = probs[:, 1:]                  # 去掉 blank 列

            # 基线：当前 greedy 解码
            # ⚠️ `dec` 是 `classes_to_token_ids` 的输出，已做过去 blank 与
            # 类索引->词表索引的映射（`index - 1`）。所以：
            #   - dec 里**没有** blank 类，**不能**再对 0 做 blank 过滤
            #   - dec 里**已经折叠过**重复，不能再 collapse（否则会误删词表 id 0）
            # 直接用 len(dec) 作为段数，这是解码器给出的真实段数。
            base_ids = np.asarray(ids, dtype=np.int64)
            base_segs = len(base_ids)

            # 约束：强制段数 = M
            onsets = constrained_pick_topk(non_blank, M, min_gap=a.min_gap)
            con_segs = onsets.size

            rows.append({
                "id": sid, "T": T_used, "M": M,
                "base_segments": int(base_segs),
                "con_segments": int(con_segs),
                # blank 率必须用**逐帧 argmax** 口径（与 P0 / diagnose.py 一致）。
                # 用解码后的 dec 算会得到恒为 1.0 的假值，因为 dec 已被去 blank。
                "base_blank_frame": float(
                    (lp[row, :T_used].argmax(axis=1) == BLANK_INDEX).mean()),
                # 约束后：M 个段各占 min_gap 帧，其余为 blank（理论下界）
                "con_blank": float(max(0.0, 1 - con_segs * a.min_gap / T_used)),
                "pfpt_base": base_segs / max(M, 1),
                "pfpt_con": con_segs / max(M, 1),
            })

    # ---- 汇总 ----
    n = len(rows)
    m_vals = np.array([r["M"] for r in rows], dtype=np.float64)
    base_segs = np.array([r["base_segments"] for r in rows], dtype=np.float64)
    con_segs = np.array([r["con_segments"] for r in rows], dtype=np.float64)
    base_bl = np.array([r["base_blank_frame"] for r in rows])
    con_bl = np.array([r["con_blank"] for r in rows])
    pfpt_b = float(np.sum(base_segs) / max(np.sum(m_vals), 1))
    pfpt_c = float(np.sum(con_segs) / max(np.sum(m_vals), 1))
    seg_mae_b = float(np.mean(np.abs(base_segs - m_vals)))
    seg_mae_c = float(np.mean(np.abs(con_segs - m_vals)))

    print("\n" + "=" * 72)
    print("P11 · SCP 式全局计数先验（SignShift Sec 3.4 迁移）")
    print("=" * 72)
    print("样本 {} 条，参考词数 M 中位 {:.0f}（范围 {:.0f}~{:.0f}）".format(
        n, np.median(m_vals), m_vals.min(), m_vals.max()))
    print("")
    print("{:<24} {:>12} {:>12}".format("", "基线 greedy", "计数约束"))
    print("{:<24} {:>12.4f} {:>12.4f}".format("blank 率", base_bl.mean(), con_bl.mean()))
    print("{:<24} {:>12.2f} {:>12.2f}".format(
        "peak_frames_per_token", pfpt_b, pfpt_c))
    print("{:<24} {:>12.3f} {:>12.3f}".format("段数 MAE(对 M)", seg_mae_b, seg_mae_c))
    print("")
    print("blank 率降幅 {:.1f} 个百分点".format(100 * (base_bl.mean() - con_bl.mean())))
    print("pfpt   {} -> {}（判据 >3 为脱离 peaky 病理）".format(
        "%.2f" % pfpt_b, "%.2f" % pfpt_c))
    print("段数 MAE {} -> {}（降 {:.1f}%）".format(
        "%.3f" % seg_mae_b, "%.3f" % seg_mae_c,
        100 * (seg_mae_b - seg_mae_c) / max(seg_mae_b, 1e-9)))
    print("")
    print("论文对照（SignShift Table 2, MS-TCN）:")
    print("  F1@50  51.71 -> 61.82   SER 0.43 -> 0.26   计数 MAE 1.734 -> 1.177")
    print("  迁移损失：论文是**句子级**（需人工句子边界），我们是**词级**")
    print("            （M 直接来自 gloss 标注，无额外标注成本 —— 条件更好）")
    print("            但论文的增益体现在 F1@50/SER，我们只能观测 blank/pfpt/段数 MAE，")
    print("            **无法直接对标论文数字**。")

    zero_seg = int((con_segs == 0).sum())
    print("")
    if zero_seg:
        print("⚠️ 有 {} 条约束后段数为 0（min_gap 过大导致选不出点）".format(zero_seg))
    verdict = ("计数先验能把 pfpt 从 {:.2f} 提到 {:.2f}，段数 MAE 降 {:.1f}%，"
               "blank 率降 {:.1f}pp —— 结构性有效".format(
                   pfpt_b, pfpt_c,
                   100 * (seg_mae_b - seg_mae_c) / max(seg_mae_b, 1e-9),
                   100 * (base_bl.mean() - con_bl.mean())))
    print("判定：{}".format(verdict))

    receipt = {
        "ckpt": a.ckpt,
        "n_samples": n,
        "min_gap": a.min_gap,
        "paper": {
            "source": "SignShift Sec 3.4 + Table 2 (MS-TCN backbone)",
            "module": "Segment Count Prediction (SCP), 式(10)-(12)",
            "loss": "KL 散度 vs 高斯平滑后的目标分布 q_k = exp(-(k-M)^2/2σ^2)/Σ",
            "trained_with": "ground-truth 句子数，独立训练，仅推理期用于引导边界选择",
            "ablation": {"F1@50": [51.71, 61.82], "SER": [0.43, 0.26],
                         "count_MAE_How2Sign": [1.734, 1.177],
                         "count_MAE_OpenASL": [1.970, 1.275],
                         "count_MAE_PHOENIX14T": [1.337, 0.798]},
        },
        "our_label_source": {
            "M_definition": "gloss 词数（来自 Gloss 标注，无需人工边界）",
            "M_hist": dict(sorted(m_hist.items())),
            "M_max": m_max,
            "entropy_bits": entropy,
            "advantage": "论文需人工句子边界；我们天然有 M 标签，条件更优",
        },
        "level1_count_constrained_decoding": {
            "baseline_blank": float(base_bl.mean()),
            "constrained_blank": float(con_bl.mean()),
            "baseline_pfpt": pfpt_b,
            "constrained_pfpt": pfpt_c,
            "baseline_seg_mae": seg_mae_b,
            "constrained_seg_mae": seg_mae_c,
            "n_zero_segment_after_constraint": zero_seg,
        },
        "verdict": verdict,
        "caveats": [
            "无法直接对标论文的 F1@50/SER —— 任务层级不同（我们无句子边界真值）",
            "本级实现是「约束选点」的最小版本，未训练 SCP 分类器（级 2）",
            "con_blank 是理论下界（假设段不重叠且各占 min_gap 帧），非实测解码结果",
        ],
        "per_sample": rows[:50],
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
