# -*- coding: utf-8 -*-
"""P1 验证：TS²-TFD 免训练边界检测能否在 48×368 landmark 上产出可用伪边界。

只读特征与 dev 标签，不训练、不写 model artifact。

关键校验（论文 TS²-TFD Table 5 的口径）：
  - 伪边界数 vs 真值 gloss 数（论文欠分割偏置：预测 9.90 vs GT 10.28）
  - 若有人工边界则算 mF1B @ 容差 1-4 帧；本项目无边界标注，
    故用「TFD 峰数 / gloss 数」与「峰数是否落在合理区间」作代理指标。
  - 复现论文 Sec 4.4.9 的跨数据集迁移崩溃（T 过大 -> 边界退化）

用法:
  ./venv/bin/python tools/blank_gov/diag_tfd.py --limit 100
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "src"))

import tfd as TFD
import bio_labels as BIO
from cslr.recognition.model import ctc_config_from_dict  # noqa: F401  (确认环境可导入)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-root", default="artifacts/part3_features/validation")
    ap.add_argument("--labels", default="data/raw/CE-CSL/label/dev.csv")
    ap.add_argument("--dim", type=int, default=368)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p1-tfd-probe.json")
    a = ap.parse_args()

    import csv

    rows = []
    with open(REPO / a.labels, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        idcol = next((c for c in cols if c.lower() in ("number", "id", "name")), cols[0])
        gcol = next((c for c in cols if "gloss" in c.lower() or "label" in c.lower()), cols[-1])
        for r in rd:
            rows.append((r[idcol], r[gcol]))
    n_gloss_by_id = {rid: len([t for t in g.split("/") if t.strip()]) for rid, g in rows}

    feat_dir = REPO / a.feat_root
    feat_map = {}
    for p in sorted(feat_dir.glob("*.npy")):
        if ".receipt." in p.name:
            continue
        try:
            shp = np.load(p).shape
        except Exception:
            continue
        if shp and shp[-1] == a.dim:
            stem = p.stem.split(".landmark")[0].split(".clip")[0]
            feat_map[stem] = p

    keys = [k for k in feat_map if k in n_gloss_by_id]
    keys.sort()
    keys = keys[: a.limit]
    print("评估 {} 条 dev 样本（特征 {} 维）".format(len(keys), a.dim))

    per = []
    auto_ratios = []
    for k in keys:
        f = np.load(feat_map[k]).astype(np.float64)
        T, K = f.shape[0], n_gloss_by_id[k]
        t_auto, m_auto = TFD.suggest_TM(T, K)
        pk_auto = TFD.detect_boundaries(f, t=t_auto, m=m_auto, metric="l2")

        y = BIO.labels_from_boundaries(T, pk_auto, dilate_k=1)
        dist = BIO.label_distribution(y)

        auto_ratios.append(len(pk_auto) / max(K, 1))
        per.append(
            {
                "id": k,
                "T": T,
                "n_gloss": K,
                "t_auto": t_auto,
                "m_auto": m_auto,
                "n_peaks": int(pk_auto.size),
                "peak_count_ratio": float(len(pk_auto) / max(K, 1)),
                "label_dist": dist,
            }
        )

    # 跨数据集迁移崩溃复现（论文 Sec 4.4.9：T 过大 -> AmF1B 崩到 3.88）
    crash = []
    for k in keys[:20]:
        f = np.load(feat_map[k]).astype(np.float64)
        T, K = f.shape[0], n_gloss_by_id[k]
        t_ok, m_ok = TFD.suggest_TM(T, K)
        n_ok = TFD.detect_boundaries(f, t=t_ok, m=m_ok, metric="l2").size
        n_bad = TFD.detect_boundaries(f, t=25, m=17, metric="l2").size
        crash.append({"id": k, "n_peaks_auto": int(n_ok), "n_peaks_T25": int(n_bad)})

    ar = np.asarray(auto_ratios, dtype=np.float64)
    print("")
    print("=" * 62)
    print("P1 TFD 伪边界验证")
    print("=" * 62)
    print("T 自动估计      中位 {}".format(int(np.median([p["t_auto"] for p in per]))))
    print("M 自动估计      中位 {}".format(int(np.median([p["m_auto"] for p in per]))))
    print("T/ref_gloss     中位 {:.2f}".format(
        np.median([p["T"] / max(p["n_gloss"], 1) for p in per])))
    print("")
    print("峰数/gloss 比   中位 {:.2f}  (论文欠分割偏置目标 <1)".format(np.median(ar)))
    print("峰数/gloss 比   范围 {:.2f} ~ {:.2f}".format(ar.min(), ar.max()))
    print("落在 0.3~2.0 区间的样本比例  {:.1%}".format(float(((ar >= 0.3) & (ar <= 2.0)).mean())))

    bd = np.asarray([p["label_dist"]["B"] for p in per])
    io = np.asarray([p["label_dist"]["I"] for p in per])
    oo = np.asarray([p["label_dist"]["O"] for p in per])
    print("")
    print("BIO 标签占比    B 中位 {:.1%} / I 中位 {:.1%} / O 中位 {:.1%}".format(
        np.median(bd), np.median(io), np.median(oo)))
    print("  EMNLP 论文 sign 任务参考比例 B:I:O = 1:5:18")
    print("  O 占比若远高于 75%，说明伪边界偏少，模型会退化为全 O 平凡解")

    # 论文的崩溃现象是「误检出大量假边界」，不是「检不出边界」。
    # TS²-TFD Sec 4.4.9：T 过大时 AmF1B 崩到 3.88，原因是 TFD 信号被抹平、
    # 局部极大值退化成噪声触发，产生大量假峰。所以判据是峰数**膨胀**。
    crash_ok = sum(1 for c in crash if c["n_peaks_T25"] > c["n_peaks_auto"])
    print("")
    print("跨 T 迁移对比（复现论文 Sec 4.4.9 崩溃现象）")
    print("  T=25 峰数 > auto 峰数（假边界膨胀）的样本  {}/{}".format(crash_ok, len(crash)))
    tot_auto = sum(c["n_peaks_auto"] for c in crash)
    tot_bad = sum(c["n_peaks_T25"] for c in crash)
    print("  合计峰数  auto={}  T25={}".format(tot_auto, tot_bad))
    if tot_bad > tot_auto:
        print("  -> 复现成功：T 过大时假边界膨胀，与论文一致（AmF1B 崩到 3.88）")
        print("  -> 结论：T 必须在本数据集上重新估计，不可跨数据集照搬")
    else:
        print("  -> 未复现膨胀：本数据集 T=25 未产生假边界，")
        print("     但这不构成「可跨数据集迁移」的证据 —— 需在有边界标注的数据上验证")

    rep = {
        "n_samples": len(keys),
        "feature_dim": a.dim,
        "peak_count_ratio_median": float(np.median(ar)),
        "peak_count_ratio_min": float(ar.min()),
        "peak_count_ratio_max": float(ar.max()),
        "in_range_0.3_2.0_frac": float(((ar >= 0.3) & (ar <= 2.0)).mean()),
        "label_B_median": float(np.median(bd)),
        "label_I_median": float(np.median(io)),
        "label_O_median": float(np.median(oo)),
        "crash_check": crash,
        "per_sample": per,
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print("收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
