#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P228 / I1：零成本 CPU 诊断 —— 删除错误与 T_conv/L 的相关性。

判据见 docs/planning/I0-PREREG-2026-11-08.md §6（19:05 写死）：
Spearman |rho| >= 0.20 且 20000 次 bootstrap 95% CI 不含 0 ⇒ 相关。
只用已落盘数据，不占 GPU、不产生候选模型、不改变 I0 判定。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(p))

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402

M = REPO / "artifacts/metrics/blank-gov"
TAG = "p228-i1"
RULE_RHO = 0.20
N_BOOT = 20000
SEED = 42


def spearman(x, y):
    """不依赖 scipy：秩相关 = Pearson(秩)。"""
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        # 处理并列：同值取平均秩
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        r = (sums / cnt)[inv]
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def main():
    d = torch.load(M / "p188_logp_p184.pt", map_location="cpu", weights_only=False)
    clean = json.loads((M / "p227-h2a-clean-uwer.json").read_text(encoding="utf-8"))
    labels = P78.read_csv_split("dev.csv")

    tmap = {sid: it["logp"].shape[0] for sid, it in d["items"].items()}
    lmap = {sid: len(OW.split_glued(OW.split_gloss_sequence(g, OW.CFG)))
            for sid, (_tr, g) in labels.items()}

    sids, tl, delrate, nb_ratio = [], [], [], []
    for it in clean["per_item"]:
        sid = str(it["sid"])
        if sid not in tmap or sid not in lmap or lmap[sid] == 0:
            continue
        sids.append(sid)
        tl.append(tmap[sid] / lmap[sid])
        delrate.append(it["del"] / max(it["ref_n"], 1))
        nb_ratio.append(it["sub"] / max(it["ref_n"], 1))   # 附报：替换率

    tl = np.asarray(tl, float)
    dr = np.asarray(delrate, float)
    sr = np.asarray(nb_ratio, float)
    assert len(sids) == 515, "期望 515 句，实际 %d" % len(sids)
    assert len(set(sids)) == 515, "sid 不唯一"

    rho = spearman(tl, dr)
    rho_sub = spearman(tl, sr)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(tl), size=(N_BOOT, len(tl)))
    boots = np.array([spearman(tl[i], dr[i]) for i in idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ci_excludes_zero = bool(lo > 0 or hi < 0)
    related = bool(abs(rho) >= RULE_RHO and ci_excludes_zero)

    # 分桶描述（低 T/L vs 高 T/L，各一半）
    med = np.median(tl)
    low_mask, high_mask = tl <= med, tl > med
    out = {
        "tag": TAG,
        "n": len(sids),
        "rule": "Spearman |rho| >= 0.20 且 bootstrap 95% CI 不含 0",
        "rho_TL_vs_del_rate": round(rho, 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "ci_excludes_zero": ci_excludes_zero,
        "related": related,
        "verdict": "RELATED" if related else "NOT_RELATED",
        "rho_TL_vs_sub_rate": round(rho_sub, 4),
        "T_conv_per_gloss": {
            "p10": round(float(np.percentile(tl, 10)), 2),
            "median": round(float(med), 2),
            "mean": round(float(tl.mean()), 2),
            "p90": round(float(np.percentile(tl, 90)), 2),
        },
        "del_rate_mean_pct": round(float(dr.mean()) * 100, 2),
        "buckets": {
            "low_TL_half": {"n": int(low_mask.sum()),
                            "del_rate_pct": round(float(dr[low_mask].mean()) * 100, 2)},
            "high_TL_half": {"n": int(high_mask.sum()),
                             "del_rate_pct": round(float(dr[high_mask].mean()) * 100, 2)},
        },
        "note": ("解释性诊断，不改变 I0 判定；del_rate 用 p227-h2a-clean 的逐句 del/ref_n，"
                 "T_conv 用 p188 后验长度，L 用 dev 参考 gloss 数"),
    }
    (M / (TAG + "-correlation.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("n=%d rho=%.4f CI[%.4f, %.4f] => %s"
          % (len(sids), rho, lo, hi, out["verdict"]), flush=True)
    print("T/L median %.2f, del_rate low/high half = %.2f / %.2f %%"
          % (med, out["buckets"]["low_TL_half"]["del_rate_pct"],
             out["buckets"]["high_TL_half"]["del_rate_pct"]), flush=True)
    print("rho(T/L, sub_rate) = %.4f" % rho_sub, flush=True)


if __name__ == "__main__":
    main()
