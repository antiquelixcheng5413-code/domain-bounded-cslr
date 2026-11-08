#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""N2 / L1：spike 邻域后验融合扫描（零训练、纯 CPU、零梯度）。

判据与网格见 docs/planning/N2-PREREG-2026-11-08.md（20:55 写死，禁止事后修改）。
数据：p188_logp_p184.pt（p184 模型 dev 515 句后验，与 H1/H2-a 的 34.34 同管线同句序）。
只做 dev，不碰 test；不加载模型、不改任何权重。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
DESK = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78       # noqa: E402
import official_wer as OW                    # noqa: E402
import DataProcessMoudle as DPM              # noqa: E402
from p197_beam_lm import greedy_fast         # noqa: E402
from p210_d4_eval import align_ops           # noqa: E402
from night_judge import paired_bootstrap     # noqa: E402

M = REPO / "artifacts/metrics/blank-gov"
TAG = "p234-n2"
KS = (1, 2, 4)
LAMS = (0.3, 0.6)
RULE_DELTA = 2.00
AUDIT_RANGE = (34.20, 34.60)
R3_HIGH, R5_INS_REL, R6_MICRO = 1.00, 1.10, 0.30


def smooth(lp, k, lam):
    """prob 域邻域均值融合：p' = (1-lam)p + lam*mean(p[t-k..t+k])。blank 通道同样参与。"""
    p = np.exp(lp.astype(np.float64))
    T = p.shape[0]
    pad = np.pad(p, ((k, k), (0, 0)), mode="edge")
    acc = np.zeros_like(p)
    for j in range(2 * k + 1):
        acc += pad[j:j + T]
    p2 = (1.0 - lam) * p + lam * acc / (2 * k + 1)
    return np.log(np.maximum(p2, 1e-12)).astype(np.float32)


def main():
    labels_tr = P78.read_csv_split("train.csv")
    labels_dv = P78.read_csv_split("dev.csv")
    word2idx, n, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")

    freq = defaultdict(int)
    for _sid, (_p, seq) in labels_tr.items():
        for t in DPM.PreWords(str(seq).split("/")):
            freq[t] += 1

    d = torch.load(M / "p188_logp_p184.pt", map_location="cpu", weights_only=False)
    sids = [s for s in d["items"] if s in labels_dv]
    print("dev sentences with posterior =", len(sids), flush=True)
    assert len(sids) == 515, "后验句数不是 515，管线异常"

    raw_refs = [labels_dv[s][1] for s in sids]
    refs = [OW.split_glued(OW.split_gloss_sequence(labels_dv[s][1], OW.CFG))
            for s in sids]
    base_lp = {}
    for s in sids:
        lp = d["items"][s]["logp"]
        base_lp[s] = (lp.numpy() if hasattr(lp, "numpy") else np.asarray(lp)
                      ).astype(np.float32)

    def metrics(hyps):
        """口径逐行照 p226_h1_knn.metrics（含 sid / low_n / low_err）。"""
        per_cls = defaultdict(lambda: {"n": 0, "err": 0})
        lo, hi = {"n": 0, "err": 0}, {"n": 0, "err": 0}
        ins = sub = dele = nref = 0
        per_item = []
        for ix, (r, h) in enumerate(zip(refs, hyps)):
            hh = OW.split_glued(list(h))
            ops, ii = align_ops(r, hh)      # ii = 插入数；句序索引必须用 ix
            for t, op in zip(r, ops):
                bk = lo if freq.get(t, 0) <= 10 else hi
                per_cls[t]["n"] += 1
                per_cls[t]["err"] += (op != "match")
                bk["n"] += 1
                bk["err"] += (op != "match")
            s_ = sum(o == "sub" for o in ops)
            d_ = sum(o == "del" for o in ops)
            ins += ii; sub += s_; dele += d_; nref += len(r)
            _ln = sum(1 for t in r if freq.get(t, 0) <= 10)
            _le = sum(1 for t, op in zip(r, ops)
                      if freq.get(t, 0) <= 10 and op != "match")
            per_item.append({"sid": sids[ix], "ref_n": len(r),
                             "edits": OW.levenshtein(r, hh), "ins": ii,
                             "sub": s_, "del": d_, "low_n": _ln, "low_err": _le})
        cls_rates = [v["err"] / v["n"] for v in per_cls.values() if v["n"] > 0]
        return {
            "WER_official": OW.evaluate(raw_refs, hyps)["WER_official"],
            "uWER_pct": round(float(np.mean(cls_rates)) * 100, 2),
            "low_bucket": {"n_tokens": lo["n"],
                           "WER_pct": round(100 * lo["err"] / max(lo["n"], 1), 2)},
            "high_bucket": {"n_tokens": hi["n"],
                            "WER_pct": round(100 * hi["err"] / max(hi["n"], 1), 2)},
            "edits": {"ins": int(ins), "sub": int(sub), "del": int(dele),
                      "n_ref_tokens": nref,
                      "micro_subdel_pct": round(100 * (sub + dele) / max(nref, 1), 2),
                      "ins_rate_pct": round(100 * ins / max(nref, 1), 2)},
            "per_item": per_item,
            "n": len(sids), "tag": TAG,
        }

    def decode(lp_map):
        return [greedy_fast(lp_map[s], idx2word) for s in sids]

    # ---- λ=0 基线（审计硬门禁） ----
    base_rep = metrics(decode(base_lp))
    base_wer = base_rep["WER_official"]
    audit_ok = AUDIT_RANGE[0] <= base_wer <= AUDIT_RANGE[1]
    print("lambda=0 baseline WER = %.2f  audit=%s" % (base_wer, audit_ok), flush=True)
    (M / (TAG + "-base-uwer.json")).write_text(
        json.dumps(base_rep, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 6 档扫描 ----
    grid, reps = {}, {}
    for k in KS:
        for lam in LAMS:
            name = "k%d_lam%g" % (k, lam)
            lp_map = {s: smooth(base_lp[s], k, lam) for s in sids}
            rep = metrics(decode(lp_map))
            reps[name] = rep
            hi_diff = rep["high_bucket"]["WER_pct"] - base_rep["high_bucket"]["WER_pct"]
            ins_rel = rep["edits"]["ins"] / max(base_rep["edits"]["ins"], 1)
            micro_diff = (rep["edits"]["micro_subdel_pct"]
                          - base_rep["edits"]["micro_subdel_pct"])
            R3 = hi_diff <= R3_HIGH
            R5 = ins_rel <= R5_INS_REL
            R6 = micro_diff <= R6_MICRO
            delta = base_wer - rep["WER_official"]
            point, lo_ci, hi_ci = paired_bootstrap(base_rep, rep)
            grid[name] = {
                "k": k, "lambda": lam,
                "WER_official": rep["WER_official"], "delta_pp": round(delta, 3),
                "uWER_pct": rep["uWER_pct"],
                "low_bucket_WER": rep["low_bucket"]["WER_pct"],
                "high_bucket_WER": rep["high_bucket"]["WER_pct"],
                "ins": rep["edits"]["ins"], "del": rep["edits"]["del"],
                "micro_subdel_pct": rep["edits"]["micro_subdel_pct"],
                "paired_delta_pp": point, "ci95": [lo_ci, hi_ci],
                "R3_high": bool(R3), "R5_ins": bool(R5), "R6_micro": bool(R6),
                "guard_pass": bool(R3 and R5 and R6),
            }
            (M / (TAG + "-" + name + "-uwer.json")).write_text(
                json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            print("%-10s WER %.2f  Δ%+.2f  CI[%.2f,%.2f]  guard=%s"
                  % (name, rep["WER_official"], delta, lo_ci, hi_ci,
                     grid[name]["guard_pass"]), flush=True)

    # ---- 选择规则（写死）：护栏内 WER 最低 ----
    eligible = [g for g in grid.values() if g["guard_pass"]]
    if not eligible:
        chosen_name, chosen = None, None
    else:
        best = min(eligible, key=lambda x: x["WER_official"])
        chosen_name = "k%d_lam%g" % (best["k"], best["lambda"])
        chosen = best

    if chosen is None:
        verdict = "FAIL"
        delta, ci = 0.0, [0.0, 0.0]
        reason = "无任一档通过护栏 R3/R5/R6，按预注册强制 λ=0 并判 FAIL"
    else:
        delta, ci = chosen["delta_pp"], chosen["ci95"]
        verdict = ("SIGNAL" if (delta >= RULE_DELTA and ci[0] > 0) else "FAIL")
        reason = ("Δ=%.2fpp %s 2.00pp 且 CI 下界 %.3f %s 0"
                  % (delta, ">=" if delta >= RULE_DELTA else "<",
                     ci[0], ">" if ci[0] > 0 else "<="))
        (M / (TAG + "-uwer.json")).write_text(
            json.dumps(reps[chosen_name], ensure_ascii=False, indent=2),
            encoding="utf-8")
    if not audit_ok:
        verdict = "INVALID"
        reason += "；⚠️ λ=0 基线 %.2f 不在 [34.20,34.60]，整个 L1 标 INVALID" % base_wer

    (M / (TAG + "-verdict.txt")).write_text(verdict, encoding="utf-8")
    out = {
        "tag": TAG, "prereg": "docs/planning/N2-PREREG-2026-11-08.md (20:55)",
        "verdict": verdict, "reason": reason,
        "audit_clean_in_range": bool(audit_ok), "base_WER_official": base_wer,
        "chosen": chosen_name, "chosen_delta_pp": delta, "chosen_ci95": ci,
        "grid": grid,
        "rules": {"delta_min_pp": RULE_DELTA, "audit_range": AUDIT_RANGE,
                  "R3_high_max_pp": R3_HIGH, "R5_ins_rel_max": R5_INS_REL,
                  "R6_micro_max_pp": R6_MICRO,
                  "selection": "guard-pass 内 WER 最低；否则 λ=0 直接 FAIL"},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (M / (TAG + "-probe.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 报告（WSL + 桌面） ----
    rows = []
    for name, g in grid.items():
        rows.append("| `%s` | %.2f | %+.2f | %.2f | %.2f | %.2f | %d | [%.3f, %.3f] | %s |"
                    % (name, g["WER_official"], g["delta_pp"], g["uWER_pct"],
                       g["low_bucket_WER"], g["high_bucket_WER"], g["ins"],
                       g["ci95"][0], g["ci95"][1],
                       "✓" if g["guard_pass"] else "✗"))
    txt = """# P234 / N2-L1 spike 邻域后验融合扫描 — 阶段报告

> 日期：%s｜判定：**%s**｜预注册 `N2-PREREG-2026-11-08.md`（20:55 写死，早于 K1 结果）
> 零训练、纯 CPU、不加载模型、不碰 test。

## 1. 立论
D3/I0 证明「每词仅 1.06 个非 blank 帧被监督」，且 I0 证明重复**输入帧**只拉长同一 spike。
L1 检验尚未验证的一点：spike **邻域帧的后验**里是否含可用于判别的信息。
与已判死项区别：P188 是跨模型 logit 集成、I0 是改输入时间分辨率、P187 是输入变换 TTA；
L1 是同一后验的时序邻域聚合。【无】CSLR 无先例；【有】本地 I0 的反证据优先写明。

## 2. 方法（固定）
`p'[t] = (1−λ)p[t] + λ·mean_{j=−k..k} p[t+j]`（prob 域、edge padding、blank 通道同样参与），
后验来自 `p188_logp_p184.pt`（dev 515 句、与 34.34 同管线同句序）。
网格 `k∈{1,2,4} × λ∈{0.3,0.6}` 共 6 档 + λ=0 基线；指标口径逐行照 `p226_h1_knn.metrics`。

## 3. 结果
λ=0 基线 WER = **%.2f**（审计区间 [34.20,34.60] ⇒ **%s**）。

| 档 | WER | Δ | uWER | 低频桶 | 高频桶 | ins | 配对 CI 95%% | 护栏 |
|---|---|---|---|---|---|---|---|---|
| `λ=0` | %.2f | — | %.2f | %.2f | %.2f | %d | — | — |
%s

**chosen** = `%s`；**判定 = %s**（%s）。
护栏 R3 高频桶恶化≤1pp / R5 插入增幅≤10%% / R6 micro 恶化≤0.30pp；
主判据 Δ≥2.00pp 且配对 bootstrap CI 下界>0。SIGNAL 也不进 test、不自动启动训练。

## 4. 机制结论
%s

## 5. 诚实边界
- 本扫描只覆盖 6 个离散点，不能排除更细的邻域权重；但 I0 已表明邻域帧只是同一 spike 的复制，
  且 P188 集成判负，先验上 L1 落 FAIL 的概率高。
- 判据、网格、选择规则、审计区间均在 20:55 写死，**未事后调整**；未跑 test、未改任何权重。
- 产物：`p234-n2-probe.json`、`p234-n2-verdict.txt`、6 档 + 基线 `-uwer.json`。
""" % (
        datetime.now().strftime("%Y-%m-%d %H:%M"), verdict,
        base_wer, "PASS" if audit_ok else "FAIL",
        base_wer, base_rep["uWER_pct"], base_rep["low_bucket"]["WER_pct"],
        base_rep["high_bucket"]["WER_pct"], base_rep["edits"]["ins"],
        "\n".join(rows), chosen_name or "（无护栏通过）", verdict, reason,
        ("**SIGNAL**：邻域后验确含可用判别信息，值得另写预注册设计训练期时序证据聚合。"
         if verdict == "SIGNAL" else
         "**FAIL**：邻域后验没有带来超出 CTC 自身 spike 的新信息，与 I0「重复帧只拉长同一 spike」"
         "、P188「同源集成无增益」互证 ⇒ 推理期时序平滑这条线关闭。"))
    for base in (REPO, DESK):
        p = base / "docs/experiments/stage-P234-N2-spike-fusion.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("verdict =", verdict, "|", reason, flush=True)
    print("报告 -> stage-P234-N2-spike-fusion.md (+桌面镜像)", flush=True)


if __name__ == "__main__":
    main()
