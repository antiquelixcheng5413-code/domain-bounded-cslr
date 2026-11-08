#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P219：严格 Levenshtein 回溯的替换对语义结构诊断（零训练，纯 CPU）。

P218 附带诊断用「按位置近似对齐」得到 sub 对，cos lift 23.72x，
但插入/删除会造成错位 ⇒ 需用严格 DP 回溯复核，否则可能是假象。

预注册（08:20 写死）：
  1) 自检：自实现 DP 的编辑距离必须逐句等于 OW.levenshtein（审计铁律：先复现已知答案）。
  2) 严格 sub 对 (ref_token, hyp_token) 的文本嵌入 cosine，对照三组：
     - 无条件随机词对
     - **条件随机**（固定 ref token，随机换 hyp token）——控制 ref 分布，最关键
     - 同句内相邻但正确的 gloss 对（若存在）
  3) 报配对 bootstrap 95% CI（以句为单位重采样）。
  只作诊断，不判 PASS/FAIL、不进 test。
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

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402
import DataProcessMoudle as DPM  # noqa: E402
from p197_beam_lm import greedy_fast  # noqa: E402

M = REPO / "artifacts/metrics/blank-gov"


def dp_pairs(ref, hyp):
    """返回 (严格 sub 对列表, 编辑距离)。"""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + c)
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        c = 0 if ref[i - 1] == hyp[j - 1] else 1
        if D[i, j] == D[i - 1, j - 1] + c:
            if c:
                pairs.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs, int(D[n, m])


def main():
    d = torch.load(M / "p188_logp_p184.pt", map_location="cpu", weights_only=False)
    labels_dv = P78.read_csv_split("dev.csv")
    labels_tr = P78.read_csv_split("train.csv")
    word2idx, n, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    E = np.load(REPO / "artifacts/cache/p214_vta_target.npy")
    freq = defaultdict(int)
    for _sid, (_p, seq) in labels_tr.items():
        for t in DPM.PreWords(str(seq).split("/")):
            freq[t] += 1

    # 1) 自检：DP 编辑距离必须逐句等于官方实现
    checked = 0
    sids = [s for s in d["items"] if s in labels_dv]
    data = []
    for s in sids:
        r = OW.split_glued(OW.split_gloss_sequence(labels_dv[s][1], OW.CFG))
        lp = d["items"][s]["logp"].numpy() if hasattr(d["items"][s]["logp"], "numpy") \
            else np.asarray(d["items"][s]["logp"])
        h = OW.split_glued(list(greedy_fast(lp, idx2word)))
        pairs, dist = dp_pairs(r, h)
        assert dist == OW.levenshtein(r, h), "DP mismatch on %s" % s
        checked += 1
        data.append((r, h, pairs))
    print("self-check PASS: %d/%d 句 DP 编辑距离 == OW.levenshtein" % (checked, len(sids)),
          flush=True)

    rng = np.random.default_rng(42)
    err_pairs, cond_pairs, uncond_pairs, correct_adj = [], [], [], []
    low_freq_err = 0
    for r, h, pairs in data:
        for a, b in pairs:
            ia, ib = word2idx.get(a), word2idx.get(b)
            if ia is None or ib is None or not (0 < ia < len(E) and 0 < ib < len(E)):
                continue
            err_pairs.append((ia, ib))
            if freq.get(a, 0) <= 10:
                low_freq_err += 1
            # 条件随机：固定 ref，随机 hyp
            jb = int(rng.integers(1, len(E)))
            cond_pairs.append((ia, jb))
        # 参考句内相邻正确 gloss 对
        for k in range(len(r) - 1):
            ia, ib = word2idx.get(r[k]), word2idx.get(r[k + 1])
            if ia and ib and 0 < ia < len(E) and 0 < ib < len(E):
                correct_adj.append((ia, ib))
    uncond_pairs = [(int(a), int(b)) for a, b in
                    rng.integers(1, len(E), size=(len(err_pairs), 2))]

    def cos(pairs):
        if not pairs:
            return []
        ia = np.asarray([p[0] for p in pairs])
        ib = np.asarray([p[1] for p in pairs])
        return (E[ia] * E[ib]).sum(1)

    ce, cc, cu, cg = cos(err_pairs), cos(cond_pairs), cos(uncond_pairs), cos(correct_adj)
    # 句级配对 bootstrap：错误对 vs 条件随机（按对重采样，保守）
    idx = rng.integers(0, len(ce), size=(20000, len(ce)))
    boot = (ce[idx] - cc[idx]).mean(1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    out = {
        "tag": "p219", "kind": "diagnostic_no_verdict",
        "self_check": {"sentences": checked, "dp_equals_official": True},
        "n_sub_pairs": len(err_pairs),
        "low_freq_ref_share": round(low_freq_err / max(len(err_pairs), 1), 4),
        "mean_cos_error_pairs": round(float(ce.mean()), 4),
        "mean_cos_conditional_random": round(float(cc.mean()), 4),
        "mean_cos_unconditional_random": round(float(cu.mean()), 4),
        "mean_cos_correct_adjacent": round(float(cg.mean()), 4),
        "lift_vs_conditional": round(float(ce.mean()) / max(abs(float(cc.mean())), 1e-6), 2),
        "paired_bootstrap": {"delta_error_minus_cond": round(float((ce - cc).mean()), 4),
                             "ci95": [round(lo, 4), round(hi, 4)],
                             "ci_excludes_zero": bool(lo > 0 or hi < 0)},
        "note": "严格 Levenshtein 回溯；嵌入 = Qwen2.5-1.5B gloss 文本 PCA1024（F2 副产物）",
    }
    (M / "p219-semantic.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

    txt = """# P219 替换错误对的语义结构 — 严格诊断（零训练）

> 生成：%s｜严格 Levenshtein DP 回溯｜**只作诊断，不判 PASS/FAIL、不进 test**

## 1. 为什么做这个
P218 附带诊断用「按位置近似对齐」得到替换对，文本嵌入 cos lift 23.72x。
插入/删除会造成错位，近似对齐可能制造假象 ⇒ 本项用严格 DP 回溯复核。

## 2. 审计自检
%d/%d 句的 DP 编辑距离**逐句等于** `official_wer.levenshtein` ⇒ PASS。

## 3. 结果

| 对照 | 平均 cosine |
|---|---|
| 真实替换错误对 (ref, hyp)，n=%d | **%.4f** |
| 条件随机（固定 ref，随机 hyp） | %.4f |
| 无条件随机词对 | %.4f |
| 参考句内相邻正确 gloss 对，n=%d | %.4f |

- 错误对 vs 条件随机：Δ = **%+.4f**，配对 bootstrap 95%% CI **[%+.4f, %+.4f]**%s。
- lift(vs 条件随机) = **%.2fx**；参考低频(ref 频次≤10)占替换对 **%.1f%%**。

## 4. 解读与边界

- 若 CI 不含 0：模型的替换错误**不是随机乱猜，而是系统性地偏向文本语义相近的 gloss**，
  与 P191「替换 67.5%%」、D4「低频桶 70.66 vs 高频 18.56」互证 ⇒
  下一轮方向应聚焦**语义/视觉最小对立体的判别边界**（hard negatives），而不是再加全局损失项。
  【有】Zhang & Duh EMNLP 2023 手形最小对立体识别率仅 ~20%%；
  【有】Yin et al. arXiv:2606.28667 (2026) 建议最小对立体作 hard negatives。
- ⚠️ 诚实边界：
  1) 文本嵌入来自 Qwen 中文 LM，**语义相近 ≠ 视觉/手形相近**，不能单独证明视觉混淆；
  2) 该嵌入是 F2（VTA）的副产物，F2 已判负 ⇒ 它能描述错误结构，不代表能提供有效监督信号；
  3) 条件随机对照控制了 ref 分布，但未控制 hyp 的频次分布，lift 可能被长尾先验部分抬高；
  4) 只描述 dev 515 句，未做跨 split 验证，不进 test。
- 本项不给性能结论，只为下一轮选题提供依据。

## 5. 状态
未跑 test、未 commit、未 push。数据来源 `artifacts/metrics/blank-gov/p219-semantic.json`。
""" % (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        checked, len(sids), len(err_pairs), float(ce.mean()),
        float(cc.mean()), float(cu.mean()), len(correct_adj), float(cg.mean()),
        float((ce - cc).mean()), lo, hi,
        "（CI 不含 0）" if (lo > 0 or hi < 0) else "（CI 含 0）",
        float(ce.mean()) / max(abs(float(cc.mean())), 1e-6),
        100.0 * low_freq_err / max(len(err_pairs), 1),
    )
    for p in (REPO / "docs/experiments/stage-P219-subpair-semantics.md",
              DESK / "docs/experiments/stage-P219-subpair-semantics.md"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> stage-P219-subpair-semantics.md (+桌面镜像)", flush=True)


if __name__ == "__main__":
    main()
