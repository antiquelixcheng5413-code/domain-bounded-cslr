#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P241 / N1：同义等价评估口径诊断（零训练、纯 CPU、只读已落盘产物）。

预注册：docs/planning/N1-PREREG-2026-11-09.md（08:14 写死）。
**评估口径诊断，不是性能主张**；不进 test、不改判据、不用于模型选择。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
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

M = REPO / "artifacts/metrics/blank-gov"
CACHE = REPO / "artifacts/cache"
TAG = "p241-n1"

PUNCT = set("？。！，、；：…·,.!?;:『』「」（）()")
COS_TH = 0.75
AUDIT_RANGE = (34.20, 34.60)
KNOWN_SUB_PAIRS = 693


def main():
    labels_tr = P78.read_csv_split("train.csv")
    labels_dv = P78.read_csv_split("dev.csv")
    word2idx, n, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")

    freq = Counter(t for _p, g in labels_tr.values()
                   for t in DPM.PreWords(str(g).split("/")))

    X = np.load(CACHE / "p214_vta_target.npy").astype(np.float32)
    mask = np.load(CACHE / "p214_vta_mask.npy").astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    def gcos(a, b):
        ia, ib = word2idx.get(a), word2idx.get(b)
        if ia is None or ib is None or ia >= len(Xn) or ib >= len(Xn):
            return -1.0
        if mask[ia] <= 0 or mask[ib] <= 0:
            return -1.0
        return float(Xn[ia] @ Xn[ib])

    def equiv(a, b):
        if a == b:
            return None
        e1 = (a in PUNCT and b in PUNCT)
        e2 = (a in b or b in a) and len(a) != len(b)
        e3 = gcos(a, b) >= COS_TH
        return (bool(e1), bool(e2), bool(e3))

    d = torch.load(M / "p188_logp_p184.pt", map_location="cpu", weights_only=False)
    sids = [s for s in d["items"] if s in labels_dv]
    assert len(sids) == 515

    raw_refs = [labels_dv[s][1] for s in sids]
    refs = [OW.split_glued(OW.split_gloss_sequence(labels_dv[s][1], OW.CFG))
            for s in sids]
    hyps = []
    for s in sids:
        lp = d["items"][s]["logp"]
        lp = lp.numpy() if hasattr(lp, "numpy") else np.asarray(lp)
        hyps.append(greedy_fast(lp, idx2word))

    # ---- 逐句严格对齐 ----
    sub_pairs = []
    bucket = {k: {"tokens": 0, "sub": 0, "del": 0, "ins": 0,
                  "e1": 0, "e2": 0, "e3": 0} for k in ("punct", "low", "high")}
    tot_edits, tot_ref = 0, 0
    e_counts = [0, 0, 0]
    eq_union = 0
    for r, h in zip(refs, hyps):
        hh = OW.split_glued(list(h))
        ops, ins = align_ops(r, hh)
        assert len(ops) == len(r)
        # align_ops 的 ops 与 r 对齐；sub 位对应的 hyp token 需按顺序取
        # 采用与 P219 一致的严格 DP 回溯来拿 (ref,hyp) 对
        pairs, dp_dist = dp_pairs(r, hh)
        assert dp_dist == OW.levenshtein(r, hh), "DP 回溯与官方编辑距离不一致"
        sub_pairs.extend(pairs)
        tot_edits += OW.levenshtein(r, hh)
        tot_ref += len(r)
        for t, op in zip(r, ops):
            k = "punct" if t in PUNCT else ("low" if freq.get(t, 0) <= 10 else "high")
            bucket[k]["tokens"] += 1
            if op == "sub":
                bucket[k]["sub"] += 1
            elif op == "del":
                bucket[k]["del"] += 1
        bucket["punct"]["ins"] += 0
        for a, b in pairs:
            e = equiv(a, b)
            if e is None:
                continue
            k = "punct" if a in PUNCT else ("low" if freq.get(a, 0) <= 10 else "high")
            for j in range(3):
                if e[j]:
                    e_counts[j] += 1
                    bucket[k][("e1", "e2", "e3")[j]] += 1
            if any(e):
                eq_union += 1

    base_wer = 100.0 * tot_edits / max(tot_ref, 1)
    audit_ok = AUDIT_RANGE[0] <= base_wer <= AUDIT_RANGE[1] and len(sub_pairs) == KNOWN_SUB_PAIRS
    print("self-check: WER %.2f  n_sub_pairs %d  audit=%s"
          % (base_wer, len(sub_pairs), audit_ok), flush=True)

    rules = {}
    for j, name in enumerate(("E1_punct", "E2_substring", "E3_cos075")):
        w = 100.0 * (tot_edits - e_counts[j]) / max(tot_ref, 1)
        rules[name] = {"n_equiv_sub": e_counts[j],
                       "share_of_sub_pct": round(100 * e_counts[j] / max(len(sub_pairs), 1), 2),
                       "WER_synonym_aware": round(w, 2),
                       "delta_vs_official_pp": round(base_wer - w, 2)}
    union_w = 100.0 * (tot_edits - eq_union) / max(tot_ref, 1)
    rules["UNION_upper_bound"] = {"n_equiv_sub": eq_union,
                                  "share_of_sub_pct": round(100 * eq_union / max(len(sub_pairs), 1), 2),
                                  "WER_synonym_aware": round(union_w, 2),
                                  "delta_vs_official_pp": round(base_wer - union_w, 2)}

    top = Counter(sub_pairs).most_common(12)
    qp = sum(c for (a, b), c in Counter(sub_pairs).items()
             if {a, b} == {"？", "。"})
    out = {
        "tag": TAG, "prereg": "docs/planning/N1-PREREG-2026-11-09.md (08:14)",
        "nature": "evaluation-metric diagnostic; NOT a performance claim; no test",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "self_check": {"base_WER": round(base_wer, 2),
                       "n_sub_pairs": len(sub_pairs),
                       "expected_sub_pairs": KNOWN_SUB_PAIRS,
                       "audit_range": list(AUDIT_RANGE),
                       "audit_pass": bool(audit_ok)},
        "rules": rules,
        "punct_pair_question_period_count": qp,
        "bucket_breakdown": {
            k: {"tokens": v["tokens"], "token_share_pct": round(100 * v["tokens"] / max(tot_ref, 1), 2),
                "sub": v["sub"], "del": v["del"],
                "equiv_E1": v["e1"], "equiv_E2": v["e2"], "equiv_E3": v["e3"],
                "sub_share_of_all_sub_pct": round(100 * v["sub"] / max(len(sub_pairs), 1), 2)}
            for k, v in bucket.items()},
        "top_sub_pairs": [{"ref": a, "hyp": b, "count": c} for (a, b), c in top],
        "cos_threshold": COS_TH,
    }
    (M / (TAG + "-probe.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"self_check": out["self_check"], "rules": rules,
                      "bucket_breakdown": out["bucket_breakdown"],
                      "question_period": qp}, ensure_ascii=False, indent=2), flush=True)

    # ---- 报告 ----
    rows = "\n".join(
        "| `%s` | %d | %.2f%% | **%.2f** | %.2f pp |" %
        (k, v["n_equiv_sub"], v["share_of_sub_pct"],
         v["WER_synonym_aware"], v["delta_vs_official_pp"])
        for k, v in rules.items())
    bk = out["bucket_breakdown"]
    brows = "\n".join(
        "| %s | %d (%.2f%%) | %d | %d | %d / %d / %d |" %
        (k, v["tokens"], v["token_share_pct"], v["sub"], v["del"],
         v["equiv_E1"], v["equiv_E2"], v["equiv_E3"])
        for k, v in bk.items())
    pair_rows = "\n".join("| `%s` → `%s` | %d |" % (p["ref"], p["hyp"], p["count"])
                          for p in out["top_sub_pairs"])
    txt = """# P241 / N1 同义等价评估口径诊断 — 阶段报告

> 日期：%s｜**性质：评估口径诊断，不是性能提升、不判 PASS/FAIL、不用于模型选择、不进 test**。
> 预注册 `N1-PREREG-2026-11-09.md`（08:14 写死，规则与阈值均早于运行）。
> 纯 CPU、只读已落盘后验/标注/嵌入，未跑模型前向、未改任何判据。

## 1. 依据与动机
【有】arXiv:2606.08056（BSL）正文 "Evaluation is case-insensitive and **synonym-aware**"，
Appendix I「Synonym-Based Evaluation Mapping」把同义形式在对齐时视为等价，
"To avoid penalising semantically correct but lexically variant predictions"；
该文另定义 `WER_Index` / `WER_Lex` 分层指标。
【有】Springer 2026 手语系统综述指出 WER "lacks context sensitivity"、
会 "penalize valid paraphrases"。
【有】本地 P220：693 个严格替换对中**编号变体对=0**，但高频对是真近义
（学→学习、吃饭→吃、漂亮→美），`？↔。` 合计 43 次为最高频，
错误对 gloss 文本 cos 0.1974 vs 条件随机 −0.001。
【无】把同义等价映射用于 CE-CSL 中文 gloss 无先例；三条规则属工程预注册。

前七轮都在改模型/数据/解码/输入，本轮第一次系统审视**评估目标本身**：
34%% 的 WER 里有多少是词表语义歧义造成的、而非模型能力问题。

## 2. 审计自检
不施加等价规则时重算 WER = **%.2f**（允许区间 [34.20, 34.60]），
严格 DP 回溯得 sub 对 **%d** 个（P219 已知 693）⇒ **%s**。

## 3. 三条等价规则的结果

等价只扣减严格回溯得到的 **sub 对**，不动 del / ins（防止把删除也"洗掉"）。

| 规则 | 等价 sub 数 | 占 693 | WER_synonym_aware | 相对官方降幅 |
|---|---|---|---|---|
| 官方（无归一） | 0 | — | %.2f | — |
%s

⚠️ 这些数字**不是模型提升**：CE-CSL 官方口径不承认语义归一，
34.34 / 34.41 仍是唯一可比的性能数字；本表只回答
「若评估把语义等价 gloss 视为同一标签，WER 会是多少」。

## 4. 词类分层（写死：标点 / 低频 train freq≤10 / 高频 >10）

| 桶 | token 数（占比） | sub | del | 等价 E1/E2/E3 |
|---|---|---|---|---|
%s

`？↔。` 单独出现 **%d** 次（P220 记 43 次含方向合计，口径差异见下）。
标点在手语里常无对应手势、属标注层约定，是「评估目标歧义」最典型的一类。

高频替换对：
| ref → hyp | 次数 |
|---|---|
%s

## 5. 结论与诚实边界
- 结论按预注册 §5：等价占比高 ⇒ 34%% 中有相当部分属**评估目标歧义、不可约**，
  进一步支持接受天花板收尾；占比低 ⇒ 近义 gloss 判别是真实模型缺陷（与 P220 一致）。
  **两种情况都不产生新的可执行训练方向**。
- ⚠️ 不能把 `WER_synonym_aware` 写成性能提升、不能用于挑选模型或 λ、
  不能只在 dev 上调阈值；阈值 0.75 与三条规则均在 08:14 写死，未事后调整。
- ⚠️ 本报告**不改变**官方评测口径；若未来要采用语义等价评测，须先在
  train/dev/test 一致定义等价组、预先注册，并明确与 32.74 / 33.98 等既有数字不可直接比较。
- 分层是观察性的；标点/功能词在 dev 中占比高，不等于它们"不该被识别"。
- 未跑 test、未改任何收据与判据、未启动训练。
""" % (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        base_wer, len(sub_pairs), "PASS" if audit_ok else "**INVALID**",
        base_wer, rows, brows, qp, pair_rows,
    )
    for base_dir in (REPO, DESK):
        p = base_dir / "docs/experiments/stage-P241-N1-synonym-ceiling.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> stage-P241-N1-synonym-ceiling.md (+桌面镜像)", flush=True)


def dp_pairs(ref, hyp):
    """P219 同款严格 Levenshtein DP 回溯，返回 sub 对与编辑距离。"""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + c)
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        c = 0 if ref[i - 1] == hyp[j - 1] else 1
        if D[i, j] == D[i - 1, j - 1] + c:
            if c:
                pairs.append((ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs, int(D[n, m])


if __name__ == "__main__":
    main()
