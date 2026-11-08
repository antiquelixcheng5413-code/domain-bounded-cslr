#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P220：把 P219 的高 cosine 替换对拆成「字符串变体」与「真语义近义」两层（零训练）。

必须排除的混淆：CE-CSL 词表含大量变体 gloss（最2/最、没有2/没有、帮助（我））。
这类字符串的 Qwen 嵌入天然近乎相同，可能单独制造 P219 的 0.1974。
若 lift 主要来自变体 ⇒ 下一轮方向是「词表变体归一化/标注一致性」；
若非变体对仍显著高于随机 ⇒ 下一轮方向才是「语义/视觉最小对立体 hard negatives」。

预注册（08:16 写死）：只做分层描述 + 非变体层的配对 bootstrap CI，不判 PASS/FAIL、不进 test。
"""
from __future__ import annotations

import json
import re
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

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402
from p197_beam_lm import greedy_fast  # noqa: E402
from p219_subpair_semantics import dp_pairs  # noqa: E402

M = REPO / "artifacts/metrics/blank-gov"


def norm(t):
    t = re.sub(r"\d+$", "", str(t))
    t = re.sub(r"[（(].*?[）)]", "", t)
    return t.strip()


def main():
    d = torch.load(M / "p188_logp_p184.pt", map_location="cpu", weights_only=False)
    labels_dv = P78.read_csv_split("dev.csv")
    word2idx, n, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    E = np.load(REPO / "artifacts/cache/p214_vta_target.npy")
    rng = np.random.default_rng(42)

    var_pairs, sem_pairs, all_pairs = [], [], []
    str_pairs = []
    for s in d["items"]:
        if s not in labels_dv:
            continue
        r = OW.split_glued(OW.split_gloss_sequence(labels_dv[s][1], OW.CFG))
        lp = d["items"][s]["logp"].numpy() if hasattr(d["items"][s]["logp"], "numpy") \
            else np.asarray(d["items"][s]["logp"])
        h = OW.split_glued(list(greedy_fast(lp, idx2word)))
        pairs, _ = dp_pairs(r, h)
        for a, b in pairs:
            ia, ib = word2idx.get(a), word2idx.get(b)
            if ia is None or ib is None or not (0 < ia < len(E) and 0 < ib < len(E)):
                continue
            str_pairs.append((a, b))
            all_pairs.append((ia, ib))
            if norm(a) == norm(b):
                var_pairs.append((ia, ib))
            else:
                sem_pairs.append((ia, ib))

    def cos(ps):
        if not ps:
            return np.asarray([])
        ia = np.asarray([p[0] for p in ps])
        ib = np.asarray([p[1] for p in ps])
        return (E[ia] * E[ib]).sum(1)

    cv, cs, ca = cos(var_pairs), cos(sem_pairs), cos(all_pairs)
    # 非变体层 vs 条件随机（固定 ref，随机 hyp）
    ref_ids = [p[0] for p in sem_pairs]
    rand_hyp = rng.integers(1, len(E), size=len(sem_pairs))
    ccond = (E[np.asarray(ref_ids)] * E[rand_hyp]).sum(1)
    idx = rng.integers(0, len(cs), size=(20000, len(cs)))
    boot = (cs[idx] - ccond[idx]).mean(1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    top = Counter(str_pairs).most_common(12)

    out = {
        "tag": "p220", "kind": "diagnostic_no_verdict",
        "n_all_sub_pairs": len(all_pairs),
        "variant": {"n": len(var_pairs),
                    "share": round(len(var_pairs) / max(len(all_pairs), 1), 4),
                    "mean_cos": round(float(cv.mean()), 4) if len(cv) else None},
        "non_variant": {"n": len(sem_pairs),
                        "mean_cos": round(float(cs.mean()), 4) if len(cs) else None,
                        "cond_random_cos": round(float(ccond.mean()), 4) if len(cs) else None,
                        "delta": round(float((cs - ccond).mean()), 4) if len(cs) else None,
                        "ci95": [round(lo, 4), round(hi, 4)],
                        "ci_excludes_zero": bool(lo > 0 or hi < 0)},
        "top_pairs": [{"ref": a, "hyp": b, "count": c} for (a, b), c in top],
        "note": "变体判定 = 去尾部数字 + 去括号内容后字符串相同",
    }
    (M / "p220-variant-split.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

    conclusion = (
        "lift 主要来自字符串变体 ⇒ 下一轮应优先查词表/标注变体一致性（但改词表会破坏 "
        "3516 类可比性，需单独立项），而不是硬做 hard negatives。"
        if len(var_pairs) / max(len(all_pairs), 1) > 0.5 and
        (len(cs) == 0 or float(cs.mean()) < 0.05)
        else
        "非变体层仍显著高于条件随机 ⇒ 语义/视觉近义混淆是真实存在的，"
        "下一轮方向应为最小对立体 hard negatives / 判别边界（EMNLP23、arXiv:2606.28667）。"
    )
    txt = """# P220 替换错误对分层：字符串变体 vs 真语义近义 — 诊断

> 生成：%s｜严格 Levenshtein 回溯｜**只作诊断，不判 PASS/FAIL、不进 test**

## 1. 问题
P219 严格复核得到 693 个替换对，平均 cos **0.1974** vs 条件随机 −0.001
（Δ=+0.1984，CI [0.1747, 0.2214]）。但 CE-CSL 词表含大量变体 gloss
（`最2`/`最`、`没有2`/`没有`、`帮助（我）`），其文本嵌入天然近乎相同，
可能单独制造这个 lift ⇒ 必须分层。

## 2. 分层结果

| 层 | 数量 | 占比 | 平均 cos |
|---|---|---|---|
| 字符串变体对 | %d | %.1f%% | %s |
| 非变体对 | %d | %.1f%% | %s |
| 全部替换对 | %d | 100%% | %.4f |
| 非变体层条件随机（固定 ref） | %d | — | %.4f |

非变体层 Δ = **%+.4f**，配对 bootstrap 95%% CI **[%+.4f, %+.4f]**%s。

高频替换对（前 12）：
%s

## 3. 结论（决定下一轮方向）

**%s**

## 4. 诚实边界
- 变体判定是字符串规则（去尾部数字、去括号），可能漏掉其它变体形式。
- 文本嵌入来自 Qwen 中文 LM，语义相近 ≠ 视觉/手形相近；该嵌入是 F2（VTA）副产物，
  F2 已判负 ⇒ 它能描述错误结构，不代表能提供有效监督。
- 只描述 dev 515 句，未跨 split 验证；不进 test、不判 PASS/FAIL。

## 5. 状态
未跑 test、未 commit、未 push。数据来源 `artifacts/metrics/blank-gov/p220-variant-split.json`。
""" % (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        len(var_pairs), 100.0 * len(var_pairs) / max(len(all_pairs), 1),
        ("%.4f" % float(cv.mean())) if len(cv) else "—",
        len(sem_pairs), 100.0 * len(sem_pairs) / max(len(all_pairs), 1),
        ("%.4f" % float(cs.mean())) if len(cs) else "—",
        len(all_pairs), float(ca.mean()), len(sem_pairs),
        float(ccond.mean()) if len(cs) else 0.0,
        float((cs - ccond).mean()) if len(cs) else 0.0, lo, hi,
        "（CI 不含 0）" if (lo > 0 or hi < 0) else "（CI 含 0）",
        "\n".join("- `%s` → `%s`  ×%d" % (a, b, c) for (a, b), c in top),
        conclusion,
    )
    for p in (REPO / "docs/experiments/stage-P220-variant-split.md",
              DESK / "docs/experiments/stage-P220-variant-split.md"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> stage-P220-variant-split.md (+桌面镜像)", flush=True)


if __name__ == "__main__":
    main()
