#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把候选收据的 per_item 按基线收据的 sid 顺序重排，使配对 bootstrap 逐句对齐。

背景：p184-base-uwer.json 的历史句序（p207 collect）与 night_eval_full.py 的自然句序
在前两句交换；night_judge.paired_bootstrap 有「sid 非空必须同序」断言（H1 那次加的）。
本工具只重排 per_item，**不改任何聚合指标、不改判据、不改数据**，并做前后求和自检。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
M = REPO / "artifacts/metrics/blank-gov"
SUM_KEYS = ("edits", "ref_n", "ins", "sub", "del", "low_n", "low_err")


def load(tag):
    return json.loads((M / (tag + "-uwer.json")).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base", default="p184-base")
    a = ap.parse_args()

    b, c = load(a.base), load(a.cand)
    bs = [str(x.get("sid", "")) for x in b["per_item"]]
    cs = [str(x.get("sid", "")) for x in c["per_item"]]
    before = hashlib.md5("|".join(cs).encode()).hexdigest()[:12]

    if bs == cs:
        print("already aligned (%s vs %s)" % (a.cand, a.base), flush=True)
        return
    assert len(bs) == len(cs), "句数不同: %d vs %d" % (len(bs), len(cs))
    assert all(bs) and all(cs), "存在空 sid"
    assert set(bs) == set(cs), "sid 集合不同，不能只靠重排修复"
    assert len(set(bs)) == len(bs), "基线存在重复 sid"

    idx = {s: i for i, s in enumerate(cs)}
    new = [c["per_item"][idx[s]] for s in bs]
    for k in SUM_KEYS:
        s0 = sum(x.get(k, 0) for x in c["per_item"])
        s1 = sum(x.get(k, 0) for x in new)
        assert s0 == s1, "重排改变了 %s 求和: %s vs %s" % (k, s0, s1)
    # 聚合字段原样保留（不重新计算，防止口径漂移）
    c["per_item"] = new
    c["per_item_alignment"] = {
        "base_tag": a.base,
        "reason": "candidate per_item sid order differed from base; "
                  "reordered by sid for valid paired bootstrap",
        "before_order_md5": before,
        "after_order_md5": hashlib.md5(
            "|".join(str(x["sid"]) for x in new).encode()).hexdigest()[:12],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (M / (a.cand + "-uwer.json")).write_text(
        json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {"cand": a.cand, "base": a.base, "n": len(new),
             "sums_verified": {k: sum(x.get(k, 0) for x in new) for k in SUM_KEYS},
             "aggregates_unchanged": True,
             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    (M / (a.cand + "-align-audit.json")).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ALIGNED %s -> %s  n=%d  sums_verified=%s"
          % (a.cand, a.base, len(new), audit["sums_verified"]), flush=True)


if __name__ == "__main__":
    main()
