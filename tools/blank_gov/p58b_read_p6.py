"""P58b：读 P6 词表消融的原始数据，判断「300 影响有限」这个结论是否可信"""
from __future__ import annotations

import json
from pathlib import Path

D = Path("/home/su127/FYP/domain-bounded-cslr/artifacts/metrics/blank-gov")

for name in ("p6-vocab-ablation-long.json", "p6-vocab-ablation.json"):
    p = D / name
    if not p.exists():
        print("缺失: %s" % name)
        continue
    d = json.load(open(p, encoding="utf-8"))
    print("=" * 72)
    print(name)
    print("=" * 72)
    print("keys: %s" % list(d.keys()))
    for key in ("train", "config", "training", "meta"):
        if key in d:
            print("\n[%s] %s" % (key, json.dumps(d[key], ensure_ascii=False)[:700]))

    print("\nresults:")
    for r in d.get("results", []):
        b = r.get("best", {})
        print("  max_tokens=%-6s min_freq=%-4s vocab=%-5s "
              "dev_tok_cov=%.3f dev_full_cov=%.3f  best_wer=%s@ep%s"
              % (r.get("max_tokens"), r.get("min_frequency"),
                 r.get("vocab_size"),
                 r.get("dev_token_coverage", -1),
                 r.get("dev_sample_full_coverage", -1),
                 b.get("wer", "-"), b.get("epoch", "-")))

    for key in ("verdict", "conclusion", "conclusions", "notes", "summary"):
        if key in d:
            print("\n[%s]" % key)
            print(json.dumps(d[key], ensure_ascii=False, indent=1)[:2500])

    # history 里看 epoch 数与曲线形态
    for r in d.get("results", []):
        hist = r.get("history") or []
        if hist:
            print("\n  max_tokens=%s 的 epoch 数 = %d，前 5 个 dev WER: %s"
                  % (r.get("max_tokens"), len(hist),
                     [round(h.get("wer", -1), 4) for h in hist[:5]]))
            print("     后 5 个: %s"
                  % [round(h.get("wer", -1), 4) for h in hist[-5:]])
    print()