#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P182: length-bias check on cached P178 dev hypotheses (CPU only)."""
import csv, json
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
store = json.loads((REPO / "tools/blank_gov/p178_tokens_vac.json").read_text(encoding="utf-8"))

ref_map = {}
with open(REPO / "data/raw/CE-CSL/label/dev.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ref_map[row["Number"]] = row["Gloss"]

rows = []
for sid, hyp in store.items():
    if sid not in ref_map:
        continue
    ref = [t for t in ref_map[sid].split("/") if t]
    rows.append((len(ref), len(hyp), sid))

n = len(rows)
ref_len = sum(r[0] for r in rows) / n
hyp_len = sum(r[1] for r in rows) / n
print(f"n = {n}")
print(f"ref tokens/utt mean = {ref_len:.2f}")
print(f"hyp tokens/utt mean = {hyp_len:.2f}   ratio = {hyp_len/ref_len:.3f}")
shorter = sum(1 for r in rows if r[1] < r[0])
longer = sum(1 for r in rows if r[1] > r[0])
equal = sum(1 for r in rows if r[1] == r[0])
print(f"hyp shorter than ref: {shorter} ({100*shorter/n:.1f}%)")
print(f"hyp equal   to  ref: {equal} ({100*equal/n:.1f}%)")
print(f"hyp longer  than ref: {longer} ({100*longer/n:.1f}%)")

for k in (1, 2, 3):
    c = sum(1 for r in rows if r[1] <= k)
    print(f"hyp length <= {k}: {c} ({100*c/n:.1f}%)")

# length-conditioned: mean ref length of the shortest hypotheses
short = [r for r in rows if r[1] <= 2]
print(f"\nof the {len(short)} utts with hyp<=2 tokens, mean ref length = "
      f"{sum(r[0] for r in short)/max(len(short),1):.2f}")
