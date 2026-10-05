"""P57b：残留 unk 词是什么？以及去 unk 的真实代价"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path


def _find_repo() -> Path:
    for c in Path(__file__).resolve().parents:
        if (c / "src" / "cslr" / "recognition"
                / "gloss_sequence.py").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo()
sys.path.insert(0, str(REPO / "src"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402


def read_csv(p: Path) -> dict[str, str]:
    with open(p, newline="", encoding="utf-8") as fh:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(fh)}


def toks(g: str) -> list[str]:
    return [t.strip() for t in g.split("/") if t.strip()]


lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")

tr = collections.Counter()
for g in lab_tr.values():
    tr.update(toks(g))
dv = collections.Counter()
for g in lab_dv.values():
    dv.update(toks(g))

print("=" * 72)
print("残留 unk 词的形态分析")
print("=" * 72)

# 1) 数字后缀是不是标注变体？
import re
NUM_SUFFIX = re.compile(r"^[^\d]*\d+$")
all_tok = set(tr) | set(dv)
with_num = {t for t in all_tok if NUM_SUFFIX.match(t)}
print("\n1) 形如 `禁止1` / `看1` 的数字后缀词：%d 个 / 全部词种 %d"
      % (len(with_num), len(all_tok)))
base_of = collections.defaultdict(set)
for t in with_num:
    base_of[NUM_SUFFIX.sub("", t)].add(t)
multi = {b: v for b, v in base_of.items() if len(v) > 1}
print("   去掉数字后有多个变体的词根：%d 个" % len(multi))
for b, v in list(multi.items())[:12]:
    print("      %-8s -> %s" % (b, sorted(v)))
print("   样例单变体：", sorted(list(with_num))[:8])

# 2) 合并数字后缀后残留 unk 会怎样
def strip_num(t: str) -> str:
    return NUM_SUFFIX.sub("", t) or t

merged_tr = collections.Counter()
for g in lab_tr.values():
    merged_tr.update({strip_num(t) for t in toks(g)})
voc_m, _ = build_ordered_vocabulary(
    [" ".join([strip_num(t) for t in toks(g)]) for g in lab_tr.values()],
    min_frequency=2, max_tokens=None)
T_m = set(voc_m.tokens)
resid_m = {t: c for t, c in dv.items() if strip_num(t) not in T_m}
n_dv = sum(dv.values())
print("\n2) 合并数字后缀变体后（词表 %d）：" % (voc_m.size - 1))
print("   dev 残留 unk token = %d / %d = %.1f%%"
      % (sum(resid_m.values()), n_dv, 100 * sum(resid_m.values()) / n_dv))
print("   残留词种 %d：%s" % (len(resid_m), list(resid_m)[:12]))
tr_m = {strip_num(t) for t in tr}
truly = [t for t in resid_m if strip_num(t) not in tr_m]
print("   其中 train 真没出现过的：%d 个 %s" % (len(truly), truly[:12]))

# 3) train 侧：合并后词表规模与频次分布
print("\n3) 合并后 train 侧词频分布（决定学不学得动）")
mc = merged_tr
for lo, hi, nm in ((1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"),
                   (21, 100, "21-100"), (101, 10**9, "100+")):
    n = sum(1 for v in mc.values() if lo <= v <= hi)
    cov = sum(v for v in mc.values() if lo <= v <= hi)
    print("   频次 %-8s 词种 %5d  覆盖 train token %d"
          % (nm, n, cov))

# 4) dev 里的 token 按「合并后的 train 频次」分桶 —— 决定 WER 能到多好
print("\n4) dev token 按「合并后 train 频次」分桶（这才是 WER 的地板结构）")
buckets = collections.Counter()
covd = collections.Counter()
for t, c in dv.items():
    f = mc.get(strip_num(t), 0)
    if f == 0:
        k = "0（unseen）"
    elif f <= 2:
        k = "1-2"
    elif f <= 10:
        k = "3-10"
    elif f <= 50:
        k = "11-50"
    elif f <= 200:
        k = "51-200"
    else:
        k = "200+"
    buckets[k] += c
    covd[k] += c
print("   %-12s %8s %8s" % ("频次桶", "token 数", "占比"))
for k in ("0（unseen）", "1-2", "3-10", "11-50", "51-200", "200+"):
    c = covd[k]
    print("   %-12s %8d %7.1f%%" % (k, c, 100 * c / n_dv))
learnable = sum(c for k, c in covd.items() if k not in ("0（unseen）",))
print("\n   train 频次 >=1 的 dev token = %d / %d = %.1f%%（这部分有可能学会）"
      % (learnable, n_dv, 100 * learnable / n_dv))
print("   => WER 地板下界 ≈ %.1f%%（即使 0 错误也躲不过 unseen 部分）"
      % (100 * (n_dv - learnable) / n_dv))

out = {
    "numeric_suffix_variants": {"n_tokens": len(with_num),
                                "n_bases_with_multiple": len(multi),
                                "examples": {k: sorted(v)
                                             for k, v in list(multi.items())[:20]}},
    "after_merging": {
        "vocab_size": voc_m.size - 1,
        "dev_residual_unk_tokens": sum(resid_m.values()),
        "dev_residual_unk_ratio": round(sum(resid_m.values()) / n_dv, 4),
        "residual_types": len(resid_m),
        "truly_unseen_in_train": len(truly),
        "truly_unseen_list": truly[:40],
    },
    "dev_token_frequency_buckets": {k: covd[k] for k in
                                    ("0（unseen）", "1-2", "3-10",
                                     "11-50", "51-200", "200+")},
    "dev_total_tokens": n_dv,
    "wer_floor_lower_bound": round(
        100 * (n_dv - learnable) / n_dv, 2),
}
p = REPO / "artifacts/metrics/blank-gov/p57b.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % p)