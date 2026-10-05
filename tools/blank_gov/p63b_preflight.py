"""P63b：训练前预检 —— 避免 2.2 小时训练白跑

检查四件事，任一不过就不该启动训练：
  1. 新特征覆盖：train/dev 是否齐（缺样本会被静默丢弃）
  2. 新特征质量：shape/NaN/presence 检出率（与旧特征对比）
  3. 3516 词表在**真实数据**上可用：参考序列长度 vs CTC 的 T>=2L-1 约束
  4. dev 样本量是否够做评估口径

第3 条最关键：3516 词表意味着更多 OOV，句子可能变长；
若某句超出 T=48 能容纳的最大长度，ctc_loss 会抛异常（库里 zero_infinity=False）。
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np


def _find_repo() -> Path:
    for c in Path(__file__).resolve().parents:
        if (c / "src" / "cslr" / "recognition"
                / "gloss_sequence.py").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo()
sys.path.insert(0, str(REPO / "src"))

from cslr.recognition.gloss_sequence import (  # noqa: E402
    build_ordered_vocabulary, split_gloss_sequence, GlossSequenceConfig,
    token_is_numeric)


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(fh)}


NEW = REPO / "artifacts/part3_features_tasksapi"
OLD = REPO / "artifacts/part3_features"
T = 48                      # 特征帧数
SENT_MAX = 24              # p40 里句子长度门槛
out = {}

print("=" * 72)
print("1. 新特征覆盖")
print("=" * 72)
lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
cov = {}
for split, lab, name in (("train", lab_tr, "train"),
                         ("validation", lab_dv, "dev")):
    fs = set(p.rsplit("/", 1)[-1].replace(".landmark.npy", "")
             for p in glob.glob(str(NEW / split / "*.landmark.npy")))
    ids = set(lab)
    missing = ids - fs
    # 同时看满足句子长度门槛的还剩多少
    ok = 0
    for sid in lab:
        if sid in missing:
            continue
        toks = split_gloss_sequence(lab[sid], GlossSequenceConfig())
        if toks and len(toks) <= SENT_MAX:
            ok += 1
    cov[name] = {"labeled": len(ids), "extracted": len(fs),
                 "missing": len(missing), "usable": ok,
                 "coverage": round(len(fs) / len(ids), 4)}
    print("  %-5s 标注 %d  已提 %d  缺 %d  满足(1<=L<=%d) %d 覆盖 %.1f%%"
          % (name, len(ids), len(fs), len(missing), SENT_MAX, ok,
             100 * len(fs) / len(ids)))
out["coverage"] = cov

print("\n" + "=" * 72)
print("2. 新特征质量（抽样 vs 旧特征基准）")
print("=" * 72)
blocks = [("hands", 0, 126), ("pose", 126, 158), ("face", 158, 182),
          ("presence", 182, 186), ("deltas", 186, 368)]


def sample_stats(root, split, step=17):
    fs = sorted(glob.glob(str(root / split / "*.landmark.npy")))[::step]
    if not fs:
        return None
    P, stds, bad = [], [], 0
    for f in fs:
        a = np.load(f)
        if a.shape != (T, 368) or not np.isfinite(a).all():
            bad += 1
            continue
        P.append(a[:, 182:186].mean(0))
        stds.append([a[:, s0:s1].std() for _, s0, s1 in blocks])
    return {"n": len(fs), "bad": bad,
            "presence": np.mean(P, 0).round(4).tolist(),
            "stds": np.mean(stds, 0).round(4).tolist()}


new_s = sample_stats(NEW, "train")
old_s = sample_stats(OLD, "train")
if new_s:
    print("  新特征（抽样 %d，异常 %d）" % (new_s["n"], new_s["bad"]))
    print("    presence(pose/handL/handR/face) = %s" % new_s["presence"])
    print("    双手和 = %.3f" % (new_s["presence"][1] + new_s["presence"][2]))
    print("    各块 std = %s" % dict(
        (b, v) for (b, _, _), v in zip(blocks, new_s["stds"])))
if old_s:
    print("  旧特征基准（抽样 %d）" % old_s["n"])
    print("    presence = %s" % old_s["presence"])
    print("    双手和 = %.3f" % (old_s["presence"][1] + old_s["presence"][2]))
    print("    各块 std = %s" % dict(
        (b, v) for (b, _, _), v in zip(blocks, old_s["stds"])))
    if new_s:
        print("\n  双手检出率差（新-旧）= %+.3f"
              % ((new_s["presence"][1] + new_s["presence"][2])
                 - (old_s["presence"][1] + old_s["presence"][2])))
out["quality"] = {"new": new_s, "old": old_s}

print("\n" + "=" * 72)
print("3. 🔴 3516 词表下的 CTC 约束 T >= 2L-1")
print("=" * 72)
cfg = GlossSequenceConfig()
for cap, mf in ((300, 2), (None, 1)):
    voc, _ = build_ordered_vocabulary(list(lab_tr.values()),
                                      min_frequency=mf, max_tokens=cap)
    Tk = set(voc.tokens)
    lens = []
    n_unk = 0
    n_tok = 0
    for sid, g in lab_dv.items():
        toks = split_gloss_sequence(g, cfg)
        lens.append(len(toks))
        n_tok += len(toks)
        n_unk += sum(1 for t in toks if t not in Tk)
    lens = np.array(lens)
    need = 2 * lens - 1                       # CTC 下界
    over = int((need > T).sum())
    print("\n  词表 %d（min_freq=%s, max=%s）" % (voc.size - 1, mf, cap))
    print("    dev 句长 mean %.2f  max %d" % (lens.mean(), lens.max()))
    print("    需要 T >= 2L-1：mean %.1f  max %d（特征 T=%d）"
          % (need.mean(), need.max(), T))
    print("    ❌ 超出 T 的句子: %d / %d" % (over, len(lens)))
    print("    dev unk token: %d / %d = %.1f%%" % (n_unk, n_tok,
                                            100 * n_unk / n_tok))
    out.setdefault("ctc_feasibility", []).append({
        "vocab": voc.size - 1, "min_freq": mf, "max_tokens": cap,
        "dev_len_mean": round(float(lens.mean()), 2),
        "dev_len_max": int(lens.max()),
        "need_T_mean": round(float(need.mean()), 1),
        "need_T_max": int(need.max()),
        "feature_T": T,
        "sentences_over_T": over,
        "dev_unk_ratio": round(n_unk / n_tok, 4),
    })

print("""
  ⚠️ 注意：p40 里有 `len(toks) > 24 就跳过` 的门槛，
     所以实际进训练的句子 max L <= 24 -> 需要 T >= 47，恰好卡在 T=48。
     **若有任何句子被判unk 后变长，或门槛被放宽，训练会直接抛异常**
     （ctc_loss 的 zero_infinity=False，会raise 而不是静默取 0）。
""")

print("=" * 72)
print("4. 预检结论")
print("=" * 72)
gates = []
g1 = cov["train"]["missing"] == 0 and cov["dev"]["missing"] == 0
gates.append(("特征覆盖完整（train/dev 无缺失）", g1,
              "train缺%d dev缺%d" % (cov["train"]["missing"],
                                    cov["dev"]["missing"])))
g2 = bool(new_s) and new_s["bad"] == 0
gates.append(("新特征无 shape/NaN 异常", g2,
              "异常 %d 条" % (new_s["bad"] if new_s else -1)))
g3 = cov["train"]["usable"] > 100and cov["dev"]["usable"] > 100
gates.append(("可用样本量足够（train>100 且 dev>100）", g3,
              "train %d dev %d" % (cov["train"]["usable"],
                                    cov["dev"]["usable"])))
ok_all = True
for nm, ok, detail in gates:
    print("  [%s] %s —— %s" % ("✓" if ok else "✗", nm, detail))
    ok_all = ok_all and ok
print("\n  => %s" % ("**全部通过，可以启动训练**" if ok_all
                    else "**未通过，先解决上面的问题**"))
out["gates"] = [{"name": n, "pass": o, "detail": d} for n, o, d in gates]
out["all_pass"] = ok_all

p = REPO / "artifacts/metrics/blank-gov/p63b-preflight.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % p)