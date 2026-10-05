"""P58：300 词表是什么时候、为什么引入的？

用 git 历史 + 全仓库扫描追溯 max_tokens=300 的来源，
并量化「只用 300 词表在能力上能做什么、不能做什么」。
"""
from __future__ import annotations

import collections
import csv
import json
import subprocess
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


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args],
                              capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as exc:                                  # noqa: BLE001
        return "(git failed: %s)" % exc


def read_csv(p: Path) -> dict[str, str]:
    with open(p, newline="", encoding="utf-8") as fh:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(fh)}


def toks(g: str) -> list[str]:
    return [t.strip() for t in g.split("/") if t.strip()]


out: dict = {"generated": str(REPO / "artifacts/metrics/blank-gov/p58.json")}

# ---------------------------------------------------------------- 1. git 溯源
print("=" * 72)
print("1. 300 这个数字从哪来 —— git 追溯")
print("=" * 72)

print("\n[A] gloss_sequence.py 里 max_tokens 的默认值")
print(git("log", "--oneline", "--all", "--",
          "src/cslr/recognition/gloss_sequence.py"))

print("\n[B] 最早引入 max_tokens 参数的提交")
print(git("log", "--all", "-S", "max_tokens", "--oneline", "--",
          "src/cslr/recognition/gloss_sequence.py"))

print("\n[C] 全仓库里出现 300 的地方（代码，排除实验脚本）")
for pat in ("max_tokens=300", "max_tokens = 300", "MAX_TOKENS = 300",
            "max_tokens=300)"):
    r = git("grep", "-n", pat, "--", "src", "app", "configs")
    if r:
        print("  模式 %s：" % pat)
        for line in r.splitlines()[:12]:
            print("    %s" % line)

print("\n[D] 首次出现 max_tokens=300 的提交时间")
r = git("log", "--all", "-S", "max_tokens=300", "--format=%h %ad %s",
        "--date=short", "--", "src", "app", "tools")
print(r if r else "  (src/app/tools 下无 max_tokens=300 字面量)")

print("\n[E] 所有 max_tokens 相关的历史提交（时间序）")
r = git("log", "--all", "--format=%h %ad %s", "--date=short",
        "-S", "max_tokens")
for line in r.splitlines():
    print("    %s" % line)

print("\n[F] GlossSequenceConfig 是否有 vocab 上限配置")
print(git("grep", "-n", "max_tokens", "--", "src/cslr"))

print("\n[G] 配置文件里的 max_tokens（真正的参数来源？）")
for d in ("configs", "config", ".", "docs"):
    r = git("grep", "-n", "max_tokens", "--", d)
    if r:
        for line in r.splitlines()[:15]:
            if "/tools/blank_gov" in line:
                continue
            print("    %s" % line)

print("\n[H] p40_rgb_main.py 里怎么设的（我们 48 轮实验都走它）")
r = git("grep", "-n", "max_tokens\\|min_frequency\\|build_ordered_vocabulary",
        "--", "tools/blank_gov/p40_rgb_main.py")
for line in r.splitlines()[:15]:
    print("    %s" % line)

print("\n[I] 该行是哪个提交引入的")
for kw in ("max_tokens=300", "max_tokens = 300"):
    r = git("log", "--all", "-S", kw, "--format=%h %ad %s", "--date=short")
    if r:
        print("    模式 %s:" % kw)
        for line in r.splitlines()[:8]:
            print("      %s" % line)

print("\n[J] a6a7100（最早引入 max_tokens 的提交）改了哪些文件")
print(git("show", "--stat", "--format=%h %ad %s%n%n%b", "--date=short",
          "a6a7100")[:2000])

# ---------------------------------------------------------------- 2. 能力边界
print("\n" + "=" * 72)
print("2. 只用 300 词表，能力上能覆盖多少")
print("=" * 72)

lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")

voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                 max_tokens=300)
T = set(voc.tokens)

tr_sent = [[t for t in toks(g)] for g in lab_tr.values()]
dv_sent = [[t for t in toks(g)] for g in lab_dv.values()]

# 1) token 级覆盖
n_all = sum(len(s) for s in dv_sent)
n_in = sum(1 for s in dv_sent for t in s if t in T)
# 2) 整句可表达：每个 token 都在词表里
full_ok = sum(1 for s in dv_sent if all(t in T for t in s))
# 3) 句内覆盖率分布
covs = []
for s in dv_sent:
    if s:
        covs.append(sum(1 for t in s if t in T) / len(s))

print("\n  词表 300")
print("  dev token 级覆盖      %d/%d = %.1f%%" % (n_in, n_all, 100 * n_in / n_all))
print("  dev 整句可表达率      %d/%d = %.1f%%" % (full_ok, len(dv_sent),
                                           100 * full_ok / len(dv_sent)))
import statistics
print("  单句覆盖率中位数      %.1f%%" % (100 * statistics.median(covs)))
n_le50 = sum(1 for c in covs if c <= 0.5)
n_zero = sum(1 for c in covs if c == 0)
print("  单句覆盖率 <=50pct 的句子 %d/%d = %.1f%%"
      % (n_le50, len(covs), 100 * n_le50 / len(covs)))
print("  单句覆盖率 ==0 的句子   %d/%d = %.1f%%"
      % (n_zero, len(covs), 100 * n_zero / len(covs)))

# 训练侧：300 词表覆盖了多少 train token
n_tr = sum(len(s) for s in tr_sent)
n_tr_in = sum(1 for s in tr_sent for t in s if t in T)
print("  train token 级覆盖   %d/%d = %.1f%%" % (n_tr_in, n_tr,
                                          100 * n_tr_in / n_tr))
# 每个 unk token 折叠成同一个类 → 训练目标里 unk 的占比
print("\n  【折叠的代价】在 300 词表下，训练 target 里 <unk> 占 %.1f pct"
      % (100 - 100 * n_tr_in / n_tr))
print("  300 个真实类 vs 1 个 unk 类 —— 负正样本比 = %.0f 比 1"
      % ((n_tr - n_tr_in) / max(n_tr_in, 1)))

# 哪些高频词被砍掉了
tr_cnt = collections.Counter()
for s in tr_sent:
    tr_cnt.update(s)
cut = [(t, c) for t, c in tr_cnt.most_common() if t not in T]
print("\n  【被砍掉的高频词】前 20 个（train 频次）")
for t, c in cut[:20]:
    print("    %-10s train 出现 %5d 次" % (t, c))
print("  被砍掉的 train token 总数 = %d（占 %.1f%%）"
      % (sum(c for _, c in cut), 100 * sum(c for _, c in cut) / n_tr))

# 扩表后对比
print("\n  【对比】不同词表规模的能力边界")
table = []
for cap, mf in ((300, 2), (1828, 2), (None, 2), (None, 1)):
    v, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=mf,
                                    max_tokens=cap)
    TT = set(v.tokens)
    ok = sum(1 for s in dv_sent if all(t in TT for t in s))
    inn = sum(1 for s in dv_sent for t in s if t in TT)
    table.append({
        "min_frequency": mf, "max_tokens": cap, "vocab": v.size - 1,
        "ctc_classes": v.size,
        "dev_token_coverage": round(inn / n_all, 4),
        "dev_sentence_full_coverage": round(ok / len(dv_sent), 4),
    })
    print("    词表 %5d (min_freq=%s,max=%s): token 覆盖 %.1f%%  整句可表达 %.1f%%"
          % (v.size - 1, mf, cap, 100 * inn / n_all, 100 * ok / len(dv_sent)))

out["git_trace"] = {
    "commits_touching_gloss_sequence": git(
        "log", "--oneline", "--all", "--",
        "src/cslr/recognition/gloss_sequence.py").splitlines(),
    "first_max_tokens_commit": git(
        "log", "--all", "-S", "max_tokens", "--format=%h %ad %s",
        "--date=short", "--", "src/cslr/recognition/gloss_sequence.py"),
    "max_tokens_300_literal_in_repo": git(
        "grep", "-n", "max_tokens=300", "--", "src", "app", "tools"),
    "all_max_tokens_commits": git(
        "log", "--all", "--format=%h %ad %s", "--date=short",
        "-S", "max_tokens").splitlines(),
}
out["capability_with_300"] = {
    "vocab_size": voc.size - 1,
    "ctc_classes": voc.size,
    "dev_token_coverage": round(n_in / n_all, 4),
    "dev_sentence_full_coverage": round(full_ok / len(dv_sent), 4),
    "median_sentence_coverage": round(float(statistics.median(covs)), 4),
    "sentences_le_50pct_coverage": round(
        sum(1 for c in covs if c <= 0.5) / len(covs), 4),
    "sentences_zero_coverage": round(
        sum(1 for c in covs if c == 0) / len(covs), 4),
    "train_token_coverage": round(n_tr_in / n_tr, 4),
    "train_target_unk_ratio": round(1 - n_tr_in / n_tr, 4),
    "neg_to_pos_class_ratio": round(
        (n_tr - n_tr_in) / max(n_tr_in, 1) * 300, 1),
    "top_cut_high_freq_tokens": [{"tok": t, "train_freq": c}
                                 for t, c in cut[:25]],
    "cut_train_tokens": sum(c for _, c in cut),
}
out["vocab_scaling_table"] = table

p = REPO / "artifacts/metrics/blank-gov/p58.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % p)