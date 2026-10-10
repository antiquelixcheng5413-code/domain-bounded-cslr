"""P97 输出模式诊断（零训练成本，只推理）。

要回答的问题（按重要性）：
① 输出坍缩是否突破？—— P84 定的判据：输出长度 > 3 token/句、distinct > 50
② 与历史对比：P84（600条,0.95 token,distinct 12）→ P87（2000条,1.78,20）→ P94（4973条,ep29,?）
③ 按训练频次的漏检率 —— P72 当时低频桶100% 全灭，现在如何？
④ 最优 10 个输出 vs 参考最频 10 个，模型在说���么
⑤ 如果坍缩仍未突破，C 方向（词级头）是否值得重测

用P94 的 best checkpoint（ep29, WER 62.49%）。
"""
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools/blank_gov"))

import importlib.util
import numpy as np
import torch
import torch.nn as nn

spec = importlib.util.spec_from_file_location(
    "p78", REPO / "tools/blank_gov/p78_train_official_tfnet.py")
p78 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p78)

import official_wer as OW

TAG = "p94-30ep"
CKPT = REPO / "artifacts/checkpoints" / ("%s-best.pt" % TAG)

print("=" * 88)
print("P97 输出模式诊断 —— checkpoint: %s" % CKPT.name)
print("=" * 88)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("  记录 epoch = %d   WER_official = %.2f%%"
      % (ck["epoch"], ck["wer_official"]))
print("  img_size = %s" % ck["config"]["img_size"])
print()

dev = "cuda"
model = p78.Net.moduleNet(ck["hidden"], ck["wordSetNum"] + 1, "VAC",
                          torch.device(dev), "CE-CSL", True).to(dev)
model.load_state_dict(ck["model_state"])
model.eval()

idx2word = ck["idx2word"]
wordSetNum = ck["wordSetNum"]

# 词表（官方口径，用于按训练频次分桶）
C = "external/TFNet/data/CE-CSL/%s.csv"
w2i, wsn, idx2w_official = p78.DPM.Word2Id(
    C % "train", C % "dev", C % "test", "CE-CSL")

raw_dv = p78.read_csv_split("dev.csv")
lab_dv = {k: v[1] for k, v in raw_dv.items()}
labels_dv = {k: v for k, v in raw_dv.items()}

ds = p78.RGBSeqDataset("dev", labels_dv, w2i,
                       p78.build_transforms(False, 160)[1], False)
from torch.utils.data import DataLoader
# 🔴 num_workers 必须 0：p78 是动态加载的模块（spec_from_file_location），
#    Python 3.14 的 forkserver 无法 pickle 它（No module named 'p78'）
tr_dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0,
                   collate_fn=p78.collate)
ls = nn.LogSoftmax(dim=-1)

hyps, refs = [], []
with torch.no_grad():
    for vid, tgt, tgt_len, dl, tl_len, sids, _ids in tr_dl:
        vid = vid.to(dev)
        out = model(vid, dl, False)
        logp = ls(out[0])          # 官方布局 [T,B,C]
        hyps += p78.greedy_decode(logp, tl_len, idx2word)
        for s in sids:
            refs.append(labels_dv[s][1])

print("推理完成：%d 句" % len(hyps))
print()

# ----① 输出长度 / distinct ----
out_lens = [len(h) for h in hyps]
all_toks = Counter()
for h in hyps:
    all_toks.update(h)
ref_lens = [len(OW.split_glued(OW.split_gloss_sequence(r))) for r in refs]
print("=" * 88)
print("① 输出坍缩判据（P84 定：输出长度 > 3、distinct > 50）")
print("=" * 88)
print("  输出长度 mean = %.2f   参考长度 mean = %.2f   欠输出比 %.2f"
      % (np.mean(out_lens), np.mean(ref_lens),
         1 - np.mean(out_lens) / np.mean(ref_lens)))
print("  distinct = %d" % len(all_toks))
p_len = np.mean(out_lens) > 3.0
p_dis = len(all_toks) > 50
print("  ⇒ 输出长度 > 3 : %s（实测 %.2f）" % ("✅ 通过" if p_len else "❌ 未过", np.mean(out_lens)))
print("  ⇒ distinct  > 50: %s（实测 %d）" % ("✅ 通过" if p_dis else "❌ 未过", len(all_toks)))
print("  ⇒ **突破坍缩阶段：%s**" % ("是" if (p_len and p_dis) else "否"))
print()

print("=" * 88)
print("② 与历史对比（同一判据）")
print("=" * 88)
print("  %-28s %10s %8s %12s" % ("实验", "输出长度", "distinct", "WER%"))
rows = [("P84 VAC 600条",0.95, 12, 89.06),
        ("P87 VAC 2000条",   1.78, 20, 80.96),
        ("P91 VAC 4973条 6ep", 1.78, 20, 72.94)]
print("  %-28s %10.2f %8d %12.2f" % rows[0])
print("  %-28s %10.2f %8d %12.2f" % rows[1])
print("  %-28s %10s %8s %12.2f" % ("P91 VAC 4973条 6ep",
                             "1.78*", "20*", 72.94))
print("  %-28s %10.2f %8d %12.2f"
      % ("P94 VAC 4973条 30ep", np.mean(out_lens), len(all_toks), 62.49))
print("  * P91 未跑验收，沿用 P87 的诊断值（两者同 160res）")
print()

print("=" * 88)
print("③ 按训练频次的漏检率（P72 当时低频桶 100% 全灭）")
print("=" * 88)
train_cnt = Counter()
raw_tr = p78.read_csv_split("train.csv")
for k, v in raw_tr.items():
    train_cnt.update(OW.split_gloss_sequence(v[1]))


def bucket(f):
    if f >= 100:
        return ">=100"
    if f >= 11:
        return "11-100"
    return "1-10"


hit = Counter()
tot = Counter()
for h, r in zip(hyps, refs):
    rt = OW.split_glued(OW.split_gloss_sequence(r))
    for t in rt:
        b = bucket(train_cnt.get(t, 1))
        tot[b] += 1
        if t in h:
            hit[b] += 1
print("  %-10s %8s %8s %10s" % ("训练频次", "dev token", "命中", "漏检率"))
for b in (">=100", "11-100", "1-10"):
    if tot[b]:
        print("  %-10s %8d %8d %9.1f%%" % (b, tot[b], hit[b],
                              100.0 * (1 - hit[b] / tot[b])))
print()

print("=" * 88)
print("④ 模型在说什么（最高频 12 个输出）")
print("=" * 88)
top = all_toks.most_common(12)
for i, (w, c) in enumerate(top, 1):
    print("  %2d. %-8s %4d  (%.1f%% of全部输出)" % (i, w, c,
                                           100.0 * c / max(sum(out_lens), 1)))
print()
ref_cnt = Counter()
for r in refs:
    ref_cnt.update(OW.split_glued(OW.split_gloss_sequence(r)))
print("  参考最频 8 个（对照）：%s"
      % ", ".join("%s×%d" % (w, c) for w, c in ref_cnt.most_common(8)))
print()

# ----⑤ C 方向的前置判断 ----
print("=" * 88)
print("⑤ C 方向（词级辅助头）现在该不该重测？")
print("=" * 88)
print("  P85 判「噪声内」时的前提：输出长度 1.78 token/句、distinct 20")
print("  现在：输出长度 %.2f、distinct %d" % (np.mean(out_lens), len(all_toks)))
if p_len and p_dis:
    print("  ⇒ ✅ **前提已变**（输出不再是「只会说标点」的状态）")
    print("  ⇒ C 方向值得重测（当时的主任务 loss +4.90% 可能因基线太弱）")
else:
    print("  ⇒ ❌ 前提未变，仍是「欠输出」状态 ⇒ C 方向暂不重测")
print()
print("  标点占比 = %.1f%%（P84 时是 80%%）"
      % (100.0 * sum(c for w, c in all_toks.items() if w in "。？") / max(sum(out_lens), 1)))

out = {
    "experiment": "P97 output-pattern diagnosis",
    "checkpoint": CKPT.name,
    "ckpt_epoch": ck["epoch"],
    "ckpt_wer": ck["wer_official"],
    "out_len_mean": round(float(np.mean(out_lens)), 3),
    "ref_len_mean": round(float(np.mean(ref_lens)), 3),
    "distinct": len(all_toks),
    "coverage_pct": round(100.0 * len(all_toks) / max(len(ref_cnt), 1), 2),
    "passed_len_gt3": bool(p_len),
    "passed_dist_gt50": bool(p_dis),
    "breakthrough": bool(p_len and p_dis),
    "top10": all_toks.most_common(10),
    "ref_top8": ref_cnt.most_common(8),
    "punct_pct": round(100.0 * sum(c for w, c in all_toks.items() if w in "。？")
                       / max(sum(out_lens), 1), 1),
    "miss_rate_by_freq": {b: {"dev_tokens": tot[b], "hit": hit[b],
                              "miss_pct": round(100.0 * (1 - hit[b] / tot[b]), 1)}
                          for b in (">=100", "11-100", "1-10") if tot[b]},
    "history": [{"exp": "P84-600", "out_len": 0.95, "distinct": 12, "wer": 89.06},
                {"exp": "P87-2000", "out_len": 1.78, "distinct": 20, "wer": 80.96},
                {"exp": "P94-4973-30ep", "out_len": round(float(np.mean(out_lens)), 3),
                 "distinct": len(all_toks), "wer": 62.49}],
    "c_direction_premise_changed": bool(p_len and p_dis),
}
dst = REPO / "artifacts/metrics/blank-gov/p97-output-diagnosis.json"
dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("收据 -> %s" % dst.name)