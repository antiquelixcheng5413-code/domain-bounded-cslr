"""P87 验收：判断 160 分辨率 + 2000 条是否突破了「输出坍缩」阶段。

🔴 P84 判据（本项目当前的判优前提）：
  训练量不足时模型只输出高频标点，任何架构/增强实验都无法显现效果。
  必须先看到「输出长度 > 3 token/句」才认为模型开始学真正的内容。

  P72(landmark, 4973条, 崩于ep22) : 2.13 token/句, distinct 22
  P84(VAC RGB,  600条, 6ep)      : 0.95 token/句, distinct 12
  本实验(VAC RGB, 2000条, 160res, 6ep) = ?

判据（事先定好，避免事后找理由）：
  输出长度 > 3 token/句 且 distinct > 50 ⇒ 突破了坍缩阶段，可进入方向对比
  否则 ⇒ 训练量仍不足，继续加大规模

用法：等训练收据出现后运行
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))

TAG = sys.argv[1] if len(sys.argv) > 1 else "p87-res160-2k"
CKPT = REPO / ("artifacts/checkpoints/%s-best.pt" % TAG)
if not CKPT.exists():
    print("❌ checkpoint 未出现：%s" % CKPT)
    raise SystemExit(1)

import csv

import DataProcessMoudle as DPM
import Net
import importlib.util

spec = importlib.util.spec_from_file_location(
    "p78", REPO / "tools/blank_gov/p78_train_official_tfnet.py")
p78 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p78)

CSV = REPO / "external/TFNet/data/CE-CSL"
lab_dv, tr_map = {}, {}
with open(CSV / "dev.csv", newline="", encoding="utf-8") as fh:
    for row in csv.reader(fh):
        if row and row[0]:
            lab_dv[row[0]] = row[3]
            tr_map[row[0]] = row[1]

w2i, wsn, idx2w = DPM.Word2Id(str(CSV / "train.csv"), str(CSV / "dev.csv"),
                             str(CSV / "test.csv"), "CE-CSL")
freq = __import__("collections").Counter()
with open(CSV / "train.csv", newline="", encoding="utf-8") as fh:
    for row in csv.reader(fh):
        if row and row[0]:
            freq.update(DPM.PreWords(row[3].split("/")))

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("checkpoint epoch=%s wer_official=%s"
      % (ck.get("epoch"), ck.get("wer_official")))

model = Net.moduleNet(ck["hidden"], ck["wordSetNum"] + 1, ck["config"]["module"],
                      torch.device("cuda:0"), "CE-CSL", True).cuda()
model.load_state_dict(ck["model_state"])
model.eval()

_, dv_tf = p78.build_transforms(hflip=False,
img_size=ck["config"].get("img_size", 160))
ds = p78.RGBSeqDataset("dev", {k: (tr_map[k], v) for k, v in lab_dv.items()},
                       w2i, dv_tf, False)
print("dev %d 条，img_size=%s" % (len(ds), ck["config"].get("img_size", 160)))

ls = nn.LogSoftmax(dim=-1)
hyps, refs = [], []
with torch.no_grad():
    for k in range(0, len(ds), 2):
        ch = [ds[i] for i in range(k, min(k + 2, len(ds)))]
        vid, tgt, tl, dl, true_len, sids, _ids = p78.collate(ch)
        out = model(vid.cuda(), dl, False)
        lp = ls(out[0])
        hyps += p78.greedy_decode(lp, true_len, idx2w)
        for s in sids:
            refs.append(lab_dv[s])

from collections import Counter
from official_wer import evaluate, split_gloss_sequence
import numpy as np

o = evaluate(refs, hyps)
print()
print("=" * 72)
print("P87 验收（判据事先定好）")
print("=" * 72)
for k in ("WER_official", "edits", "ref_tokens", "exact_rate_pct"):
    if k in o:
        print("  %-16s %s" % (k, o[k]))

hl = [len(h) for h in hyps]
rl = [len(split_gloss_sequence(r)) for r in refs]
allh = [t for h in hyps for t in h]
c = Counter(allh)

print()
print("=== 核心判据：输出长度 ===")
print("  输出 token mean = %.2f（参考 %.2f）" % (np.mean(hl), np.mean(rl)))
print("  distinct = %d   覆盖率 = %.1f%%"
      % (len(c), 100.0 * len(c) / max(len(allh), 1)))
print()
print("=== 与历史对比 ===")
print("  P72 landmark 4973条(崩) : 2.13 token/句  distinct 22")
print("  P84 RGB 600条 224res     : 0.95 token/句  distinct 12")
print("  P87 RGB 2000条 160res    : %.2f token/句  distinct %d"
      % (np.mean(hl), len(c)))
print()

passed_len = np.mean(hl) > 3.0
passed_dist = len(c) > 50
print("=== 判定 ===")
print("  输出长度 > 3 token/句 : %s (实测 %.2f)"
      % ("✓ 过" if passed_len else "✗ 未过", np.mean(hl)))
print("  distinct > 50: %s (实测 %d)"
      % ("✓ 过" if passed_dist else "✗ 未过", len(c)))
if passed_len and passed_dist:
    print("  ⇒ **突破坍缩阶段** ⇒ 可以开始做方向对比实验（C/A/B 都有意义了）")
else:
    print("  ⇒ **仍在坍缩阶段** ⇒ 训练量仍不足，方向对比实验暂时无意义")
print()
print("最高频 10 输出:", c.most_common(10))
rt = [t for r in refs for t in w2i]
print("参考 distinct = %d / %d token" % (len(set(rt)), len(rt)))

out = {
    "tag": TAG, "epoch": ck.get("epoch"),
    "img_size": ck["config"].get("img_size"),
    "max_train": ck["config"].get("max_train"),
    "wer_official": o["WER_official"],
    "out_len_mean": round(float(np.mean(hl)), 3),
    "distinct": len(c),
    "coverage_pct": round(100.0 * len(c) / max(len(allh), 1), 2),
    "passed_len_gt3": bool(passed_len),
    "passed_dist_gt50": bool(passed_dist),
    "top10": c.most_common(10),
}
dst = REPO / ("artifacts/metrics/blank-gov/%s-accept.json" % TAG)
dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("收据 -> %s" % dst)