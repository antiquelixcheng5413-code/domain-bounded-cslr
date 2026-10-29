#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P183 (v2): 推理时修复 VAC@160 欠生成 —— 全量 515 确认版。

两阶段合一：
  默认：循环前向 dev（每片 50 条，分片存盘 artifacts/cache/p183_logits_{s}_{e}.pt，
        已存在的片自动跳过 ⇒ 抗 WSL 崩溃续跑）；前向完自动解码。
  仅解码：P183_DECODE=1 ⇒ 跳过前向，直接加载所有分片跑解码网格。

解码网格（仅 greedy 族，beam+lp 已在 v1 确认死路）：
  greedy bias∈{0, 0.5, 1.0, 2.0}
指标：WER（official_wer，语料级）、生成率、单句号率；对比官方 53.73（baseline 应≈该值⇒路径正确）。
"""
import sys, os, time, glob
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (REPO / "tools" / "blank_gov", REPO / "external" / "TFNet", REPO / "src"):
    sys.path.insert(0, str(p))
os.environ.setdefault("CSLR_TFNET_DEVICE", "cuda")

import torch
import p78_train_official_tfnet as P
import p147_beam_decode as B
from official_wer import evaluate

P.OFFICIAL_PAD = True
P.RGB_FIX = True
P.BLANK_BIAS = 0.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLANK = B.BLANK
CACHE_DIR = REPO / "artifacts" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_TOTAL = int(os.environ.get("P183_N", "515"))
CHUNK = int(os.environ.get("P183_CHUNK", "50"))
DO_DECODE = os.environ.get("P183_DECODE", "") == "1"
print("device=", DEV, "N_TOTAL=", N_TOTAL, "CHUNK=", CHUNK, "DECODE=", DO_DECODE, flush=True)

word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
    str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
    str(P.CSV_DIR / "test.csv"), "CE-CSL")
ls = torch.nn.LogSoftmax(dim=-1)
ckpt = torch.load(str(REPO / "artifacts/checkpoints/p150-rgbfix-best.pt"),
                  map_location="cpu", weights_only=False)
model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                       torch.device(DEV), "CE-CSL", True).to(DEV)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()
tf = P.build_transforms(hflip=False, img_size=160)[1]
labels = P.read_csv_split("dev.csv")
ds = P.RGBSeqDataset("dev", labels, word2idx, tf, False)

if not DO_DECODE:
    for s in range(0, N_TOTAL, CHUNK):
        e = min(s + CHUNK, N_TOTAL)
        fn = CACHE_DIR / ("p183_logits_%d_%d.pt" % (s, e))
        if fn.exists():
            print("  skip fwd %d-%d (cached)" % (s, e), flush=True)
            continue
        cache, lens, refs = [], [], []
        t0 = time.time()
        with torch.no_grad():
            for i in range(s, e):
                batch = P.collate([ds[i]])
                vid, _, _, dl, _, sids, _ = batch
                out = model(vid.to(DEV), dl, False)
                lp = ls(out[0])[:, 0, :]
                L = int(dl.view(-1)[0])
                Lc = min(P._conv_len(L), lp.shape[0])
                cache.append(lp[:Lc].to("cpu", torch.float32))
                lens.append(L)
                for b in sids:
                    refs.append(labels[b][1])
        torch.save({"lp": cache, "lens": lens, "refs": refs}, fn)
        print("  fwd %d-%d %.0fs" % (s, e, time.time() - t0), flush=True)
    print("all forward chunks done", flush=True)

# ---- 解码（从所有分片加载）----
all_lp, all_lens, all_refs = [], [], []
for fn in sorted(glob.glob(str(CACHE_DIR / "p183_logits_*.pt"))):
    d = torch.load(fn, map_location="cpu", weights_only=False)
    all_lp += d["lp"]
    all_lens += d["lens"]
    all_refs += d["refs"]
print("loaded %d clips for decode" % len(all_lp), flush=True)


def gen_rate(hyps, ref_tokens):
    return 100.0 * sum(len(h) for h in hyps) / max(ref_tokens, 1)


def single_period(hyps):
    return sum(1 for h in hyps if h == ["。"]) / max(len(hyps), 1)


def report(tag, hyps):
    o = evaluate(all_refs, hyps)
    g = gen_rate(hyps, o["ref_tokens"])
    sp = single_period(hyps)
    print("  %-18s WER %7.2f%%  gen %6.1f%%  single_period %5.1f%%"
          % (tag, o["WER_official"], g, 100 * sp), flush=True)
    return o["WER_official"]


def greedy_bias(lp, L, bias):
    if bias:
        x = lp[:L].clone()
        x[:, BLANK] -= bias
    else:
        x = lp[:L]
    return P.greedy_decode(x.unsqueeze(1), torch.LongTensor([[L]]), idx2word)


print("==== greedy + blank penalty（全量 515）====", flush=True)
res = {}
for bias in (0.0, 0.5, 1.0, 2.0):
    hyps = []
    for lp, L in zip(all_lp, all_lens):
        hyps += greedy_bias(lp, L, bias)
    res[bias] = report("greedy bias=%.1f" % bias, hyps)

base = res[0.0]
print("==== Δ vs baseline（官方 53.73 应≈ baseline 路径校验）====", flush=True)
for bias in (0.5, 1.0, 2.0):
    print("  bias=%.1f  ΔWER %+.2f pp" % (bias, res[bias] - base), flush=True)
