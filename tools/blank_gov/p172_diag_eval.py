#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P172 dev_eval 诊断：复现 dev_eval 解码链路，打印中间值，定位 WER=0/gen=0。
不依赖训练 ckpt（用随机 VAC 即可，重点查 refs/decode/evaluate 通路）。
"""
import sys, time
from pathlib import Path
import torch, numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))
import p78_train_official_tfnet as P
from official_wer import evaluate

dev = "cuda"
word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
    str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
    str(P.CSV_DIR / "test.csv"), "CE-CSL")
vocab = wordSetNum + 1
labels_dv = P.read_csv_split("dev.csv")
print("[DIAG] vocab=%d  n_dev_labels=%d  sample keys=%s"
      % (vocab, len(labels_dv), list(labels_dv.keys())[:3]))

P.OFFICIAL_PAD = True; P.RGB_FIX = True
tr_tf, dv_tf = P.build_transforms(hflip=True, img_size=160)
dv_ds = P.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
dv_ld = torch.utils.data.DataLoader(dv_ds, batch_size=1, shuffle=False,
                                    collate_fn=P.collate, num_workers=0)
print("[DIAG] n_dev_batches=%d" % len(dv_ld))

model = P.Net.moduleNet(1024, vocab, "VAC", torch.device(dev),
                        "CE-CSL", True).to(dev)
ls = torch.nn.LogSoftmax(dim=-1)

refs, hyps, ntok = [], [], 0
n_b = 0
with torch.no_grad():
    for vid, tgt, tgt_len, dl, _tl, sids, _ids in dv_ld:
        vid = vid.to(dev)
        out = model(vid, dl, False)
        logp = ls(out[0].float().cpu())
        P.BLANK_BIAS = 0.0
        hyps += P.greedy_decode(logp, dl, idx2word)
        ntok += sum(len(q) for q in hyps[-len(sids):])
        for b in sids:
            refs.append(labels_dv[b][1])
        n_b += 1
        if n_b <= 3:
            print("[DIAG] batch%d sid=%s  dl=%s  out0_shape=%s  hyp=%s  ref=%s"
                  % (n_b, sids, dl.tolist(), tuple(out[0].shape),
                     hyps[-1], labels_dv[str(sids[0])][1]))
        if n_b >= 8:
            break

print("[DIAG] processed batches=%d  n_refs=%d  n_hyps=%d  ntok=%d"
      % (n_b, len(refs), len(hyps), ntok))
o = evaluate(refs, hyps)
print("[DIAG] evaluate ->", {k: o[k] for k in ("WER_official","edits","ref_tokens","n_sentences")})
gen = 100.0 * ntok / max(o["ref_tokens"], 1)
print("[DIAG] computed gen=%.1f%%" % gen)
print("[DIAG] sample ref[0]=%r  hyp[0]=%r" % (refs[0], hyps[0]))
