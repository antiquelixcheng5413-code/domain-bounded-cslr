#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P205 / E4 前置探针：landmark 单模态可识别性（独立 CTC 头，快速训练）

目的：判断 `data/processed/ce-csl-landmark48`（每条 clip 固定 48 帧 × 368 维
MediaPipe 特征）是否携带足够的 gloss 信息。若单模态 WER 明显高于 RGB 基线
（34.34%），说明融合的先验价值不足，E4 应停止，不必投入管线改动。

不做融合、不碰 RGB 主干、不加载图像。

用法：
  venv/bin/python tools/blank_gov/p205_landmark_only.py --epochs 2 --seed 42
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import p78_train_official_tfnet as P  # noqa: E402
from official_wer import evaluate  # noqa: E402

LM_ROOT = REPO / "data/processed/ce-csl-landmark48"
OUTDIR = REPO / "artifacts/metrics/blank-gov"


def log(*a):
    print("[E4-probe %s] %s" % (time.strftime("%H:%M:%S"),
                               " ".join(str(x) for x in a)), flush=True)


class LmEncoder(nn.Module):
    """[B,48,368] -> [48,B,C] log_softmax。全 fp32，不进 autocast。"""

    def __init__(self, dim=368, hidden=256, n_class=3516, drop=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(drop))
        self.rnn = nn.LSTM(hidden, hidden // 2, num_layers=1,
                           batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden, n_class)

    def forward(self, x):                       # [B,T,D]
        h, _ = self.rnn(self.proj(x))           # [B,T,H]
        return torch.log_softmax(self.head(h), dim=-1).transpose(0, 1)


def load_bank(sids, root=LM_ROOT):
    """一次读入 [N,48,368]（float32，约 350 MB）；缺文件记 0 行。"""
    t0 = time.time()
    arr = np.zeros((len(sids), 48, 368), dtype=np.float32)
    miss = 0
    for i, sid in enumerate(sids):
        p = root / ("%s.npy" % sid)
        if p.exists():
            arr[i] = np.load(p)
        else:
            miss += 1
    log("bank: %d 条，缺失 %d，读入 %.1fs（%.0f MB）"
        % (len(sids), miss, time.time() - t0, arr.nbytes / 1e6))
    return arr


def greedy(logp, idx2word, blank=0):
    """[T,B,C] -> [ [word,...] ]（官方 collapse 口径）。"""
    T, B, _ = logp.shape
    best = logp.argmax(dim=-1)                  # [T,B]
    out = []
    for b in range(B):
        seq, prev = [], blank
        for t in range(T):
            k = int(best[t, b])
            if k != prev and k != blank:
                seq.append(idx2word[k] if k < len(idx2word) else "<unk>")
            prev = k
        out.append(seq)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-dev", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    word2idx, word_set_num, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    n_class = word_set_num + 1
    tr = P.read_csv_split("train.csv")
    dv = P.read_csv_split("dev.csv")
    tr_ids = list(tr.keys())
    dv_ids = list(dv.keys())
    if a.max_train:
        tr_ids = tr_ids[:a.max_train]
    if a.max_dev:
        dv_ids = dv_ids[:a.max_dev]

    tr_bank = load_bank(tr_ids)
    dv_bank = load_bank(dv_ids)
    mu = tr_bank.mean(axis=(0, 1), keepdims=True)
    sd = tr_bank.std(axis=(0, 1), keepdims=True) + 1e-6

    model = LmEncoder(n_class=n_class).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    log("landmark-only：train=%d dev=%d  C=%d  参数量=%.2fM"
        % (len(tr_ids), len(dv_ids), n_class,
           sum(p.numel() for p in model.parameters()) / 1e6))

    for ep in range(1, a.epochs + 1):
        model.train()
        perm = np.random.permutation(len(tr_ids))
        t0, tot = time.time(), 0.0
        for s in range(0, len(perm), a.batch):
            idx = perm[s:s + a.batch]
            x = (tr_bank[idx] - mu) / sd
            x = torch.from_numpy(x).cuda()
            tg = [tr[tr_ids[i]][1] for i in idx]
            ids = [[word2idx[w] for w in P.DPM.PreWords(t.split("/"))
                    if w in word2idx] for t in tg]
            keep = [k for k, v in enumerate(ids) if v]
            x = x[keep]
            tgt = torch.LongTensor([w for k in keep for w in ids[k]]).cuda()
            tgt_len = torch.LongTensor([len(ids[k]) for k in keep]).cuda()
            lens = torch.full((len(keep),), 48, dtype=torch.long).cuda()
            lp = model(x)
            loss = ctc(lp, tgt, lens, tgt_len)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
        # dev greedy
        model.eval()
        hyps = []
        with torch.no_grad():
            for s in range(0, len(dv_ids), a.batch):
                idx = list(range(s, min(s + a.batch, len(dv_ids))))
                x = (dv_bank[idx] - mu) / sd
                x = torch.from_numpy(x).cuda()
                hyps.extend(greedy(model(x).float(), idx2word))
        refs = [dv[sid][1] for sid in dv_ids]
        wer = evaluate(refs, hyps)
        log("ep%d  ctc=%.3f  dev WER=%.2f%%  (%.0fs)"
            % (ep, tot / max(1, (len(perm) + a.batch - 1) // a.batch),
               wer["WER_official"], time.time() - t0))
        OUTDIR.mkdir(parents=True, exist_ok=True)
        (OUTDIR / ("p205-landmark-only-ep%d.json" % ep)).write_text(
            __import__("json").dumps({"epoch": ep, "dev_wer": wer},
                                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
