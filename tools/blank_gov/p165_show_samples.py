#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P165：抽 5 条 dev 的真实识别结果（ref vs hyp），带逐 token 标注

怎么选的（防"挑好看的"）：
  跑完整 dev 515 条，算每条的 WER，按 WER 排序后取
  **10% / 30% / 50% / 70% / 90% 分位** 各一条 ⇒ 好/中/差都覆盖。

标注含义：
  =  命中（ref 与 hyp 一致）
  D  漏词（ref 有，hyp 没有）
  S  替换（ref 有，hyp 说是别的）
  +  多说（hyp 多出来的）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import p78_train_official_tfnet as P  # noqa: E402
from official_wer import evaluate  # noqa: E402
from official_wer import split_gloss_sequence, split_glued, CFG  # noqa: E402

CK = REPO / "artifacts/checkpoints/p150-rgbfix-best.pt"


def log(*a):
    print("[P165 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def align(ref, hyp):
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + c)
    tags, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            if D[i][j] == D[i - 1][j - 1] + c:
                tags.append(("S" if c else "=", ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and D[i][j] == D[i - 1][j] + 1:
            tags.append(("D", ref[i - 1], "—"))
            i -= 1
            continue
        tags.append(("+", "—", hyp[j - 1]))
        j -= 1
    tags.reverse()
    return tags


def main() -> int:
    dev = "cuda"
    P.RGB_FIX = True                    # ★ P150 是 RGB 训练的
    labels = P.read_csv_split("dev.csv")
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")

    ck = torch.load(CK, map_location="cpu", weights_only=False)
    assert "model_state" in ck, "键名变了：%s" % list(ck)[:8]
    P.OFFICIAL_PAD = bool(ck.get("config", {}).get("official_pad", False))
    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    log("载入 %s（epoch %s）" % (CK.name, ck.get("epoch")))

    _, tf = P.build_transforms(hflip=False, img_size=160)
    ds = P.RGBSeqDataset("dev", labels, word2idx, tf, False)
    ls = torch.nn.LogSoftmax(dim=-1)

    cache, lens, refs, sids_all = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            batch = P.collate([ds[i]])
            vid, _t, _tl, dl, _tl_len, sids, _ids = batch
            out = model(vid.to(dev), dl, False)
            lp = ls(out[0])[:, 0, :]
            L = min(P._conv_len(int(dl.view(-1)[0])), lp.shape[0])
            cache.append(lp[:L].to("cpu", torch.float16))
            lens.append(int(dl.view(-1)[0]))
            sids_all.append(list(sids))
            for b in sids:
                refs.append(labels[b][1])
    log("dev %d 条推理 %.0fs" % (len(ds), time.time() - t0))

    def decode(bias):
        hyps, ntok = [], 0
        P.BLANK_BIAS = float(bias)
        for lp16, L in zip(cache, lens):
            lp = lp16.float().unsqueeze(1)
            if bias:
                lp = lp.clone()
                lp[..., 0] -= bias
            t = P.greedy_decode(lp, torch.LongTensor([[L]]), idx2word)
            hyps += t
            ntok += sum(len(q) for q in t)
        P.BLANK_BIAS = 0.0
        return hyps, ntok

    rows = []
    for b in (0.0, 1.0):
        hyps, _n = decode(b)
        rows.append((b, hyps))

    refs_tok = [split_glued(split_gloss_sequence(r, CFG)) for r in refs]
    per = []
    for i, rt in enumerate(refs_tok):
        h0 = split_glued(rows[0][1][i])
        h1 = split_glued(rows[1][1][i])
        per.append({"i": i, "ref": rt, "hyp0": h0, "hyp1": h1,
                   "len": len(rt),
                   "wer0": evaluate([refs[i]], [rows[0][1][i]])["WER_official"],
                   "wer1": evaluate([refs[i]], [rows[1][1][i]])["WER_official"]})

    srt = sorted(per, key=lambda d: d["wer1"])
    picked = []
    for q in (0.10, 0.30, 0.50, 0.70, 0.90):
        picked.append(srt[min(len(srt) - 1, int(q * len(srt)))])

    print()
    log("──── 5 条真实样例（按 WER 分位取，非挑好看的）────")
    for d in picked:
        print()
        log("样例 #%d   参考长度 %d   WER(bias0)=%.2f%%   WER(bias1.0)=%.2f%%"
            % (d["i"], d["len"], d["wer0"], d["wer1"]))
        log("  参考 REF : %s" % " ".join(d["ref"]))
        log("  识别 HYP : %s" % " ".join(d["hyp1"]))
        tg = align(d["ref"], d["hyp1"])
        marks = " ".join("%s%s" % (t[0], t[1] if t[0] != "S" else (t[1] + ">" + t[2]))
                           for t in tg)
        log("  逐词     : %s" % marks)

    print()
    log("──── 逐句 WER 分布（dev 515）────")
    ws = sorted(d["wer1"] for d in per)
    for q in (0, 25, 50, 75, 100):
        log("  P%-3d  %6.1f%%" % (q, ws[min(len(ws) - 1, int(q / 100.0 * len(ws)))]))
    exact = sum(1 for d in per if d["ref"] == split_glued(list(d["hyp1"])))
    log("  整句全对 : %d / %d = %.2f%%" % (exact, len(per), 100.0 * exact / len(per)))

    out = REPO / "artifacts/metrics/blank-gov/p165-dev-samples.json"
    out.write_text(json.dumps({
        "ckpt": CK.name, "epoch": ck.get("epoch"), "rgb_fix": True,
        "n_dev": len(refs),
        "per_sample": per,
        "picked": [d["i"] for d in picked],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
