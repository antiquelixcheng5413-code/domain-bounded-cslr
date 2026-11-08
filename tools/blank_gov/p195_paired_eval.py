#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P195 终评：P184(freeze-best) vs P195(kd-best) 同一 dev 口径对比。

输出：语料级 WER、逐句均值 WER、配对 bootstrap 95% CI（向量化，秒级）。
解码与 p78 _dev_eval 完全一致（log_softmax(out[0]) + greedy + blank=0）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(p))

import p78_train_official_tfnet as P78  # noqa: E402

P78.OFFICIAL_PAD = True
P78.RGB_FIX = True
P78.BLANK_BIAS = 0.0
DEV = "cuda"

CKPTS = {
    "p184": REPO / "artifacts/checkpoints/p184-tfnet-freeze-best.pt",
    "p195": REPO / "artifacts/checkpoints/p195-kd160-best.pt",
}
OUT = REPO / "artifacts/metrics/blank-gov/p195-paired-dev.json"


def lev(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def decode_all(ck, labels_dv, word2idx, wordSetNum, idx2word, loader):
    mdl = P78.Net.moduleNet(1024, wordSetNum + 1, "TFNet", DEV, "CE-CSL", True)
    sd = torch.load(str(ck), map_location=DEV, weights_only=False)
    sd = sd.get("model_state", sd)
    mdl.load_state_dict(sd, strict=False)
    mdl = mdl.to(DEV).eval()
    hyps = {}
    with torch.no_grad():
        for vid, _t, _tl, dl, _tll, sids, _ids in loader:
            out = mdl(vid.to(DEV), dl, False)
            logp = F.log_softmax(out[0], dim=-1)
            got = P78.greedy_decode(logp, dl.view(-1), idx2word)
            for bi, sid in enumerate(sids):
                h = got[bi]
                hyps[str(sid)] = " ".join(h) if isinstance(h, list) else h
    return hyps


def main():
    labels_dv = P78.read_csv_split("dev.csv")
    word2idx, wordSetNum, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    tf = P78.build_transforms(hflip=False, img_size=160, y_shift=0,
                              temporal_rescale=0.0)[1]
    ds = P78.RGBSeqDataset("dev", labels_dv, word2idx, tf, False)
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=P78.collate, num_workers=0)
    print("dev items = %d, vocab = %d" % (len(ds), wordSetNum + 1), flush=True)

    hyps = {tag: decode_all(ck, labels_dv, word2idx, wordSetNum, idx2word,
                            loader) for tag, ck in CKPTS.items()}
    sids = sorted(hyps["p184"])
    refs = [labels_dv[s][1] for s in sids]
    r_tok = [r.strip().strip("/").split("/") for r in refs]
    L = np.array([max(1, len(t)) for t in r_tok])
    E = {}
    for tag in ("p184", "p195"):
        E[tag] = np.array([lev(t, hyps[tag][s].split())
                           for t, s in zip(r_tok, sids)])

    n = len(sids)
    rng = np.random.default_rng(20261104)
    idx = rng.integers(0, n, size=(20000, n))
    out = {"n": n, "sids": sids}
    stats = {}
    for tag in ("p184", "p195"):
        corpus = float(E[tag].sum() / L.sum() * 100)
        sent = float((E[tag] / L).mean() * 100)
        stats[tag] = {"corpus": corpus, "sent": sent}
        print("%s  语料级 %.2f  逐句均值 %.2f" % (tag, corpus, sent), flush=True)

    d_sent = (E["p184"] - E["p195"]) / L
    d_corp_s = ((E["p184"][idx].sum(1) / L[idx].sum(1))
                - (E["p195"][idx].sum(1) / L[idx].sum(1))) * 100
    d_sent_b = d_sent[idx].mean(1) * 100
    out.update({
        "p184": stats["p184"], "p195": stats["p195"],
        "delta_corpus": {"mean": float(d_corp_s.mean()),
                         "ci": [float(np.percentile(d_corp_s, 2.5)),
                                float(np.percentile(d_corp_s, 97.5))]},
        "delta_sent": {"mean": float(d_sent_b.mean()),
                       "ci": [float(np.percentile(d_sent_b, 2.5)),
                              float(np.percentile(d_sent_b, 97.5))],
                       "better": int((d_sent > 0).sum()),
                       "worse": int((d_sent < 0).sum()),
                       "tie": int((d_sent == 0).sum())},
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "sids"},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
