#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P174 冒烟：验证教师缓存核心管线（TFNet@160 权重加载 + 数据集 + 单条前向 + 时间维）。"""
import sys
import numpy as np
import torch
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))

import p78_train_official_tfnet as P  # noqa: E402

CK = Path("/mnt/c/Users/su127/Desktop/中文手语识别/官方权重/"
          "TFNet-CE-CSL-CSLDaily-32.46.pth")


def log(*a):
    print("[P174smoke]", " ".join(str(x) for x in a), flush=True)


def main():
    dev = "cuda"
    word2idx, wordSetNum, _ = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    vocab = wordSetNum + 1
    model = P.Net.moduleNet(1024, vocab, "TFNet", torch.device(dev),
                           "CE-CSL", True).to(dev)
    sd = torch.load(str(CK), map_location="cpu", weights_only=False,
                    mmap=True)["moduleNet_state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    P.OFFICIAL_PAD = True
    P.RGB_FIX = True
    log("TFNet@160 权重 strict 加载 OK")

    _, tf = P.build_transforms(hflip=False, img_size=160)
    tr_ds = P.RGBSeqDataset("train", P.read_csv_split("train.csv"),
                           word2idx, tf, False)
    dv_ds = P.RGBSeqDataset("dev", P.read_csv_split("dev.csv"),
                           word2idx, tf, False)
    log("train=%d dev=%d vocab=%d" % (len(tr_ds), len(dv_ds), vocab))

    for name, ds in [("train", tr_ds), ("dev", dv_ds)]:
        batch = P.collate([ds[0]])
        vid, _t, _tl, dl, _tl_len, sids, _ids = batch
        with torch.no_grad():
            out = model(vid.to(dev), dl, False)
        logits = out[0].float().cpu()[:, 0, :]
        T_raw = int(logits.shape[0])
        canonical_T = int(P._conv_len(int(dl.view(-1)[0])))
        log("%s sid=%s T_raw=%d canonical_T=%d vocab=%d 前向OK"
            % (name, str(sids[0]), T_raw, canonical_T, vocab))
    log("SMOKE_OK")


if __name__ == "__main__":
    main()
