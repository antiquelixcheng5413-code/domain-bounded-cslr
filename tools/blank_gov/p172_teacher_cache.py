#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P172-step1：冻结 TFNet 发布权重@224 作教师，对 train+dev 全部样本前向，
逐样本缓存 teacher_logits 到 .npy，供学生蒸馏使用。

时间对齐（关键）
  教师 TFNet 在官方 OFFICIAL_PAD 口径下，原始 T 不恒定（49/65/102…），
  规范长度 canonical_T = _conv_len(dl) 随帧数变。p156 解码即 min(_conv_len, seq.shape[1])
  从前截 —— 本脚本对每个样本另存 canonical_T，训练时取前 canonical_T 帧，与学生(=canonical_T)对齐。

存储
  每个样本一个 .npy：<OUT>/teacher_logits/<sid>.npy  shape=[T_raw, vocab]
  meta.json：index[sid]=[split, T_raw, canonical_T, dl]

铁律
  1. 教师严格冻结（eval+no_grad），只前向一次，绝不训练。
  2. RGB_FIX=True + img_size=224（与 P156 零样本同口径）。
  3. 不在本脚本做任何训练/扫描。

用法
  python tools/blank_gov/p172_teacher_cache.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))

import p78_train_official_tfnet as P  # noqa: E402

CK = Path("/mnt/c/Users/su127/Desktop/中文手语识别/官方权重/"
          "TFNet-CE-CSL-CSLDaily-32.46.pth")
OUT_DIR = REPO / "artifacts/cache/p172_teacher"
ARR = OUT_DIR / "teacher_logits"
ARR.mkdir(parents=True, exist_ok=True)
META = OUT_DIR / "teacher_meta.json"


def log(*a):
    print("[P172cache %s] %s" % (time.strftime("%H:%M:%S"),
                                 " ".join(str(x) for x in a)), flush=True)


def main() -> int:
    dev = "cuda"
    word2idx, wordSetNum, _ = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    vocab = wordSetNum + 1
    log("词表 %d（vocab=%d）" % (wordSetNum, vocab))

    model = P.Net.moduleNet(1024, vocab, "TFNet",
                            torch.device(dev), "CE-CSL", True).to(dev)
    sd = torch.load(str(CK), map_location="cpu", weights_only=False,
                    mmap=True)["moduleNet_state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    P.OFFICIAL_PAD = True
    P.RGB_FIX = True
    log("载入 TFNet 发布权重（strict 逐键吻合），冻结")

    _, tf = P.build_transforms(hflip=False, img_size=224)

    tr_labels = P.read_csv_split("train.csv")
    dv_labels = P.read_csv_split("dev.csv")
    tr_ds = P.RGBSeqDataset("train", tr_labels, word2idx, tf, False)
    dv_ds = P.RGBSeqDataset("validation", dv_labels, word2idx, tf, False)
    n_tr, n_dv = len(tr_ds), len(dv_ds)
    log("train=%d  dev=%d" % (n_tr, n_dv))

    meta = {"vocab": vocab, "n_train": n_tr, "n_dev": n_dv, "index": {}}
    t0 = time.time()

    def cache_one(ds, split_key):
        for i in range(len(ds)):
            batch = P.collate([ds[i]])
            vid, _t, _tl, dl, _tl_len, sids, _ids = batch
            with torch.no_grad():
                out = model(vid.to(dev), dl, False)
            logits = out[0].float().cpu()[:, 0, :]      # [T_raw, vocab]
            T_raw = int(logits.shape[0])
            canonical_T = int(P._conv_len(int(dl.view(-1)[0])))
            assert T_raw >= canonical_T, ("T_raw %d < canonical %d @%s"
                                         % (T_raw, canonical_T, sids[0]))
            sid = str(sids[0])
            np.save(str(ARR / ("%s.npy" % sid)), logits.numpy())
            meta["index"][sid] = [split_key, T_raw, canonical_T,
                                 int(dl.view(-1)[0])]
            if (i + 1) % 200 == 0 or (i + 1) == len(ds):
                log("  %s %d/%d（%.0fs, GPU %.0fMiB，maxT_raw=%d）"
                    % (split_key, i + 1, len(ds), time.time() - t0,
                       torch.cuda.memory_allocated(dev) / 1e6, T_raw))

    cache_one(tr_ds, "train")
    cache_one(dv_ds, "dev")

    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    log("meta -> %s（index=%d）" % (META, len(meta["index"])))
    log("✅ 教师缓存完成：train=%d dev=%d，vocab=%d" % (n_tr, n_dv, vocab))
    return 0


if __name__ == "__main__":
    sys.exit(main())
