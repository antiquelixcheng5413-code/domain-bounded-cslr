#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P174-step1：冻结 TFNet 发布权重@160 作教师，对 train+dev 全部样本前向，
逐样本缓存 teacher_logits 到 .npy，供 P174 同分辨率软标签 KD 使用。

与 P172 的唯一区别：教师输入 img_size=160（与学生 VAC@160 同分辨率），
消除 P172 跨分辨率(224→160)导致的分布/对齐不匹配。

时间对齐：canonical_T=_conv_len(dl)；训练时取前 canonical_T 帧与学生对齐。

铁律
  1. 教师严格冻结（eval+no_grad），只前向一次，绝不训练。
  2. RGB_FIX=True + img_size=160（与 P156 零样本 160 口径一致）。
  3. split 名统一用 "dev"（与训练脚本一致，避免 sid 错位）。
  4. 不在本脚本做任何训练/扫描。

用法
  python tools/blank_gov/p174_teacher_cache.py
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
OUT_DIR = REPO / "artifacts/cache/p174_teacher"
ARR = OUT_DIR / "teacher_logits"
ARR.mkdir(parents=True, exist_ok=True)
META = OUT_DIR / "teacher_meta.json"


def log(*a):
    print("[P174cache %s] %s" % (time.strftime("%H:%M:%S"),
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

    _, tf = P.build_transforms(hflip=False, img_size=160)

    tr_labels = P.read_csv_split("train.csv")
    dv_labels = P.read_csv_split("dev.csv")
    tr_ds = P.RGBSeqDataset("train", tr_labels, word2idx, tf, False)
    dv_ds = P.RGBSeqDataset("dev", dv_labels, word2idx, tf, False)
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
