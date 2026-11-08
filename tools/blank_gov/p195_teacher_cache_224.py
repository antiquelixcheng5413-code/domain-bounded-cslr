#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P195-step1：冻结 TFNet 发布权重 @224 作教师，缓存 train 全部样本的教师 logits。

用途：P195 跨分辨率蒸馏（同骨架 TFNet@224 教师 → TFNet@160 学生）。
与 P174 缓存的区别：img_size=224、只缓存 train、**可断点续跑**
（每条样本落一个 .npy + 追加一行 teacher_meta.jsonl，CUDA 故障后重跑自动跳过）。

时间对齐：canonical_T = _conv_len(dl)，与学生侧 lgt 逐样本一致 ⇒ 训练时无需插值。

用法
  python tools/blank_gov/p195_teacher_cache_224.py [--limit N] [--img 224]
"""
import argparse
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
OUT_DIR = REPO / "artifacts/cache/p195_teacher"
ARR = OUT_DIR / "teacher_logits"
PROG = OUT_DIR / "teacher_meta.jsonl"
META = OUT_DIR / "teacher_meta.json"
ARR.mkdir(parents=True, exist_ok=True)


def log(*a):
    print("[P195cache %s] %s" % (time.strftime("%H:%M:%S"),
                                 " ".join(str(x) for x in a)), flush=True)


def load_done() -> dict:
    done = {}
    if PROG.exists():
        for line in PROG.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[rec["sid"]] = [rec["split"], rec["T_raw"],
                                rec["canonical_T"], rec["dl"]]
    return done


def write_meta(done: dict, vocab: int, img: int) -> None:
    meta = {"vocab": vocab, "img": img, "n_train": -1, "index": done}
    META.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P195 教师缓存（TFNet@224）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只缓存前 N 条（0=全部），用于冒烟")
    ap.add_argument("--img", type=int, default=224)
    a = ap.parse_args()
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
    log("载入 TFNet 发布权重（strict），冻结；img_size=%d" % a.img)

    _, tf = P.build_transforms(hflip=False, img_size=a.img)
    tr_labels = P.read_csv_split("train.csv")
    tr_ds = P.RGBSeqDataset("train", tr_labels, word2idx, tf, False)
    done = load_done()
    n_run = len(tr_ds) if a.limit <= 0 else min(a.limit, len(tr_ds))
    log("train=%d（已缓存 %d，本次最多 %d）" % (len(tr_ds), len(done), n_run))

    t0 = time.time()
    new = 0
    with PROG.open("a", encoding="utf-8") as fh:
        for i in range(n_run):
            batch = P.collate([tr_ds[i]])
            vid, _t, _tl, dl, _tl_len, sids, _ids = batch
            sid = str(sids[0])
            if sid in done:
                continue
            try:
                with torch.no_grad():
                    out = model(vid.to(dev), dl, False)
                logits = out[0].float().cpu()[:, 0, :]   # [T_raw, vocab]
            except torch.AcceleratorError as exc:
                log("CUDA 故障，保存进度并退出：%s" % str(exc).splitlines()[0])
                write_meta(done, vocab, a.img)
                return 2
            T_raw = int(logits.shape[0])
            canonical_T = int(P._conv_len(int(dl.view(-1)[0])))
            assert T_raw >= canonical_T, ("T_raw %d < canonical %d @%s"
                                          % (T_raw, canonical_T, sid))
            np.save(str(ARR / ("%s.npy" % sid)), logits.numpy())
            rec = {"sid": sid, "split": "train", "T_raw": T_raw,
                   "canonical_T": canonical_T, "dl": int(dl.view(-1)[0])}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[sid] = ["train", T_raw, canonical_T, int(dl.view(-1)[0])]
            new += 1
            if new % 200 == 0:
                log("  +%d（累计 %d/%d，%.0fs）"
                    % (new, len(done), len(tr_ds), time.time() - t0))

    write_meta(done, vocab, a.img)
    log("教师缓存完成：%d/%d（本次新增 %d）"
        % (len(done), len(tr_ds), new))
    return 0 if len(done) >= len(tr_ds) else 3


if __name__ == "__main__":
    sys.exit(main())
