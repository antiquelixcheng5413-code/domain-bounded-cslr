#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P224 / H1-a：导出 CTC greedy 段级实例特征，构建 train 记忆库 / dev 查询段。

段 = 连续相同的非 blank greedy 路径；段特征 = 该段 classifier11 输入帧特征的均值 + L2。
train 输出只含 Z/Y（记忆库），dev 额外含 SID/SE/NT（用于后验融合定位）。
判据与参数见 docs/planning/H1-PREREG-2026-11-08.md（启动前写死）。

用法：venv/bin/python -u tools/blank_gov/p224_h1_dump.py --split train|dev
断点：H1_RESUME=1 从 artifacts/cache/p224_<split>_partial.pkl 续跑
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78  # noqa: E402
from p197_beam_lm import build_model  # noqa: E402
from csgc_module import FeatureCapture  # noqa: E402

P78.OFFICIAL_PAD = True
P78.RGB_FIX = True
P78.BLANK_BIAS = 0.0
CACHE = REPO / "artifacts/cache"
CKPT = REPO / "artifacts/checkpoints/p184-tfnet-freeze-best.pt"
MAX_T = int(os.environ.get("H1_MAX_T", "400"))
RESUME = os.environ.get("H1_RESUME", "0") == "1"


def seg_defs(lp):
    """返回 [(cls, s, e)]，连续相同非 blank 段。"""
    path = lp.argmax(1)
    out, cur, st = [], None, 0
    for t, x in enumerate(path):
        x = int(x)
        if x != cur:
            if cur is not None and cur != 0:
                out.append((cur, st, t))
            cur, st = x, t
    if cur is not None and cur != 0:
        out.append((cur, st, len(path)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels = P78.read_csv_split(a.split + ".csv")
    word2idx, n, _ = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")

    out_p = CACHE / ("p224_%s_segments.pkl" % a.split)
    part_p = CACHE / ("p224_%s_partial.pkl" % a.split)
    Z, Y, SID, SE, NT = [], [], [], [], {}
    seen, skipped = 0, 0
    if RESUME and part_p.exists():
        with open(part_p, "rb") as f:
            st = pickle.load(f)
        Z, Y, SID, SE, NT = st["Z"], st["Y"], st["SID"], st["SE"], st["NT"]
        seen, skipped = st["seen"], st.get("skipped", 0)
        print("RESUME seen=%d skipped=%d segments=%d" % (seen, skipped, len(Z)), flush=True)

    _, tf = P78.build_transforms(hflip=False, img_size=160, y_shift=0,
                                 temporal_rescale=0.0)
    ds = P78.RGBSeqDataset(a.split, labels, word2idx, tf, False)
    _lim = int(os.environ.get("H1_LIMIT", "0"))   # 冒烟用，正式跑不设
    if _lim > 0:
        ds.items = ds.items[:_lim]
    ds.items = ds.items[seen:]          # resume：跳过已处理（含被跳过的长视频）
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=P78.collate, num_workers=0, pin_memory=False)
    mdl = build_model(CKPT, 1024, n, dev)
    mdl.eval()
    cap = FeatureCapture().attach(mdl)
    t0 = datetime.now()

    with torch.no_grad():
        for bi, (vid, _tgt, _tl, dl, _tll, sids, _ids) in enumerate(loader):
            _lens = [int(x) for x in dl.view(-1).tolist()]
            if _lens and max(_lens) > MAX_T:
                skipped += len(sids)
                seen += len(sids)
                continue
            cap.reset()
            out = mdl(vid.to(dev), dl, False)
            if dev == "cuda":
                torch.cuda.synchronize()
            logits = out[0].float().cpu()
            feat = cap.feat.float().cpu()
            lp_all = F.log_softmax(logits, dim=-1).numpy()
            lens = dl.view(-1).tolist()
            for b, sid in enumerate(sids):
                T = min(P78._conv_len(int(lens[b])), lp_all.shape[0])
                lp = lp_all[:T, b, :].astype(np.float32)
                f = feat[:T, b, :].numpy().astype(np.float32)
                NT[str(sid)] = int(T)
                for cls, s0, e0 in seg_defs(lp):
                    z = f[s0:e0].mean(0)
                    nrm = float(np.linalg.norm(z))
                    if nrm < 1e-8:
                        continue
                    Z.append((z / nrm).astype(np.float32))
                    Y.append(int(cls))
                    SID.append(str(sid))
                    SE.append((int(s0), int(e0)))
                seen += 1
            if bi % 100 == 0 and dev == "cuda":
                torch.cuda.empty_cache()
            if bi % 400 == 0:
                CACHE.mkdir(parents=True, exist_ok=True)
                with open(part_p, "wb") as pf:
                    pickle.dump({"Z": Z, "Y": Y, "SID": SID, "SE": SE,
                                 "NT": NT, "seen": seen, "skipped": skipped}, pf)
                print("%s forward %d seen=%d skipped=%d segs=%d elapsed=%.0fs partial"
                      % (a.split, bi, seen, skipped, len(Z),
                         (datetime.now() - t0).total_seconds()), flush=True)
    cap.detach()
    del mdl
    if dev == "cuda":
        torch.cuda.empty_cache()

    data = {"split": a.split,
            "Z": np.asarray(Z, dtype=np.float32),
            "Y": np.asarray(Y, dtype=np.int32),
            "SID": SID, "SE": SE, "NT": NT,
            "seen": seen, "skipped": skipped,
            "note": "segment = consecutive identical non-blank CTC greedy path; "
                    "z = mean classifier11 input feature, L2 normalized"}
    with open(out_p, "wb") as f:
        pickle.dump(data, f)
    print("SAVED %s  segments=%d  Z=%s  seen=%d skipped=%d"
          % (out_p, len(Z), data["Z"].shape, seen, skipped), flush=True)
    (CACHE / ("p224_%s_meta.json" % a.split)).write_text(json.dumps({
        "split": a.split, "n_segments": len(Z), "seen": seen,
        "skipped_long": skipped, "max_frames": MAX_T,
        "z_shape": list(data["Z"].shape)}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
