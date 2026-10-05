# -*- coding: utf-8 -*-
"""P49 · 口径对照：我报的 WER vs 你在网页上感受到的东西

## 用户质疑
「对啊，你说的wer和我感觉的不一样」

这句话是对的，而且差异来自**三层叠加**，我此前只讲了一部分。

## 三层差异
1. **指标口径**：`folded` WER 把 738 个 ref `<unk>` 折叠成同一个符号，
   模型吐 unk 就算命中。剔除后真实 WER = 0.7919（不是 0.5211）。
2. **特征链路不同**（最关键）：0.5211 是在**离线预提取特征**上算的；
   你在网页上传视频走 **MediaPipe 实时提取**，是另一条链路。
3. **统计粒度不同**：WER 是 2838 个 token 的平均值；
   你看的是几条短视频的输出，而输出越短 unk 占比越高（P46 实测）。

## 本脚本把三层同时打印出来
同一批 dev 视频：
  A. 离线特征 + folded WER      <- 我一直报的
  B. 离线特征 + 剔 unk WER      <- 真实能力
  C. 实时特征 + folded WER      <- 你网页上那条链路
  D. 实时特征 + 剔 unk WER
  + 逐样本输出（你实际看到的）
"""
from __future__ import annotations

import collections
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "app" / "backend"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402
from cslr.recognition.training import decode_batch                    # noqa: E402
from p40_rgb_main import levenshtein, DualInputCTC, read_csv          # noqa: E402

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM = REPO / "artifacts/part3_features"
DEV_VID = Path("/mnt/c/Users/su127/Desktop/csl视频/dev")
UNK = "<unk>"


def _extract_one(arg):
    """子进程：提取一个视频的特征。"""
    vp, = arg
    try:
        import os
        os.environ.setdefault("LD_LIBRARY_PATH", "/home/su127/libs")
        sys.path.insert(0, str(REPO / "app" / "backend"))
        from realtime_landmark import RealtimeLandmarkExtractor
        # 每个进程建一次模型（建模型约 0.9s，摊到多帧上可忽略）
        ex = RealtimeLandmarkExtractor()
        return ex.extract_to_48x368(str(vp))
    except Exception as exc:                                        # noqa: BLE001
        print("  提取失败 %s: %s" % (vp.name, str(exc)[:60]), flush=True)
        return None


def find_video(sid: str) -> Path | None:
    if not DEV_VID.exists():
        return None
    for d in sorted(DEV_VID.iterdir()):
        p = d / (sid + ".mp4")
        if p.exists():
            return p
    return None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(
        REPO / "artifacts/metrics/blank-gov/p49-numbers-vs-feelings.json"))
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_dv.values() if False else
                                      read_csv(REPO / "data/raw/CE-CSL/"
                                               "label/train.csv").values(),
                                      min_frequency=2, max_tokens=300)

    # ---- 收集同批样本 ----
    sids, refs = [], []
    for sid in sorted(lab_dv):
        if len(sids) >= a.n:
            break
        p = LM / "validation" / (sid + ".landmark.npy")
        vp = find_video(sid)
        if not p.exists() or vp is None:
            continue
        ids = voc.encode(lab_dv[sid])
        if not ids or len(ids) > 24:
            continue
        sids.append(sid)
        refs.append(voc.decode(list(ids)))
    print("样本 = %d 条 dev 视频" % len(sids), flush=True)

    off = [np.load(LM / "validation" / (s + ".landmark.npy")).astype(np.float32)
           for s in sids]

    # ---- 并行提取实时特征 ----
    t0 = time.time()
    vps = [find_video(s) for s in sids]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        on = list(ex.map(_extract_one, [(v,) for v in vps]))
    ok = sum(1 for x in on if x is not None)
    el = time.time() - t0
    print("实时提取 %d/%d 条，耗时 %.1fs（%.2fs/条，%d 并行）"
          % (ok, len(sids), el, el / max(len(sids), 1), a.workers), flush=True)
    keep = [i for i, x in enumerate(on) if x is not None]
    off = [off[i] for i in keep]
    on = [on[i] for i in keep]
    refs = [refs[i] for i in keep]
    sids = [sids[i] for i in keep]

    # ---- 载入 P42 ----
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = DualInputCTC(lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                         vocab=int(voc.size), hidden=cfg["hidden"],
                         layers=cfg["layers"], dropout=cfg["dropout"],
                         use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                         mode=cfg.get("mode", "add")).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    def run(feats):
        hyps = []
        with torch.no_grad():
            for k in range(0, len(feats), 32):
                ch = feats[k:k + 32]
                x = torch.from_numpy(np.stack(ch).astype(np.float32)).to(device)
                il = torch.full((len(ch),), x.shape[1], dtype=torch.long,
                                device=device)
                lg = model(x, il, None, None)
                lp = torch.log_softmax(lg.float(), -1).cpu().numpy()
                ol = np.full((lg.size(0),), lg.size(1), dtype=np.int64)
                dec, _, _ = decode_batch(lp, ol, 1)
                hyps.extend(voc.decode(list(s)) for s in dec)
        return hyps

    def score(hyps, tag):
        E_f = sum(levenshtein(list(r), h) for r, h in zip(refs, hyps))
        N_f = sum(len(r) for r in refs)
        refs_no = [[t for t in r if t != UNK] or [UNK] for r in refs]
        E_n = sum(levenshtein(r, h) for r, h in zip(refs_no, hyps))
        N_n = sum(len(r) for r in refs_no)
        ex = sum(1 for r, h in zip(refs, hyps) if levenshtein(list(r), h) == 0)
        H = [t for h in hyps for t in h]
        n_unk = sum(1 for t in H if t == UNK)
        # 「几乎全 unk」的样本数（用户体验最差的那批）
        all_unk = sum(1 for h in hyps
                      if h and all(t == UNK for t in h))
        short = sum(1 for h in hyps if len(h) <= 2)
        return {
            "tag": tag,
            "wer_folded": round(E_f / max(N_f, 1), 4),
            "wer_excl_unk": round(E_n / max(N_n, 1), 4),
            "exact": "%d/%d (%.1f%%)" % (ex, len(refs),
                                         100 * ex / max(len(refs), 1)),
            "unk_ratio": round(n_unk / max(len(H), 1), 4),
            "hyp_len_mean": round(len(H) / max(len(hyps), 1), 2),
            "samples_all_unk": all_unk,
            "samples_len_le2": short,
            "hyps": hyps,
        }

    h_off = run(off)
    h_on = run(on)
    a_row = score(h_off, "A 离线特征 + folded WER（我以前报的）")
    b_row = score(h_off, "B 离线特征 + 剔 unk（真实能力）")
    c_row = score(h_on, "C 实时特征 + folded WER（你网页上的）")
    d_row = score(h_on, "D 实时特征 + 剔 unk（你网页上的真实水平）")

    print("\n" + "=" * 78)
    print("同一批 %d 条 dev 视频，四个格子" % len(refs))
    print("=" * 78)
    print("%-38s %8s %8s %10s %6s %6s"
          % ("", "folded", "剔unk", "exact", "unk%", "len"))
    for r in (a_row, b_row, c_row, d_row):
        print("%-38s %8.4f %8.4f %10s %5.1f%% %6.2f"
              % (r["tag"], r["wer_folded"], r["wer_excl_unk"], r["exact"],
                 100 * r["unk_ratio"], r["hyp_len_mean"]), flush=True)

    print("\n=== 你在网页上看到的（实时特征前 12 条）===")
    for i in range(min(12, len(refs))):
        h = h_on[i]
        u = sum(1 for t in h if t == UNK)
        real = [t for t in h if t != UNK]
        print("  %s" % sids[i])
        print("     真实标注 : %s" % "/".join(refs[i]))
        print("     网页输出 : %s%s"
              % ("/".join(h) if h else "(空)",
                 "   << 全是 unk" if h and u == len(h) else ""))

    n_all = c_row["samples_all_unk"]
    n_le2 = c_row["samples_len_le2"]
    print("\n=== 差异归因（实时特征 %d 条）===" % len(refs))
    print("  %-42s %5.1f%%" % ("输出里 unk 占 >=70% 的样本",
                               100 * sum(
                                   1 for h in h_on
                                   if h and sum(1 for t in h if t == UNK)
                                   / len(h) >= 0.7) / len(h_on)))
    print("  %-42s %5.1f%%" % ("输出全是 unk 的样本", 100 * n_all / len(h_on)))
    print("  %-42s %5.1f%%" % ("输出长度 <= 2 的样本（看着就是没认出来）",
                               100 * n_le2 / len(h_on)))
    print("  %-42s %.1f%%" % ("完全正确（exact）",
                              float(c_row["exact"].split("(")[1]
                                    .rstrip(")")) ))

    hyps_off = a_row.pop("hyps")
    hyps_on = c_row.pop("hyps")
    b_row.pop("hyps")
    d_row.pop("hyps")

    receipt = {
        "experiment": "P49", "date": "2026-10-05",
        "user_challenge": "「你说的wer和我感觉的不一样」",
        "why_they_differ": [
            "口径差异：我报的 folded WER 0.5211 折叠了 ref 的 OOV，"
            "模型吐 unk 即命中；剔除后真实水平低得多",
            "链路差异：0.5211 在离线预提取特征上算，"
            "你网页上传走 MediaPipe 实时提取（不同坐标系）",
            "粒度差异：WER 是 token 平均；你看的是短视频，"
            "而输出越短 unk 占比越高（P46 实测 >=70% unk 的占 13.2%）",
        ],
        "n_samples": len(refs),
        "four_cells": {"A_offline_folded": a_row, "B_offline_excl_unk": b_row,
                       "C_realtime_folded": c_row, "D_realtime_excl_unk": d_row},
        "extraction_seconds": round(el, 1),
        "workers": a.workers,
    }
    p = Path(a.out)
    p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
