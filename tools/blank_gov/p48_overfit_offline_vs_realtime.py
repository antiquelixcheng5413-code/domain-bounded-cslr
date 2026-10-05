# -*- coding: utf-8 -*-
"""P48 · 同一批 train 视频：离线特征 vs 实时特征，各做过拟合

## 用户质疑
「真的过拟合了吗，我在网页上测 train 里的数据，效果也不好」

## 为什么这个质疑很可能是对的
P47c 的 8 条过拟合用的是 `artifacts/part3_features/train/*.npy`
= **离线批量预提取**特征。
而用户在网页上传视频走的是 `realtime_landmark.py` = **MediaPipe 实时提取**。
P44c 已实测两者分布不同：
  presence 离线 [0.98,0.94,1.00,1.00] vs 线上 [1.00,0.44,0.54,1.00]
  deltas 段 std 比 3.17 倍
=> **「离线特征能过拟合」完全不能推出「线上能过拟合」。**
我犯的错：把离线结论当成了线上结论。

## 本脚本做严格对照
同一批 train 视频（同一批 sample_id，同一批视频文件）：
  A. 离线预提取特征（.npy）      -> 过拟合到多少 WER
  B. MediaPipe 实时提取特征        -> 过拟合到多少 WER
  C. P42 训练好的模型，对两者各跑一遍 dev -> 泛化表现

如果 A 能到 0 而 B 不能，说明问题在**提取器**而不是模型。
如果 A、B 都能到 0，说明模型没问题，网页效果差另有原因。
"""
from __future__ import annotations

import collections
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "app" / "backend"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402
from cslr.recognition.model import BLANK_INDEX                        # noqa: E402
from cslr.recognition.training import ctc_loss, decode_batch           # noqa: E402
from p40_rgb_main import levenshtein, DualInputCTC, read_csv          # noqa: E402

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM = REPO / "artifacts/part3_features"
VID = Path("/mnt/c/Users/su127/Desktop/csl视频/train")
UNK = "<unk>"
N_VID = 8
STEPS = 600


def find_video(sid: str) -> Path | None:
    for d in sorted(VID.iterdir()) if VID.exists() else []:
        p = d / (sid + ".mp4")
        if p.exists():
            return p
    return None


def overfit(feats, tgts, refs, voc, tag, device, steps=STEPS):
    """在一组特征上过拟合，返回最终 train WER。

    🔴 索引空间铁律：`voc.decode()` 接受**词表索引**（0-based），
    而 `tgts` 里存的是 **CTC 类索引 = 词表索引 + 1**（类 0 是 blank）。
    参考序列必须用词表索引还原，否则整体错位一格，
    会出现「loss 降到 0.001 但 WER=1.0」这种自相矛盾的结果（P48 首版就踩了）。
    """
    torch.manual_seed(0)
    x = torch.from_numpy(np.stack(feats)).to(device)
    T = max(len(t) for t in tgts)
    y = torch.zeros(len(tgts), T, dtype=torch.long, device=device)
    tl = torch.zeros(len(tgts), dtype=torch.long, device=device)
    for i, t in enumerate(tgts):
        y[i, :len(t)] = torch.tensor(t, device=device)
        tl[i] = len(t)
    il = torch.full((len(tgts),), x.shape[1], dtype=torch.long, device=device)

    model = DualInputCTC(lm_dim=x.shape[2], rgb_dim=512, vocab=int(voc.size),
                         hidden=256, layers=2, dropout=0.3,
                         use_rgb=False, use_lm=True, mode="add").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, il, None, None)
        loss = ctc_loss(logits, il, y, tl, il)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                lp = torch.log_softmax(
                    model(x, il, None, None).float(), -1).cpu().numpy()
                dec, _, _ = decode_batch(
                    lp, np.full((len(tgts),), lp.shape[1], dtype=np.int64), 1)
            E = sum(levenshtein(refs[i], voc.decode(list(dec[i])))
                    for i in range(len(tgts)))
            N = sum(len(r) for r in refs)
            print("  [%s] step %3d loss=%.4f trainWER=%.4f"
                  % (tag, step, float(loss.detach()), E / max(N, 1)),
                  flush=True)
    model.eval()
    with torch.no_grad():
        lp = torch.log_softmax(model(x, il, None, None).float(),
                              -1).cpu().numpy()
        dec, _, _ = decode_batch(
            lp, np.full((len(tgts),), lp.shape[1], dtype=np.int64), 1)
    hyps = [voc.decode(list(s)) for s in dec]
    E = sum(levenshtein(refs[i], hyps[i]) for i in range(len(tgts)))
    N = sum(len(r) for r in refs)
    final = E / max(N, 1)
    print("  [%s] FINAL trainWER=%.4f  ndist=%d" % (tag, final,
                                                    len({t for h in hyps for t in h})),
          flush=True)
    del model
    torch.cuda.empty_cache()
    return final, hyps


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_VID)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default=str(
        REPO / "artifacts/metrics/blank-gov/p48-overfit-offline-vs-realtime.json"))
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)

    from realtime_landmark import RealtimeLandmarkExtractor

    # ---- 收集同批样本 ----
    off, on, tg, refs, sids = [], [], [], [], []
    ex = RealtimeLandmarkExtractor()
    for sid in sorted(lab_tr):
        if len(sids) >= a.n:
            break
        p = LM / "train" / (sid + ".landmark.npy")
        vp = find_video(sid)
        if not p.exists() or vp is None:
            continue
        ids = voc.encode(lab_tr[sid])
        if not ids or len(ids) > 24:
            continue
        off.append(np.load(p).astype(np.float32))
        t0 = time.time()
        on.append(ex.extract_to_48x368(str(vp)))
        tg.append([i + 1 for i in ids])       # CTC 类空间（blank=0）
        refs.append(voc.decode(list(ids)))    # 参考用词表空间
        sids.append(sid)
        print("  %s  实时提取 %.1fs  ref=%s"
              % (sid, time.time() - t0, "/".join(refs[-1])), flush=True)
    print("\n样本 = %d：%s\n" % (len(sids), sids), flush=True)

    # ---- 特征分布对照 ----
    O = np.stack(off)
    Nn = np.stack(on)
    print("=== 同一批视频，两种特征的分布 ===")
    print("%-18s %12s %12s %8s" % ("block", "offline_mean", "online_mean",
                                   "std比"))
    blocks = {"hands[0:126]": (0, 126), "pose[126:158]": (126, 158),
              "face[158:182]": (158, 182),
              "presence[182:186]": (182, 186),
              "deltas[186:368]": (186, 368)}
    dist = {}
    for nm, (s0, s1) in blocks.items():
        om, nm_ = O[:, :, s0:s1].mean(), Nn[:, :, s0:s1].mean()
        osd, nsd = O[:, :, s0:s1].std(), Nn[:, :, s0:s1].std()
        dist[nm] = {"off_mean": round(float(om), 4),
                    "on_mean": round(float(nm_), 4),
                    "std_ratio": round(float(nsd / (osd + 1e-9)), 3)}
        print("%-18s %12.4f %12.4f %8.2f" % (nm, om, nm_, nsd / (osd + 1e-9)))
    print("\npresence 各位检出率：")
    pres = {}
    for k, nm in enumerate(["pose", "handL", "handR", "face"]):
        ro, rn = float(O[:, :, 182 + k].mean()), float(Nn[:, :, 182 + k].mean())
        pres[nm] = {"offline": round(ro, 4), "online": round(rn, 4)}
        print("  %-6s offline=%.3f online=%.3f%s"
              % (nm, ro, rn, "  <<< 差异大" if abs(ro - rn) > 0.15 else ""))

    # ---- 对照过拟合 ----
    print("\n=== A. 离线预提取特征 过拟合 ===", flush=True)
    w_off, h_off = overfit(off, tg, refs, voc, "离线", device, a.steps)
    print("\n=== B. MediaPipe 实时特征 过拟合 ===", flush=True)
    w_on, h_on = overfit(on, tg, refs, voc, "实时", device, a.steps)

    print("\n=== 结论 ===")
    print("  离线特征过拟合 WER = %.4f" % w_off)
    print("  实时特征过拟合 WER = %.4f" % w_on)
    print("  => 用户判断%s"
          % ("正确！实时特征无法过拟合，问题在提取器不在模型"
             if w_on > 0.15 else
             "存疑：两种特征都能过拟合，网页效果差另有原因"))

    print("\n=== 逐样本输出 ===")
    for i, sid in enumerate(sids):
        print("  %s" % sid)
        print("     ref  : %s" % "/".join(refs[i]))
        print("     离线 : %s" % ("/".join(h_off[i]) or "(空)"))
        print("     实时 : %s" % ("/".join(h_on[i]) or "(空)"))

    receipt = {
        "experiment": "P48", "date": "2026-10-05",
        "user_challenge": "「真的过拟合了吗，我在网页上测 train 里的数据，效果也不好」",
        "why_the_challenge_is_valid":
            "P47c 的过拟合用的是离线预提取 .npy，而网页走 MediaPipe 实时提取；"
            "P44c 已实测两者分布不同。把离线结论当成线上结论是我的错误",
        "n_samples": len(sids), "sids": sids, "steps": a.steps,
        "feature_distribution": dist, "presence": pres,
        "overfit_offline_wer": round(w_off, 4),
        "overfit_realtime_wer": round(w_on, 4),
        "verdict": ("实时特征无法过拟合 -> 瓶颈在提取器"
                    if w_on > 0.15 else
                    "两种特征都能过拟合 -> 瓶颈在别处"),
        "per_sample": [{"sid": sids[i], "ref": refs[i],
                        "offline": h_off[i], "realtime": h_on[i]}
                       for i in range(len(sids))],
    }
    p = Path(a.out)
    p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
