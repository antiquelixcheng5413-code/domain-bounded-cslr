# -*- coding: utf-8 -*-
"""P44c · 线上(MediaPipe 实时)特征 vs 离线(预提取)特征的分布差异

## 为什么要做

P44b 修掉归一化 bug 后，线上输出**反而变成几乎全是 `<unk>`**。
说明除了归一化，还有第二个原因。

最可能的嫌疑：**线上特征与训练特征不是同一种东西**
- 训练/评估用的是 `artifacts/part3_features/<split>/*.landmark.npy`（离线批量提取）
- 线上用的是 `realtime_landmark.py`（MediaPipe tasks 实时提取）

如果两者分布不同，模型收到的就是没见过的信号 -> 只会吐 unk。

本脚本对**同一个视频**做 A/B：
  A. 离线预提取特征（训练同源）
  B. 线上实时提取特征
逐维统计差异，并各跑一次模型看输出。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "app" / "backend"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402
from cslr.recognition.training import decode_batch                    # noqa: E402
from p40_rgb_main import levenshtein, DualInputCTC, read_csv           # noqa: E402

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM = REPO / "artifacts/part3_features"
DEV_VID = Path("/mnt/c/Users/su127/Desktop/csl视频/dev/A")
UNK = "<unk>"
BLOCKS = {
    "hands[0:126]": (0, 126), "pose[126:158]": (126, 158),
    "face[158:182]": (158, 182), "presence[182:186]": (182, 186),
    "deltas[186:368]": (186, 368),
}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=str(
        REPO / "artifacts/metrics/blank-gov/p44c-online-vs-offline.json"))
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = DualInputCTC(lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                         vocab=int(voc.size), hidden=cfg["hidden"],
                         layers=cfg["layers"], dropout=cfg["dropout"],
                         use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                         mode=cfg.get("mode", "add")).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    def run(feats, refs):
        outs = []
        with torch.no_grad():
            for k in range(0, len(feats), 16):
                ch = feats[k:k + 16]
                x = torch.from_numpy(np.stack(ch).astype(np.float32)).to(device)
                il = torch.full((len(ch),), x.shape[1], dtype=torch.long,
                                device=device)
                lg = model(x, il, None, None)
                lp = torch.log_softmax(lg.float(), dim=-1).cpu().numpy()
                ol = np.full((lg.size(0),), lg.size(1), dtype=np.int64)
                dec, _, _ = decode_batch(lp, ol, 1)
                outs.extend(voc.decode(list(s)) for s in dec)
        E = sum(levenshtein(list(refs[i]), outs[i]) for i in range(len(outs)))
        N = sum(len(r) for r in refs)
        H = [t for o in outs for t in o]
        return {"wer": round(E / max(N, 1), 4),
                "unk_ratio": round(sum(1 for t in H if t == UNK) / max(len(H), 1), 4),
                "len_mean": round(len(H) / max(len(outs), 1), 3),
                "distinct": len(set(H)),
                "exact": sum(1 for i in range(len(outs))
                             if levenshtein(list(refs[i]), outs[i]) == 0),
                "n": len(outs)}, outs

    # ---- 收集 A/B 特征 ----
    from realtime_landmark import RealtimeLandmarkExtractor
    ex = RealtimeLandmarkExtractor()

    sids = sorted(s for s in lab_dv
                  if (LM / "validation" / (s + ".landmark.npy")).exists()
                  and (DEV_VID / (s + ".mp4")).exists())[:a.n]
    print("对比样本 %d 个：%s" % (len(sids), sids), flush=True)

    off, on, refs = [], [], []
    for s in sids:
        off.append(np.load(LM / "validation" / (s + ".landmark.npy")).astype(np.float32))
        ids = voc.encode(lab_dv[s])
        refs.append(voc.decode(list(ids)))
        vp = DEV_VID / (s + ".mp4")
        if vp.exists():
            on.append(ex.extract_to_48x368(str(vp)))
    print("offline=%d online=%d" % (len(off), len(on)), flush=True)

    # ---- 逐维分布对比（只对比有交集的部分）----
    m = min(len(off), len(on))
    A = np.stack(off[:m])       # (m,48,368) 训练同源
    Bm = np.stack(on[:m])       # (m,48,368) 线上实时
    print("\n=== 逐块分布对比（%d 个视频）===" % m)
    print("%-18s %10s %10s %10s %10s %8s" %
          ("block", "off_mean", "on_mean", "off_std", "on_std", "ratio"))
    blocks = {}
    for name, (s0, s1) in BLOCKS.items():
        am, bm = A[:, :, s0:s1].mean(), Bm[:, :, s0:s1].mean()
        asd, bsd = A[:, :, s0:s1].std(), Bm[:, :, s0:s1].std()
        blocks[name] = {"off_mean": round(float(am), 5),
                        "on_mean": round(float(bm), 5),
                        "off_std": round(float(asd), 5),
                        "on_std": round(float(bsd), 5),
                        "std_ratio": round(float(bsd / (asd + 1e-9)), 3)}
        print("%-18s %10.4f %10.4f %10.4f %10.4f %8.2f" %
              (name, am, bm, asd, bsd, bsd / (asd + 1e-9)))

    # presence 位对比（0/1 掩码，最直观的语义差异）
    pa, pb = A[:, :, 182:186], Bm[:, :, 182:186]
    print("\n=== presence 各位检出率（离线 vs 线上）===")
    names = ["pose", "handL", "handR", "face"]
    pres = {}
    for k in range(4):
        ra, rb = float(pa[:, :, k].mean()), float(pb[:, :, k].mean())
        pres[names[k]] = {"offline": round(ra, 4), "online": round(rb, 4)}
        print("  %-6s offline=%.3f  online=%.3f  %s"
              % (names[k], ra, rb, "<<< 差异大" if abs(ra - rb) > 0.15 else ""))

    # ---- A/B 跑模型 ----
    r_off, o_off = run(off[:m], refs[:m])
    r_on, o_on = run(on[:m], refs[:m])
    print("\n=== 同一批视频，两种特征 ===")
    print("离线预提取(训练同源): WER=%.4f unk=%.1f%% len=%.2f distinct=%d exact=%d/%d"
          % (r_off["wer"], r_off["unk_ratio"] * 100, r_off["len_mean"],
             r_off["distinct"], r_off["exact"], r_off["n"]))
    print("线上实时提取       : WER=%.4f unk=%.1f%% len=%.2f distinct=%d exact=%d/%d"
          % (r_on["wer"], r_on["unk_ratio"] * 100, r_on["len_mean"],
             r_on["distinct"], r_on["exact"], r_on["n"]))

    print("\n=== 逐样本输出对照 ===")
    for i, s in enumerate(sids[:m]):
        print("  %s" % s)
        print("     ref : %s" % "/".join(refs[i]))
        print("     off : %s" % "/".join(o_off[i]))
        print("     on  : %s" % ("/".join(o_on[i]) if i < len(o_on) else "(无)"))

    receipt = {
        "experiment": "P44c", "date": "2026-10-05",
        "question": "修掉归一化 bug 后线上输出反而几乎全是 <unk>",
        "n_videos": m, "sids": sids[:m],
        "block_stats": blocks,
        "presence": pres,
        "offline_feature": r_off,
        "online_feature": r_on,
        "delta_wer": round(r_on["wer"] - r_off["wer"], 4),
        "per_sample": [{"sid": sids[i], "ref": refs[i],
                        "offline": o_off[i],
                        "online": o_on[i] if i < len(o_on) else None}
                       for i in range(m)],
    }
    p = Path(a.out)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
