# -*- coding: utf-8 -*-
"""P44b · 验证服务端的归一化是否为 bug（口径不一致）

## 疑点

`app/backend/ctc_landmark_service.py:157` 的注释写「归一化：与训练一致」，
并对输入套了 `FeatureNormalizer`。

但 P40/P42 的训练与评估口径是 `to_batch()` 里的
    lm[i] = torch.from_numpy(s["lm"])
**完全不做归一化**。

=> 线上服务的输入分布与训练分布不同，模型收到的不是它学过的信号。

本脚本用**同一批真实 dev 视频的预提取特征**（绕过 MediaPipe，
排除提取器差异），只切换「归一化 / 不归一化」这一个变量，
量化这个 bug 到底损失多少。
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402
from cslr.recognition.dataset import FeatureNormalizer                # noqa: E402
from cslr.recognition.training import decode_batch                    # noqa: E402
from p40_rgb_main import levenshtein, DualInputCTC, read_csv           # noqa: E402

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM = REPO / "artifacts/part3_features"
UNK = "<unk>"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)

    items = []
    for sid in sorted(lab_dv):
        p = LM / "validation" / (sid + ".landmark.npy")
        if not p.exists():
            continue
        ids = voc.encode(lab_dv[sid])
        if not ids or len(ids) > 24:
            continue
        items.append({"sid": sid, "lm": np.load(p).astype(np.float32),
                      "ref": voc.decode(list(ids))})
    print("dev = %d" % len(items), flush=True)

    paths = sorted((LM / "train").glob("*.landmark.npy"))[:1500]
    nrm = FeatureNormalizer.fit([np.load(p) for p in paths])

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = DualInputCTC(lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                         vocab=int(voc.size), hidden=cfg["hidden"],
                         layers=cfg["layers"], dropout=cfg["dropout"],
                         use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                         mode=cfg.get("mode", "add")).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    out = {}
    for tag, use_nrm in (("train_path_NO_norm", False),
                         ("service_path_WITH_norm", True)):
        recs = []
        with torch.no_grad():
            for k in range(0, len(items), 32):
                ch = items[k:k + 32]
                arrs = [nrm.apply(s["lm"]) if use_nrm else s["lm"] for s in ch]
                lm = torch.from_numpy(np.stack(arrs).astype(np.float32)).to(device)
                il = torch.full((len(ch),), lm.shape[1], dtype=torch.long, device=device)
                logits = model(lm, il, None, None)
                lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
                ol = np.full((logits.size(0),), logits.size(1), dtype=np.int64)
                dec, _, _ = decode_batch(lp, ol, 1)
                for r, seq in enumerate(dec):
                    recs.append({"ref": ch[r]["ref"],
                                 "hyp": voc.decode(list(seq))})
        E = sum(levenshtein(list(r["ref"]), r["hyp"]) for r in recs)
        N = sum(len(r["ref"]) for r in recs)
        H = [t for r in recs for t in r["hyp"]]
        n_h = len(H)
        ex = sum(1 for r in recs if levenshtein(list(r["ref"]), r["hyp"]) == 0)
        out[tag] = {
            "wer": round(E / max(N, 1), 4),
            "unk_ratio_hyp": round(sum(1 for t in H if t == UNK) / max(n_h, 1), 4),
            "hyp_len_mean": round(n_h / len(recs), 3),
            "exact": ex,
            "distinct": len(set(H)),
            "n_samples": len(recs),
        }
        print("%-24s WER=%.4f  hyp<unk>=%.1f%%  len=%.2f  exact=%d  distinct=%d"
              % (tag, out[tag]["wer"], out[tag]["unk_ratio_hyp"] * 100,
                 out[tag]["hyp_len_mean"], ex, out[tag]["distinct"]), flush=True)

    a, b = out["train_path_NO_norm"], out["service_path_WITH_norm"]
    receipt = {
        "experiment": "P44b", "date": "2026-10-05",
        "finding": ("服务端口径与训练口径不一致：训练 to_batch() 不做归一化，"
                    "但 ctc_landmark_service.py 对输入套了 FeatureNormalizer"),
        "train_path_NO_norm": a,
        "service_path_WITH_norm": b,
        "delta_wer": round(b["wer"] - a["wer"], 4),
        "verdict": ("服务端归一化是 BUG，会显著抬高 <unk> 比例并恶化 WER"
                    if b["wer"] > a["wer"] else "归一化无害或有益"),
    }
    p = REPO / "artifacts/metrics/blank-gov/p44b-norm-mismatch.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
