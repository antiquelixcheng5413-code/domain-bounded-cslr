# -*- coding: utf-8 -*-
"""P48b · 用训练好的 P42 checkpoint，分别喂「离线特征」和「实时特征」

## 用户质疑
「真的过拟合了吗，我在网页上测 train 里的数据，效果也不好」

## P48 已确认
全新模型对两种特征都能过拟合到 WER 0.0000
=> 架构有记忆能力，实时特征本身也不是不可学。

## 但这留下一个矛盾
线上跑的是 **P42 checkpoint，它是用【离线预提取特征】训练的**。
而网页上传视频走 **MediaPipe 实时提取**。
P48 实测同一批视频：deltas 段 std 比 **3.12 倍**。
=> 这是**域偏移（domain shift）**：模型在分布 A 上训练，却被喂分布 B。

## 本脚本做决定性对比（同一个 checkpoint，唯一变量=特征来源）
取 60 条 train 视频（模型见过的），分别用：
  A. 离线预提取特征  -> 应接近 train WER 0.0065（记忆效应）
  B. MediaPipe 实时特征 -> 若明显变差，就是域偏移

并同时给出 dev 上两种特征的 WER，判断「线上/线下」差距有多大。

## 附带：查 receipt 看离线特征当初是怎么提的
`artifacts/part3_features/train/*.receipt.json` 里可能有提取器版本，
如果线上和当初用的不是同一个提取器，那就是根因。
"""
from __future__ import annotations

import collections
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
from cslr.recognition.training import decode_batch                    # noqa: E402
from p40_rgb_main import levenshtein, DualInputCTC, read_csv          # noqa: E402

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM = REPO / "artifacts/part3_features"
VID = Path("/mnt/c/Users/su127/Desktop/csl视频/train")
DEV_VID = Path("/mnt/c/Users/su127/Desktop/csl视频/dev")
UNK = "<unk>"


def find_video(root: Path, sid: str) -> Path | None:
    if not root.exists():
        return None
    for d in sorted(root.iterdir()):
        p = d / (sid + ".mp4")
        if p.exists():
            return p
    return None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-dev", type=int, default=60)
    ap.add_argument("--out", default=str(
        REPO / "artifacts/metrics/blank-gov/p48b-domain-shift.json"))
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)

    # ---- receipt：离线特征当初是怎么提的 ----
    rcp = sorted((LM / "train").glob("*.receipt.json"))
    receipt_sample = None
    if rcp:
        receipt_sample = json.loads(rcp[0].read_text(encoding="utf-8"))
    print("=== 离线特征的 receipt（提取器信息）===")
    print(json.dumps(receipt_sample, ensure_ascii=False, indent=1)[:900]
          if receipt_sample else "(无 receipt)", flush=True)

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = DualInputCTC(lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                         vocab=int(voc.size), hidden=cfg["hidden"],
                         layers=cfg["layers"], dropout=cfg["dropout"],
                         use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                         mode=cfg.get("mode", "add")).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    print("\ncheckpoint epoch = %s" % blob.get("epoch"), flush=True)

    from realtime_landmark import RealtimeLandmarkExtractor
    ex = RealtimeLandmarkExtractor()

    def run(feats, refs):
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
        E = sum(levenshtein(list(r), h) for r, h in zip(refs, hyps))
        N = sum(len(r) for r in refs)
        ex_n = sum(1 for r, h in zip(refs, hyps)
                   if levenshtein(list(r), h) == 0)
        H = [t for h in hyps for t in h]
        return {"wer": round(E / max(N, 1), 4), "exact": ex_n,
                "n": len(refs),
                "exact_ratio": round(ex_n / max(len(refs), 1), 4),
                "unk_ratio": round(sum(1 for t in H if t == UNK)
                                   / max(len(H), 1), 4),
                "len_mean": round(len(H) / max(len(hyps), 1), 3),
                "hyps": hyps}

    out = {}

    # ================= TRAIN: 离线 vs 实时 =================
    for split_name, labs, vroot, feat_split, n_max in (
            ("train", lab_tr, VID, "train", a.n_train),
            ("dev", lab_dv, DEV_VID, "validation", a.n_dev)):
        off, on, refs = [], [], []
        for sid in sorted(labs):
            if len(refs) >= n_max:
                break
            p = LM / feat_split / (sid + ".landmark.npy")
            vp = find_video(vroot, sid)
            if not p.exists() or vp is None:
                continue
            ids = voc.encode(labs[sid])
            if not ids or len(ids) > 24:
                continue
            off.append(np.load(p).astype(np.float32))
            on.append(ex.extract_to_48x368(str(vp)))
            refs.append(voc.decode(list(ids)))
        print("\n=== %s：%d 条，离线 vs 实时 ===" % (split_name, len(refs)),
              flush=True)
        r_off = run(off, refs)
        r_on = run(on, refs)
        print("  离线特征: WER=%.4f  exact=%d/%d (%.1f%%)  unk=%.1f%%  len=%.2f"
              % (r_off["wer"], r_off["exact"], r_off["n"],
                 100 * r_off["exact_ratio"], 100 * r_off["unk_ratio"],
                 r_off["len_mean"]), flush=True)
        print("  实时特征: WER=%.4f  exact=%d/%d (%.1f%%)  unk=%.1f%%  len=%.2f"
              % (r_on["wer"], r_on["exact"], r_on["n"],
                 100 * r_on["exact_ratio"], 100 * r_on["unk_ratio"],
                 r_on["len_mean"]), flush=True)
        print("  => 域偏移代价: WER %+.4f，exact %+d"
              % (r_on["wer"] - r_off["wer"], r_on["exact"] - r_off["exact"]))
        r_off.pop("hyps")
        r_on.pop("hyps")
        out[split_name] = {"offline": r_off, "realtime": r_on,
                           "domain_shift_wer": round(r_on["wer"] - r_off["wer"], 4),
                           "domain_shift_exact": r_on["exact"] - r_off["exact"]}

    # ================= 结论 =================
    tr = out["train"]
    print("\n=== 结论 ===")
    if tr["realtime"]["wer"] - tr["offline"]["wer"] > 0.05:
        print("  ✅ 域偏移确认：**模型在 train 离线特征上表现好，"
              "换实时特征明显变差**")
        print("     这就是「网页上传 train 视频效果也不好」的原因。")
    else:
        print("  ❌ 域偏移不是主因：train 上两种特征差别不大")
    print("  训练集记忆程度: 离线 exact=%d/%d -> 实时 exact=%d/%d"
          % (tr["offline"]["exact"], tr["offline"]["n"],
             tr["realtime"]["exact"], tr["realtime"]["n"]))

    receipt = {
        "experiment": "P48b", "date": "2026-10-05",
        "user_challenge": "「我在网页上测 train 里的数据，效果也不好」",
        "p48_result": "全新模型对离线/实时两种特征都能过拟合到 WER 0.0000",
        "offline_extractor_receipt": receipt_sample,
        "train": out["train"], "dev": out["dev"],
        "diagnosis": (
            "P42 是用【离线预提取特征】训练的，网页走【MediaPipe 实时提取】。"
            "同一批 train 视频上，两种特征的 WER 差值即域偏移代价。"
            if tr["realtime"]["wer"] - tr["offline"]["wer"] > 0.05 else
            "域偏移不是主因"),
    }
    p = Path(a.out)
    p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
