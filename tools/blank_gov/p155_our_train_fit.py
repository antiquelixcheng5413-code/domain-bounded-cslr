#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P155：我们的 P123 在 train 子集上的拟合程度（对照官方的 0.36%）

动机（P154）
  官方权重 train WER **0.36%** / dev 33.92% ⇒ 训练集几乎背下来了（过拟合但判别力强）
  我们 P123：train CTC loss ~5.5 / dev 55.84% —— **但从没测过 train WER**

  这个对照直接决定方向：
    我们 train WER ≈ 0.x%  ⇒ 同样过拟合，差距纯粹在判别力
    我们 train WER 很高     ⇒ **欠拟合** ⇒ 加训练/加容量是明确方向

用法
  python tools/blank_gov/p155_our_train_fit.py [n_train]
"""
import json
import random
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import p78_train_official_tfnet as P  # noqa: E402
from official_wer import evaluate  # noqa: E402

CK = REPO / "artifacts/checkpoints/p123-official55bf16-best.pt"
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SEED = 42
OFFICIAL = {"train_wer": 0.36, "dev_wer": 33.92, "peak_sharp": 0.8641,
            "p_blank": 0.8722}


def log(*a):
    print("[P155 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def main() -> int:
    dev = "cuda"
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    P.OFFICIAL_PAD = bool(ck.get("config", {}).get("official_pad", False))
    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    # 🔴 P123 是 BGR 训练的 ⇒ 这里用**它自己的训练分布**（RGB_FIX=False）才公平
    P.RGB_FIX = False
    log("载入 %s（epoch %s）；用其训练时的输入口径 RGB_FIX=False" % (CK.name, ck.get("epoch")))

    lab_tr = P.read_csv_split("train.csv")
    idx = random.Random(SEED).sample(range(len(lab_tr)), min(N_TRAIN, len(lab_tr)))
    _, tf = P.build_transforms(hflip=False, img_size=160)
    ds = P.RGBSeqDataset("train", lab_tr, word2idx, tf, False)
    # 🔴 不替换 ds.items：sids 必须保持"全量索引"，否则 refs 与 hyp 会错位
    log("ds 全量 %d 条，取子集 %d 条（idx 前 5 个：%s）" % (len(ds), len(idx), idx[:5]))

    ls = torch.nn.LogSoftmax(dim=-1)
    refs, hyps, ntok = [], [], 0
    ps, pb = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in idx:
            batch = P.collate([ds[i]])
            vid, _t, _tl, dl, _tl_len, sids, _ids = batch
            out = model(vid.to(dev), dl, False)
            lp = ls(out[0])[:, 0, :]
            L = min(P._conv_len(int(dl.view(-1)[0])), lp.shape[0])
            lp = lp[:L]
            p = lp.exp()
            pb.append(p[:, 0].mean().item())
            srt = p[:, 1:].sort(-1, descending=True).values
            top1 = srt[:, 0]
            arg = lp.argmax(-1).tolist()
            pm = top1.tolist()
            segs, cur, prev = [], None, -1
            for a, v in zip(arg, pm):
                if a != prev:
                    if cur is not None:
                        segs.append(cur)
                    cur = [a, v]
                else:
                    cur[1] = max(cur[1], v)
                prev = a
            if cur is not None:
                segs.append(cur)
            nb = [s[1] for s in segs if s[0] != 0]
            ps.append((sum(nb) / len(nb)) if nb else 0.0)
            P.BLANK_BIAS = 0.0
            t = P.greedy_decode(lp.to("cpu", torch.float32).unsqueeze(1),
                              torch.LongTensor([[L]]), idx2word)
            hyps += t
            ntok += sum(len(q) for q in t)
            for b in sids:
                refs.append(lab_tr[b][1])
    log("推理 %d 条 %.0fs" % (len(idx), time.time() - t0))

    o = evaluate(refs, hyps)
    g = 100.0 * ntok / max(o["ref_tokens"], 1)
    print()
    log("──── 我们 P123 在 train 子集（%d 条，seed %d）────" % (len(idx), SEED))
    log("  %-14s %12s %12s" % ("指标", "我们P123", "官方TFNet"))
    log("  %-14s %12.2f %12.2f   ← 关键" % ("train WER", o["WER_official"],
                                            OFFICIAL["train_wer"]))
    log("  %-14s %12.1f%% %11.1f%%" % ("生成率", g, 100.2))
    log("  %-14s %12.4f %12.4f" % ("峰锐度", sum(ps) / len(ps), OFFICIAL["peak_sharp"]))
    log("  %-14s %12.4f %12.4f" % ("p_blank", sum(pb) / len(pb), OFFICIAL["p_blank"]))
    log("  %-14s %12.2f %12.2f" % ("dev WER", 55.84, OFFICIAL["dev_wer"]))

    tw = o["WER_official"]
    print()
    if tw < 5:
        log("  ⇒ ✅ 我们也是**过拟合**（train 已背下来）⇒ 差距纯粹是判别力，加训练无用")
    elif tw < 20:
        log("  ⇒ ⚪ 中等：train 未完全背下，可能还有一点训练空间")
    else:
        log("  ⇒ 🔴 我们是**欠拟合**（train %.2f%% vs 官方 0.36%%）" % tw)
        log("     ⇒ 加大训练/容量/修正输入是明确方向，与官方的处境完全不同")

    out = REPO / "artifacts/metrics/blank-gov/p155-our-train-fit.json"
    out.write_text(json.dumps({
        "ckpt": CK.name, "n_train": len(idx), "seed": SEED,
        "rgb_fix": False, "img_size": 160,
        "ours": {"train_wer": o["WER_official"], "gen_rate": g,
                 "peak_sharp": sum(ps) / len(ps), "p_blank": sum(pb) / len(pb),
                 "dev_wer": 55.84},
        "official_tfnet": OFFICIAL,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
