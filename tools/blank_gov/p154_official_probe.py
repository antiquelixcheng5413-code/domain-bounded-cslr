#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P154：在官方权重上"略作修改"的可行性探测（零训练成本）

两件事
  A. blank bias 对官方模型还有没有用？
     官方 p_blank=0.872、生成率 94.9%（我们 0.875 / 77.2%）
     ⇒ 官方已经"敢输出"了，再压 blank 可能反而过度生成 ⇒ 先验预期是**无效或有害**
     但只需解码，零训练成本，值得实测确认
  B. 官方权重在我们的 train 子集上拟合到什么程度？
     若 train WER 已经很低 ⇒ 继续训练收益有限；
     若 train 上还有空间 + ckpt 只训到 epoch 38（官方 Train.py epochNum=55）
     ⇒ "继续训到 55 轮"就是最现实的改进手段

用法
  python tools/blank_gov/p154_official_probe.py [n_dev] [n_train]
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

CK = Path("/mnt/c/Users/su127/Desktop/中文手语识别/官方权重/TFNet-CE-CSL-CSLDaily-32.46.pth")
N_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 150
BIAS_LIST = (0.0, 0.25, 0.5, 0.75, 1.0)
SEED = 42


def log(*a):
    print("[P154 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def run_split(model, split, labels, idxs, word2idx, idx2word, dev):
    ds = P.RGBSeqDataset(split, labels, word2idx,
                         P.build_transforms(hflip=False, img_size=224)[1], False)
    if idxs is not None:
        ds.items = [ds.items[i] for i in idxs]
    ls = torch.nn.LogSoftmax(dim=-1)
    cache, lens, refs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            batch = P.collate([ds[i]])
            vid, _t, _tl, dl, _tl_len, sids, _ids = batch
            out = model(vid.to(dev), dl, False)
            lp = ls(out[0])[:, 0, :]
            L = min(P._conv_len(int(dl.view(-1)[0])), lp.shape[0])
            cache.append(lp[:L].to("cpu", torch.float32))
            lens.append(int(dl.view(-1)[0]))
            for b in sids:
                refs.append(labels[b][1])
    log("%-5s %d 条推理 %.0fs" % (split, len(ds), time.time() - t0))
    return cache, lens, refs


def decode(cache, lens, idx2word, bias):
    P.BLANK_BIAS = float(bias)
    hyps, ntok = [], 0
    for lp, L in zip(cache, lens):
        x = lp.unsqueeze(1).clone()
        if bias:
            x[..., 0] -= bias
        t = P.greedy_decode(x, torch.LongTensor([[L]]), idx2word)
        hyps += t
        ntok += sum(len(q) for q in t)
    P.BLANK_BIAS = 0.0
    return hyps, ntok


def main() -> int:
    dev = "cuda"
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    model = P.Net.moduleNet(1024, wordSetNum + 1, "TFNet",
                            torch.device(dev), "CE-CSL", True).to(dev)
    sd = torch.load(str(CK), map_location="cpu", weights_only=False,
                    mmap=True)["moduleNet_state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    P.RGB_FIX = True
    P.OFFICIAL_PAD = True

    # ---- A. dev 上的 bias 扫描 ----
    lab_dev = P.read_csv_split("dev.csv")
    c_dev, l_dev, refs_dev = run_split(model, "dev", lab_dev, None,
                                       word2idx, idx2word, dev)
    print()
    log("──── A. 官方权重 + blank bias（dev %d 条）────" % len(c_dev))
    log("  %-8s %10s %10s" % ("bias", "WER", "生成率"))
    rows = {}
    base = None
    for b in BIAS_LIST:
        hyps, ntok = decode(c_dev, l_dev, idx2word, b)
        o = evaluate(refs_dev, hyps)
        g = 100.0 * ntok / max(o["ref_tokens"], 1)
        rows["bias%.2f" % b] = {"WER": o["WER_official"], "gen_rate": g}
        if base is None:
            base = o["WER_official"]
        log("  %-8.2f %10.2f %9.1f%%   Δ %+6.2f"
            % (b, o["WER_official"], g, o["WER_official"] - base))
    best_b = min(rows, key=lambda k: rows[k]["WER"])
    d = rows[best_b]["WER"] - base
    log("  最佳 %s = %.2f（Δ %+.2f pp）" % (best_b, rows[best_b]["WER"], d))
    log("  ⇒ %s" % ("✅ 有效（>噪声 1.556）" if d <= -1.556
                    else "⚪ 无效/在噪声内（符合先验：官方生成率已 94.9%）"))

    # ---- B. train 子集：拟合程度 ----
    lab_tr = P.read_csv_split("train.csv")
    idx = random.Random(SEED).sample(range(len(lab_tr)), min(N_TRAIN, len(lab_tr)))
    c_tr, l_tr, refs_tr = run_split(model, "train", lab_tr, idx,
                                    word2idx, idx2word, dev)
    hyps_tr, ntok_tr = decode(c_tr, l_tr, idx2word, 0.0)
    o_tr = evaluate(refs_tr, hyps_tr)
    g_tr = 100.0 * ntok_tr / max(o_tr["ref_tokens"], 1)
    print()
    log("──── B. 官方权重在 train 子集（%d 条）────" % len(c_tr))
    log("  train WER %.2f%%（生成率 %.1f%%）  vs  dev %.2f%%"
        % (o_tr["WER_official"], g_tr, base))
    gap = base - o_tr["WER_official"]
    log("  dev − train = %+.2f pp" % gap)
    if o_tr["WER_official"] < 8:
        log("  ⇒ train 已拟合得很深（<8%%）⇒ 继续训练收益可能有限")
    else:
        log("  ⇒ train 上还有空间 ⇒ 继续训到 55 轮值得一试")
    log("  ⚠️ ckpt 只到 epoch %s（官方 Train.py epochNum=55）" % 38)

    out = REPO / "artifacts/metrics/blank-gov/p154-official-probe.json"
    out.write_text(json.dumps({
        "ckpt": CK.name, "module": "TFNet", "img_size": 224,
        "rgb_fix": True, "official_pad": True,
        "dev_bias_sweep": rows, "best_bias": best_b, "delta_best": d,
        "train_subset": {"n": len(c_tr), "WER": o_tr["WER_official"],
                         "gen_rate": g_tr},
        "dev_train_gap_pp": gap,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
