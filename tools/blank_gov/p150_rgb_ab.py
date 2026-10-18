#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P150：通道顺序 A/B —— 现有 P123 ckpt 在 BGR / 真RGB 两种输入下的 dev WER

背景（P149 实测）
  我们落盘的 jpg 由「cv2 抽帧 -> cvtColor(BGR2RGB) -> imencode」写出，磁盘语义 R/B 颠倒；
  训练读图再 cvtColor(BGR2RGB) => 喂给模型的实际是 **BGR**（逐通道 MAE：R 15.9 / G 1.0 / B 16.3）。
  官方用 imageio.get_data()（RGB）后同样做一次 cvtColor(BGR2RGB) => 官方最终是 **RGB**。

本实验回答（不依赖 imageio）
  通道顺序对这个模型是不是"实质性"的？
  用 **BGR 训出来的** P123 ckpt，直接喂真 RGB：
    - 若 WER 明显变差 => 通道被模型学进去了 => 修正后必须重训，收益可期
    - 若几乎不变     => 通道在该任务上不敏感，修正意义有限
"""
import json
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
REF_BGR_BIAS0 = 55.84          # 已知答案：现状（BGR 输入，bias=0）


def log(*a):
    print("[P150 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def run(model, dev, word2idx, idx2word, labels, rgb_fix):
    P.RGB_FIX = bool(rgb_fix)
    _, tf = P.build_transforms(hflip=False, img_size=160)
    ds = P.RGBSeqDataset("dev", labels, word2idx, tf, False)
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
    log("  RGB_FIX=%s 推理 %d 条 %.0fs" % (rgb_fix, len(ds), time.time() - t0))
    res = {}
    for bias in (0.0, 0.75):
        P.BLANK_BIAS = float(bias)
        hyps, ntok = [], 0
        for lp, L in zip(cache, lens):
            x = lp.unsqueeze(1).clone()
            if bias:
                x[..., 0] -= bias
            t = P.greedy_decode(x, torch.LongTensor([[L]]), idx2word)
            hyps += t
            ntok += sum(len(q) for q in t)
        o = evaluate(refs, hyps)
        res["bias%.2f" % bias] = {
            "WER": o["WER_official"],
            "gen_rate": 100.0 * ntok / max(o["ref_tokens"], 1)}
    P.BLANK_BIAS = 0.0
    return res


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
    labels = P.read_csv_split("dev.csv")

    rows = {}
    for fix in (False, True):
        rows["bgr" if not fix else "rgb"] = run(
            model, dev, word2idx, idx2word, labels, fix)

    got = rows["bgr"]["bias0.00"]["WER"]
    ok = abs(got - REF_BGR_BIAS0) < 0.05
    log("已知答案校验（BGR 输入 bias0 = %.2f）：实测 %.2f ⇒ %s"
        % (REF_BGR_BIAS0, got, "✅" if ok else "❌"))

    print()
    log("──── 通道顺序 A/B（dev 515，P123 best，同一 ckpt）────")
    log("  %-10s %10s %10s" % ("输入", "bias0", "bias0.75"))
    for k in ("bgr", "rgb"):
        log("  %-10s %10.2f %10.2f" % (k, rows[k]["bias0.00"]["WER"],
                                       rows[k]["bias0.75"]["WER"]))
    d = rows["rgb"]["bias0.00"]["WER"] - rows["bgr"]["bias0.00"]["WER"]
    log("  Δ(真RGB − BGR) = %+.2f pp" % d)
    if d > 1.556:
        log("  ⇒ ✅ 通道是实质性的：同一 ckpt 换输入顺序就掉 %.2f pp" % d)
        log("     ⇒ 模型学到的是 BGR 语义；用真 RGB 重训才有意义")
    elif d < -1.556:
        log("  ⇒ ⚠️ 真 RGB 输入反而更好 ⇒ 修正可能无需重训即可受益")
    else:
        log("  ⇒ ⚪ 差异在噪声内：该模型对通道不敏感")

    out = REPO / "artifacts/metrics/blank-gov/p150-rgb-ab.json"
    out.write_text(json.dumps({
        "ckpt": CK.name, "n": len(labels), "rows": rows,
        "delta_rgb_minus_bgr": d, "sanity_bgr_ok": bool(ok),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
