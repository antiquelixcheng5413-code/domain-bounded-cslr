#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P153：官方 TFNet 权重的**后验诊断**，与我们的 P123 逐项对比

P152 已证：官方权重在我们的帧上跑出语料级 33.92（官方自报 33.19）
⇒ 数据侧等价，差距全在训练/模型侧。
⇒ 那么"差在哪一层"？用 P146 那套帧级后验指标直接对比：

  p_blank      帧级 blank 概率均值（我们 0.875，官方？）
  peak_sharp   greedy 折叠后每个输出 token 对应峰内的最大非blank 概率（我们 0.564）
  生成率        我们 77.2%（P152 实测官方 ~95%）
  sub/del/ins  误差构成

用法
  python tools/blank_gov/p153_official_posterior.py [n_dev]
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

CK = Path("/mnt/c/Users/su127/Desktop/中文手语识别/官方权重/TFNet-CE-CSL-CSLDaily-32.46.pth")
# 我们 P123 的对照值（P146 实测，dev 515，greedy）
OURS = {"p_blank": 0.8750, "peak_sharp": 0.564, "gen_rate": 77.2,
        "WER": 55.84, "sub": 54.4, "del": 43.2, "ins": 2.4}
N_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def log(*a):
    print("[P153 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def frame_stats(lp):
    """同 P146：p_blank / 熵 / margin / 峰锐度 / frac>0.2"""
    p = lp.exp()
    pb = p[:, 0]
    ent = -(p * lp).sum(-1)
    srt = p[:, 1:].sort(-1, descending=True).values
    top1 = srt[:, 0]
    margin = (srt[:, 0] - srt[:, 1]).mean().item()
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
    return {
        "p_blank": pb.mean().item(),
        "entropy": ent.mean().item(),
        "margin": margin,
        "peak_sharp": (sum(nb) / len(nb)) if nb else 0.0,
        "frac_gt02": (top1 > 0.2).float().mean().item(),
        "n_peak": len(nb),
        "T": int(lp.shape[0]),
    }


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
    log("官方 TFNet 载入完成（strict=True，0 missing）")

    P.RGB_FIX = True
    P.OFFICIAL_PAD = True
    labels = P.read_csv_split("dev.csv")
    _, tf = P.build_transforms(hflip=False, img_size=224)
    ds = P.RGBSeqDataset("dev", labels, word2idx, tf, False)
    if N_DEV:
        ds.items = ds.items[:N_DEV]

    ls = torch.nn.LogSoftmax(dim=-1)
    cache, lens, refs, stats = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            batch = P.collate([ds[i]])
            vid, _t, _tl, dl, _tl_len, sids, _ids = batch
            out = model(vid.to(dev), dl, False)
            lp = ls(out[0])[:, 0, :]
            L = min(P._conv_len(int(dl.view(-1)[0])), lp.shape[0])
            lp = lp[:L]
            stats.append(frame_stats(lp))
            cache.append(lp.to("cpu", torch.float32))
            lens.append(int(dl.view(-1)[0]))
            for b in sids:
                refs.append(labels[b][1])
    log("推理 %d 条 %.0fs" % (len(ds), time.time() - t0))

    P.BLANK_BIAS = 0.0
    hyps, ntok = [], 0
    for lp, L in zip(cache, lens):
        t = P.greedy_decode(lp.unsqueeze(1), torch.LongTensor([[L]]), idx2word)
        hyps += t
        ntok += sum(len(q) for q in t)
    o = evaluate(refs, hyps)
    gen = 100.0 * ntok / max(o["ref_tokens"], 1)

    def agg(k):
        return sum(s[k] for s in stats) / len(stats)

    cur = {"p_blank": agg("p_blank"), "peak_sharp": agg("peak_sharp"),
           "entropy": agg("entropy"), "margin": agg("margin"),
           "frac_gt02": 100 * agg("frac_gt02"), "n_peak": agg("n_peak"),
           "avg_T": agg("T")}
    ed = o.get("edits", {})
    tot = max(sum(ed.values()) if isinstance(ed, dict) else 1, 1)

    print()
    log("──── 官方 TFNet vs 我们 P123(VAC)：帧级后验 ────")
    log("  %-14s %12s %12s %10s" % ("指标", "官方TFNet", "我们P123", "差"))
    for k, name in (("p_blank", "p_blank"), ("peak_sharp", "峰锐度"),
                    ("entropy", "熵"), ("margin", "margin")):
        if k in OURS:
            log("  %-14s %12.4f %12.4f %+10.4f"
                % (name, cur[k], OURS[k], cur[k] - OURS[k]))
        else:
            log("  %-14s %12.4f %12s" % (name, cur[k], "-"))
    log("  %-14s %12.1f%% %11.1f%% %+10.1f pp"
        % ("生成率", gen, OURS["gen_rate"], gen - OURS["gen_rate"]))
    log("  %-14s %12.2f %12.2f %+10.2f pp"
        % ("WER(语料级)", o["WER_official"], OURS["WER"],
           o["WER_official"] - OURS["WER"]))
    log("  %-14s %12.1f %12.1f" % ("峰/样本", cur["n_peak"], 4.3))
    log("  %-14s %12.1f%%" % ("frac>0.2", cur["frac_gt02"]))
    if isinstance(ed, dict) and ed:
        log("  误差构成：sub %.1f%% / del %.1f%% / ins %.1f%%"
            % (100 * ed.get("sub", 0) / tot, 100 * ed.get("del", 0) / tot,
               100 * ed.get("ins", 0) / tot))
    log("  （我们：sub %.1f%% / del %.1f%% / ins %.1f%%）"
        % (OURS["sub"], OURS["del"], OURS["ins"]))

    out = REPO / "artifacts/metrics/blank-gov/p153-official-posterior.json"
    out.write_text(json.dumps({
        "ckpt": CK.name, "module": "TFNet", "img_size": 224, "n": len(ds),
        "rgb_fix": True, "official_pad": True,
        "official": {"wer": o["WER_official"], "gen_rate": gen, **cur,
                     "edits": ed},
        "ours_p123": OURS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
