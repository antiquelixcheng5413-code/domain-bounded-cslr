#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P146：CTC 后验校准 / peakiness 诊断（纯评估，零训练）

目的
  给三件「只有现象、没有机理」的事一个量化解释：
    1. 为什么 sub 占误差 54.4%（大头），del 只 43.2%
    2. 为什么 blank bias 0.75 能改善 ~1pp（模型是否系统性低估 blank）
    3. 为什么 beam=10 让输出长度 +109%（后验是否"不尖" ⇒ 每帧都想吐词）

帧级指标（bias=0 的原始后验，未经任何修正）
  p_blank_mean    帧级 blank 概率均值（低 ⇒ 模型不愿停 ⇒ del 少？）
  ent_mean        帧级后验熵均值（低 ⇒ 尖峰）
  margin_mean     帧级 top1-top2 概率差均值（大 ⇒ 自信且尖）
  frac_hot05      非 blank top1 > 0.5 的帧占比（**高 ⇒ beam 必崩**）
  n_hot05         满足上条的帧数绝对值
  gen_len / ref_len   输出长度 vs 参考长度（生成率）

关键对照
  **train 子集 vs dev**：若 train 后验明显更"尖"而 dev 平 ⇒ 过拟合的后验层面证据

用法
  python tools/blank_gov/p146_calibration_probe.py [n_dev] [n_train]
    默认 515 / 300
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
N_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 0      # 0 = 全量 515
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 300
SEED = 42
DEV_REF_WER = 55.84      # 已知答案：dev 全量、bias=0 必须复现这个数


def log(*a):
    print("[P146 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def frame_stats(lp):
    """lp: [T, V] log-prob（float）。全在 GPU 上算，只回标量。

    核心指标 peak_sharp：greedy 折叠后**每个输出 token 所对应峰内的最大非blank 概率**。
    训练到位的 CTC 模型峰心应接近 1；若普遍只有 0.1~0.3 ⇒ 模型是在"蒙"。
    """
    p = lp.exp()
    pb = p[:, 0]
    ent = -(p * lp).sum(-1)
    srt = p[:, 1:].sort(-1, descending=True).values
    top1 = srt[:, 0]
    margin = (srt[:, 0] - srt[:, 1]).mean().item()
    hot = (top1 > 0.5)

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
        "p_blank_mean": pb.mean().item(),
        "ent_mean": ent.mean().item(),
        "margin_mean": margin,
        "pnb_top1_mean": top1.mean().item(),
        "frac_hot05": hot.float().mean().item(),
        "frac_gt02": (top1 > 0.2).float().mean().item(),
        "n_hot05": int(hot.sum().item()),
        "n_peak": len(nb),
        "peak_sharp": (sum(nb) / len(nb)) if nb else 0.0,
        "T": int(lp.shape[0]),
    }



def run(split, labels, idxs, word2idx, tf, model, dev, idx2word):
    ds = P.RGBSeqDataset(split, labels, word2idx, tf, False)
    if idxs is not None:
        ds.items = [ds.items[i] for i in idxs]
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
            cache.append(lp.to("cpu", torch.float16))
            lens.append(int(dl.view(-1)[0]))
            for b in sids:
                refs.append(labels[b][1])
    log("%-5s %d 条推理 %.0fs" % (split, len(ds), time.time() - t0))

    def decode(bias):
        hyps, ntok = [], 0
        P.BLANK_BIAS = float(bias)
        for lp16, L in zip(cache, lens):
            lp = lp16.float().unsqueeze(1)
            if P.BLANK_BIAS:
                lp = lp.clone()
                lp[..., 0] -= P.BLANK_BIAS
            toks = P.greedy_decode(lp, torch.LongTensor([[L]]), idx2word)
            hyps += toks
            ntok += sum(len(t) for t in toks)
        return hyps, ntok

    res = {}
    for b in (0.0, 0.75):
        hyps, ntok = decode(b)
        o = evaluate(refs, hyps)
        res["bias%.2f" % b] = {
            "WER": o["WER_official"], "ref_tokens": o["ref_tokens"],
            "gen_rate": 100.0 * ntok / max(o["ref_tokens"], 1),
        }
    P.BLANK_BIAS = 0.0

    def agg(k):
        v = [s[k] for s in stats]
        return sum(v) / len(v)

    agg_d = {
        "n": len(stats),
        "p_blank_mean": agg("p_blank_mean"),
        "ent_mean": agg("ent_mean"),
        "margin_mean": agg("margin_mean"),
        "frac_hot05": agg("frac_hot05"),
        "frac_gt02": agg("frac_gt02"),
        "pnb_top1_mean": agg("pnb_top1_mean"),
        "n_peak": agg("n_peak"),
        "peak_sharp": agg("peak_sharp"),
        "avg_T": agg("T"),
        "hot_per_sample": agg("n_hot05"),
    }
    return res, agg_d, refs, cache, lens


def main() -> int:
    dev = "cuda"
    log("ckpt = %s" % CK.name)
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")

    ck = torch.load(CK, map_location="cpu", weights_only=False)
    P.OFFICIAL_PAD = bool(ck.get("config", {}).get("official_pad", False))
    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    log("载入 epoch %s；official_pad=%s（取自 ckpt）" % (ck.get("epoch"), P.OFFICIAL_PAD))
    _, tf = P.build_transforms(hflip=False, img_size=160)

    # ---- dev ----
    lab_dev = P.read_csv_split("dev.csv")
    idx_dev = None if not N_DEV else list(range(min(N_DEV, len(lab_dev))))
    r_dev, a_dev, refs_dev, _, _ = run(
        "dev", lab_dev, idx_dev, word2idx, tf, model, dev, idx2word)

    # ---- train 子集（seed 固定，可复现）----
    lab_tr = P.read_csv_split("train.csv")
    rnd = random.Random(SEED)
    n_tr = min(N_TRAIN, len(lab_tr))
    idx_tr = rnd.sample(range(len(lab_tr)), n_tr)
    r_tr, a_tr, _, _, _ = run(
        "train", lab_tr, idx_tr, word2idx, tf, model, dev, idx2word)

    # ---- 已知答案验证 ----
    print()
    log("──── 校验：dev 全量 bias=0 必须 = %.2f ────" % DEV_REF_WER)
    got = r_dev["bias0.00"]["WER"]
    ok = abs(got - DEV_REF_WER) < 0.05
    log("  实测 %.2f  ⇒ %s" % (got, "✅ 代码路径正确" if ok else "❌ 路径有误，结论作废"))

    print()
    log("──── 帧级后验统计（bias=0 原始输出）────")
    log("%-6s %8s %8s %8s %8s %9s %9s %7s" %
        ("split", "p_blank", "pnb_top1", "margin", "entropy", "peak锐度", "frac>0.2", "峰/样本"))
    for nm, a in (("dev", a_dev), ("train", a_tr)):
        log("%-6s %8.4f %8.4f %8.4f %8.3f %9.3f %8.1f%% %7.1f" %
            (nm, a["p_blank_mean"], a["pnb_top1_mean"], a["margin_mean"],
             a["ent_mean"], a["peak_sharp"], 100 * a["frac_gt02"], a["n_peak"]))
    log("  ⇒ train 与 dev 的差 = 过拟合在后验层面的证据")
    log("    Δentropy %+.3f   Δp_blank %+.4f   Δpeak锐度 %+.3f   Δfrac>0.2 %+.1f pp"
        % (a_tr["ent_mean"] - a_dev["ent_mean"],
           a_tr["p_blank_mean"] - a_dev["p_blank_mean"],
           a_tr["peak_sharp"] - a_dev["peak_sharp"],
           100 * (a_tr["frac_gt02"] - a_dev["frac_gt02"])))

    print()
    log("──── 生成率与 beam 崩溃的机理 ────")
    for nm, r, a in (("dev", r_dev, a_dev), ("train", r_tr, a_tr)):
        hot = a["hot_per_sample"]
        gen_rate = r["bias0.00"]["gen_rate"]
        gen_tok = r["bias0.00"]["ref_tokens"] / max(a["n"], 1) * gen_rate / 100.0
        ratio = hot / max(gen_tok, 1e-6)
        log("  %-5s 生成率 %5.1f%%，实心峰 %.1f 帧/样本，实际输出 %.1f token/样本"
            % (nm, gen_rate, hot, gen_tok))
        log("        实心峰/实际输出 = %.2f  ⇒ 每吐一个词就有 %.1f 帧处于「高置信非blank」"
            % (ratio, ratio))
    log("  ⇒ 该比值越大于 1，beam 越容易在峰之间插重复词（P?: beam10 输出 +109%% 与此同源）")

    print()
    log("──── dev 上的 bias 复核（应与 P134 一致）────")
    log("  bias=0    %.2f（生成率 %.1f%%）"
        % (r_dev["bias0.00"]["WER"], r_dev["bias0.00"]["gen_rate"]))
    log("  bias=0.75 %.2f（生成率 %.1f%%）  Δ %+.2f pp"
        % (r_dev["bias0.75"]["WER"], r_dev["bias0.75"]["gen_rate"],
           r_dev["bias0.75"]["WER"] - r_dev["bias0.00"]["WER"]))

    out = REPO / "artifacts/metrics/blank-gov/p146-calibration-probe.json"
    out.write_text(json.dumps({
        "ckpt": CK.name, "epoch": ck.get("epoch"),
        "n_dev": a_dev["n"], "n_train": a_tr["n"], "seed": SEED,
        "sanity_dev_wer_bias0": {"got": got, "expected": DEV_REF_WER, "ok": bool(ok)},
        "dev": {"wer": r_dev, "posterior": a_dev},
        "train": {"wer": r_tr, "posterior": a_tr},
        "delta_train_minus_dev": {
            "entropy": a_tr["ent_mean"] - a_dev["ent_mean"],
            "p_blank": a_tr["p_blank_mean"] - a_dev["p_blank_mean"],
            "frac_hot05": a_tr["frac_hot05"] - a_dev["frac_hot05"],
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
