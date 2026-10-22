#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P166：Gloss 语言模型浅融合（shallow fusion）

为什么做
  sub 占误差 **52.9%**（认错词）。视觉判别力短期提不动，但语言模型正好治"认错词"。
  【有】ASR 领域 shallow fusion 是标准做法；【间接】CSLR gloss 识别上。

🔴 有效性红线
  LM **只用 train split 的 gloss 序列**训练，dev / test 绝不参与 ⇒ 评估才有效。
  （官方不用 LM ⇒ 这是我们对官方的偏离，报数时必须注明。）

两段式（为了不占 GPU）
  阶段1 `cache`：跑一次 dev 推理（约 118s），把 log-prob 存盘
  阶段2 `grid` ：加载缓存，建 LM，跑 λ 网格 —— **全程 CPU**
  ⇒ D4 开跑后，LM 还能继续调参而不抢卡。

用法
  ./venv/bin/python -u tools/blank_gov/p166_gloss_lm.py cache
  ./venv/bin/python -u tools/blank_gov/p166_gloss_lm.py grid
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))

from official_wer import evaluate  # noqa: E402
from official_wer import split_gloss_sequence, split_glued, CFG  # noqa: E402

CACHE = REPO / "artifacts/cache/p150_dev_lp.pkl"
CK = REPO / "artifacts/checkpoints/p150-rgbfix-best.pt"
REF_WER = 53.73                      # 已知答案：dev 全量、greedy / bias=0
NEG = -1e30


def log(*a):
    print("[P166 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def lse(a, b):
    if a < b:
        a, b = b, a
    if b <= -1e29:
        return a
    return a + math.log1p(math.exp(b - a))


# ============================================================
# 阶段 1：缓存 dev 后验（唯一需要 GPU 的部分）
# ============================================================
def stage_cache() -> int:
    import torch
    import p78_train_official_tfnet as P

    P.RGB_FIX = True                 # ★ P150 是 RGB 训练的
    dev = "cuda"
    labels = P.read_csv_split("dev.csv")
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")

    ck = torch.load(CK, map_location="cpu", weights_only=False)
    assert "model_state" in ck, "键名变了：%s" % list(ck)[:8]
    P.OFFICIAL_PAD = bool(ck.get("config", {}).get("official_pad", False))
    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    log("载入 %s（epoch %s）" % (CK.name, ck.get("epoch")))

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
            cache.append(lp[:L].to("cpu", torch.float16).numpy())
            lens.append(int(dl.view(-1)[0]))
            for b in sids:
                refs.append(labels[b][1])
    log("dev %d 条推理 %.0fs" % (len(ds), time.time() - t0))

    # 已知答案校验：greedy 必须复现 53.73
    P.BLANK_BIAS = 0.0
    hyps = []
    for lp16, L in zip(cache, lens):
        lp = torch.from_numpy(lp16.astype("float32")).unsqueeze(1)
        hyps += P.greedy_decode(lp, torch.LongTensor([[L]]), idx2word)
    o = evaluate(refs, hyps)
    log("校验 greedy WER = %.2f（应 %.2f）⇒ %s"
        % (o["WER_official"], REF_WER,
           "OK 路径正确" if abs(o["WER_official"] - REF_WER) < 0.05 else "!! 结论作废"))

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({"lps": cache, "lens": lens, "refs": refs,
                     "idx2word": idx2word, "ckpt": CK.name}, f, protocol=4)
    log("缓存 -> %s（%.0f MB）" % (CACHE, CACHE.stat().st_size / 1e6))
    return 0


# ============================================================
# 阶段 2：建 LM（只用 train）+ CPU 网格
# ============================================================
def build_lm(order=2, smooth=0.1):
    """用 **train** 的 gloss 序列建 n-gram LM（bigram，回退到 unigram）。"""
    import p78_train_official_tfnet as P
    labs = P.read_csv_split("train.csv")
    seqs = []
    for k in labs:
        g = labs[k][1] if isinstance(labs, dict) else k[1]
        seqs.append(split_glued(split_gloss_sequence(g, CFG)))
    uni = Counter()
    bi = defaultdict(Counter)
    for s in seqs:
        for w in s:
            uni[w] += 1
        for a, b in zip(s, s[1:]):
            bi[a][b] += 1
    tot_uni = sum(uni.values())
    V = max(len(uni), 1)
    return {"uni": uni, "bi": bi, "tot_uni": tot_uni, "V": V,
            "smooth": smooth, "n_seq": len(seqs), "order": order}


def lm_logp(lm, prev, w):
    """logP(w | prev)，带回退与平滑。"""
    uni, bi, V, tot = lm["uni"], lm["bi"], lm["V"], lm["tot_uni"]
    a = lm["smooth"]
    if prev is not None and prev in bi:
        c = bi[prev]
        den = sum(c.values()) + a * V
        num = c.get(w, 0) + a * (uni.get(w, 0) + 1.0) / (tot + V)
        if den > 0:
            return math.log(max(num, 1e-12)) - math.log(den)
    # 回退到 unigram
    return math.log(max(uni.get(w, 0) + a, 1e-12)) - math.log(tot + a * V)


def beam_with_lm(lp, lm, lam, beam_width=10, topk=40, blank=0, word_bonus=0.0):
    """CTC prefix beam + LM 浅融合。lp: [T,V] float32 numpy。"""
    T, Vv = lp.shape
    k = min(topk, Vv)
    idx = lp.argpartition(-k, axis=-1)[:, -k:]          # [T,k] 候选 id
    cand = []
    for t in range(T):
        ids = idx[t]
        vs = [float(lp[t, i]) for i in ids]
        pb = float(lp[t, blank])
        items = [(-1, pb)] + [(int(i), v) for i, v in zip(ids, vs) if int(i) != blank]
        cand.append(sorted(items, key=lambda x: -x[1])[:k])

    beam = {(): (0.0, NEG, 0.0)}                         # pref -> (pb, pnb, lm)
    for t in range(T):
        new = {}
        for pref, (pb, pnb, lmsc) in beam.items():
            ptot = lse(pb, pnb)
            for cid, cv in cand[t]:
                if cid < 0:                              # blank
                    b = new.get(pref, (NEG, NEG, lmsc))
                    new[pref] = (lse(b[0], ptot + cv), b[1], lmsc)
                    continue
                last = pref[-1] if pref else None
                if pref and cid == last:
                    b = new.get(pref, (NEG, NEG, lmsc))
                    new[pref] = (b[0], lse(b[1], pnb + cv), lmsc)
                np_ = pref + (cid,)
                add = lam * lm_logp(lm, last, cid) + word_bonus
                b = new.get(np_, (NEG, NEG, NEG))
                new[np_] = (b[0], lse(b[1], ptot + cv), max(b[2], lmsc + add))
        beam = dict(sorted(new.items(),
                           key=lambda kv: -(lse(kv[1][0], kv[1][1]) + kv[1][2]))[:beam_width])
    if not beam:
        return []
    best = max(beam.items(), key=lambda kv: lse(kv[1][0], kv[1][1]) + kv[1][2])
    return list(best[0])


def stage_grid() -> int:
    if not CACHE.exists():
        log("!! 没有缓存，请先跑：... p166_gloss_lm.py cache")
        return 1
    with open(CACHE, "rb") as f:
        d = pickle.load(f)
    lps, lens, refs, idx2word = d["lps"], d["lens"], d["refs"], d["idx2word"]
    log("载入缓存 %d 条（ckpt %s）" % (len(lps), d["ckpt"]))

    lm = build_lm()
    log("LM：只用 **train** 的 %d 条序列，词表 %d，bigram 上下文 %d 个"
        % (lm["n_seq"], lm["V"], len(lm["bi"])))

    def decode(lam, bw, wb=0.0):
        hyps = []
        for lp16, L in zip(lps, lens):
            lp = lp16.astype("float32")
            ids = beam_with_lm(lp, lm, lam, beam_width=bw, word_bonus=wb)
            hyps.append([idx2word[i] for i in ids])
        return hyps

    print()
    log("──── λ 网格（beam=10，topk=40），全程 CPU ────")
    log("  %-8s %10s %11s %10s" % ("λ(LM权重)", "WER", "生成率", "Δ vs λ=0"))
    if len(sys.argv) > 2:                    # 允许自定义 λ 网格（扫低端甜点）
        grid = [float(x) for x in sys.argv[2:]]
        log("（自定义 λ 网格：%s）" % grid)
    else:
        grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    best, rows = None, {}
    for lam in grid:
        t0 = time.time()
        hyps = decode(lam, 10)
        o = evaluate(refs, hyps)
        ntok = sum(len(h) for h in hyps)
        g = 100.0 * ntok / max(o["ref_tokens"], 1)
        rows["%.1f" % lam] = {"WER": o["WER_official"], "gen_rate": g}
        if best is None:
            base = o["WER_official"]
        log("  %-8.1f %10.2f %10.1f%% %+10.2f   (%.0fs)"
            % (lam, o["WER_official"], g, o["WER_official"] - base, time.time() - t0))
        if best is None or o["WER_official"] < best[1]:
            best = (lam, o["WER_official"])

    print()
    log("校验：λ=0 时应≈greedy 的 %.2f ⇒ 实测 %.2f"
        % (REF_WER, rows["0.0"]["WER"]))
    log("最优 λ = %s，WER %.2f（Δ %+.2f pp vs λ=0）"
        % (best[0], best[1], best[1] - rows["0.0"]["WER"]))
    print()
    log("🔴 判定提醒：噪声阈值 1.556pp。若改善 < 1.556pp ⇒ 属'不可判'，别当结论。")
    log("🔴 口径提醒：官方不用 LM ⇒ 这是偏离；若有效，报数时必须注明含 LM。")

    out = REPO / "artifacts/metrics/blank-gov/p166-gloss-lm.json"
    out.write_text(json.dumps({
        "ckpt": d["ckpt"], "lm_train_split_only": True,
        "lm": {"n_seq": lm["n_seq"], "vocab": lm["V"], "bigram_ctx": len(lm["bi"])},
        "grid": rows, "best_lambda": best[0], "best_wer": best[1],
        "baseline_lambda0": rows["0.0"]["WER"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


def stage_bonus() -> int:
    """负面结果的稳健性检验：LM 会抑制输出（生成率 77.8%→2.9%），
    补一个"词插入奖励"β 看能不能救回来。"""
    if not CACHE.exists():
        log("!! 没有缓存，先跑 cache")
        return 1
    with open(CACHE, "rb") as f:
        d = pickle.load(f)
    lps, lens, refs, idx2word = d["lps"], d["lens"], d["refs"], d["idx2word"]
    lm = build_lm()
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    betas = [float(x) for x in sys.argv[3:]] or [0.0, 1.0, 2.0, 3.0, 5.0]

    def decode(wb):
        hyps = []
        for lp16, L in zip(lps, lens):
            lp = lp16.astype("float32")
            ids = beam_with_lm(lp, lm, lam, beam_width=10, word_bonus=wb)
            hyps.append([idx2word[i] for i in ids])
        return hyps

    print()
    log("──── 词插入奖励 β 扫描（λ 固定 %.2f，beam=10）────" % lam)
    log("  %-8s %10s %11s %10s" % ("β", "WER", "生成率", "Δ vs 基线"))
    base = None
    best = None
    rows = {}
    for b in betas:
        hyps = decode(b)
        o = evaluate(refs, hyps)
        ntok = sum(len(h) for h in hyps)
        g = 100.0 * ntok / max(o["ref_tokens"], 1)
        rows["%.1f" % b] = {"WER": o["WER_official"], "gen_rate": g}
        if base is None:
            base = 53.73                      # greedy 基线
        log("  %-8.1f %10.2f %10.1f%% %+10.2f" % (b, o["WER_official"], g,
                                                  o["WER_official"] - base))
        if best is None or o["WER_official"] < best[1]:
            best = (b, o["WER_official"])
    print()
    log("最优 β = %s，WER %.2f（vs greedy 基线 53.73：%+.2f pp）"
        % (best[0], best[1], best[1] - 53.73))
    log("⇒ 若仍无改善 ⇒ n-gram LM 这条线可以判定为死路")

    out = REPO / "artifacts/metrics/blank-gov/p166-gloss-lm-bonus.json"
    out.write_text(json.dumps({"lam": lam, "rows": rows, "best": best[0],
                               "best_wer": best[1]},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if mode == "cache":
        raise SystemExit(stage_cache())
    if mode == "bonus":
        raise SystemExit(stage_bonus())
    raise SystemExit(stage_grid())
