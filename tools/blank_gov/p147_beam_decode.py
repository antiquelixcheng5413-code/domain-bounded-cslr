#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P147：官方解码口径复核 —— 官方评估用的是 **beam search**，不是 greedy

🔴 源码核对（2026-10-18）
  external/TFNet/Train.py:152   decoder = decode.Decode(word2idx, wordSetNum+1, "beam")
  external/TFNet/decode.py:24   ctcdecode.CTCBeamDecoder(vocab, beam_width=10,
                                                        blank_id=0, num_processes=10)
  external/TFNet/Train.py:282 / 384  训练中与**验证时**都调用 decoder.decode(...)
  ⇒ 官方 dev 42.1 / test 41.9 是 **beam_width=10** 解码得到的，不是 greedy
  ⇒ 我们全程 greedy ⇒ 这是真实口径差异（旧记忆写成"官方用 greedy"，已更正）

  官方 decode.py **没有传 LM / 词典** ⇒ 等价于纯 CTC beam search。
  ctcdecode==0.4 在本环境装不上（源里只有 1.0.1/1.0.2）⇒ 本脚本自己实现
  标准 CTC prefix beam search（Graves 2006），每帧取 top-40 候选
  （对齐 ctcdecode 默认 cutoff_top_n=40）。

用法
  python tools/blank_gov/p147_beam_decode.py [n_dev] [beam_width]
"""
import json
import math


def lse(a, b):
    """log(exp(a)+exp(b)) 稳定版（本环境的 math 没有 logaddexp）。"""
    if a < b:
        a, b = b, a
    if b <= -1e29:
        return a
    return a + math.log1p(math.exp(b - a))
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
BEAM = int(sys.argv[2]) if len(sys.argv) > 2 else 10
TOPK = 40            # 对齐 ctcdecode 默认 cutoff_top_n=40
BLANK = 0
NEG = -1e30
DEV_REF_GREEDY = 55.84


def log(*a):
    print("[P147 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def prefix_beam_search(lp, beam_width=BEAM, blank=BLANK, topk=TOPK):
    """标准 CTC prefix beam search（log 空间）。lp: [T, V] log-prob。返回 id 列表。"""
    T, V = lp.shape
    k = min(topk, V)
    topv, topi = lp.topk(k, dim=-1)                 # [T, k]
    tv = topv.tolist()
    ti = topi.tolist()

    beam = {(): (0.0, NEG)}                          # prefix -> (p_blank, p_nonblank)
    for t in range(T):
        row_v, row_i = tv[t], ti[t]
        p_blank_t = NEG
        cand = []                                    # (id, logp)
        for v, i in zip(row_v, row_i):
            if i == blank:
                p_blank_t = v
            else:
                cand.append((i, v))
        if p_blank_t == NEG:                         # blank 不在 top-k 内，补算
            p_blank_t = lp[t, blank].item()
        new = {}
        for pref, (pb, pnb) in beam.items():
            ptot = lse(pb, pnb)
            # 1) 走 blank
            b = new.get(pref, (NEG, NEG))
            new[pref] = (lse(b[0], ptot + p_blank_t), b[1])
            if pref:
                last = pref[-1]
                for cid, cv in cand:
                    if cid == last:
                        # 2) 重复上一字符（不延长 prefix）
                        b = new.get(pref, (NEG, NEG))
                        new[pref] = (b[0], lse(b[1], pnb + cv))
                    # 3) 延长 prefix
                    np_ = pref + (cid,)
                    b = new.get(np_, (NEG, NEG))
                    new[np_] = (b[0], lse(b[1], ptot + cv))
            else:
                for cid, cv in cand:
                    np_ = (cid,)
                    b = new.get(np_, (NEG, NEG))
                    new[np_] = (b[0], lse(b[1], ptot + cv))
        # 裁剪到 beam_width
        beam = dict(sorted(new.items(),
                           key=lambda kv: -lse(kv[1][0], kv[1][1]))[:beam_width])
    if not beam:
        return []
    best = max(beam.items(), key=lambda kv: lse(kv[1][0], kv[1][1]))
    return list(best[0])


def main() -> int:
    dev = "cuda"
    log("ckpt=%s  beam=%d  topk=%d" % (CK.name, BEAM, TOPK))
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    P.OFFICIAL_PAD = bool(ck.get("config", {}).get("official_pad", False))
    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    _, tf = P.build_transforms(hflip=False, img_size=160)

    labels = P.read_csv_split("dev.csv")
    ds = P.RGBSeqDataset("dev", labels, word2idx, tf, False)
    if N_DEV:
        ds.items = ds.items[:N_DEV]
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
    log("推理 %d 条 %.0fs" % (len(ds), time.time() - t0))

    # ---- greedy（我们的口径）----
    P.BLANK_BIAS = 0.0
    hyps_g, ntok_g = [], 0
    for lp, L in zip(cache, lens):
        t = P.greedy_decode(lp.unsqueeze(1), torch.LongTensor([[L]]), idx2word)
        hyps_g += t
        ntok_g += sum(len(x) for x in t)
    o_g = evaluate(refs, hyps_g)
    log("greedy    WER %.2f%%（生成率 %.1f%%）" %
        (o_g["WER_official"], 100.0 * ntok_g / max(o_g["ref_tokens"], 1)))
    if N_DEV == 0:
        ok = abs(o_g["WER_official"] - DEV_REF_GREEDY) < 0.05
        log("  已知答案校验 55.84 ⇒ %s" % ("✅ 代码路径正确" if ok else "❌ 路径有误"))
    else:
        ok = True
        log("  子集 %d 条，跳过全量校验（全量才是 55.84）" % len(ds))

    # ---- beam（官方口径）x 两个 bias ----
    rows = {"greedy": {}, "beam": {}}
    for bias in (0.0, 0.75):
        hyps_b, ntok_b = [], 0
        tb = time.time()
        for lp, L in zip(cache, lens):
            x = lp[:L] if lp.shape[0] > L else lp
            if bias:
                x = x.clone()
                x[:, BLANK] -= bias
            ids = prefix_beam_search(x, BEAM)
            hyps_b.append([idx2word[j] if 0 <= j < len(idx2word) else "<unk>"
                           for j in ids])
            ntok_b += len(ids)
        o_b = evaluate(refs, hyps_b)
        rows["beam"]["bias%.2f" % bias] = {
            "WER": o_b["WER_official"],
            "gen_rate": 100.0 * ntok_b / max(o_b["ref_tokens"], 1)}
        log("beam-%d bias%.2f  WER %.2f%%（生成率 %.1f%%）%.0fs"
            % (BEAM, bias, o_b["WER_official"],
               100.0 * ntok_b / max(o_b["ref_tokens"], 1), time.time() - tb))
    # ---- greedy x 两个 bias（对照组）----
    for bias in (0.0, 0.75):
        P.BLANK_BIAS = float(bias)
        hyps_g, ntok_g = [], 0
        for lp, L in zip(cache, lens):
            x = lp.unsqueeze(1).clone()
            if bias:
                x[..., 0] -= bias
            t = P.greedy_decode(x, torch.LongTensor([[L]]), idx2word)
            hyps_g += t
            ntok_g += sum(len(q) for q in t)
        o = evaluate(refs, hyps_g)
        rows["greedy"]["bias%.2f" % bias] = {
            "WER": o["WER_official"],
            "gen_rate": 100.0 * ntok_g / max(o["ref_tokens"], 1)}
    P.BLANK_BIAS = 0.0
    o_g = rows["greedy"]["bias0.00"]
    o_b = rows["beam"]["bias0.00"]

    d = o_b["WER"] - o_g["WER"]
    print()
    log("──── 结论 ────")
    log("  %-14s %8s %8s" % ("口径", "bias0", "bias0.75"))
    for k in ("greedy", "beam"):
        log("  %-14s %8.2f %8.2f" % (k, rows[k]["bias0.00"]["WER"],
                                     rows[k]["bias0.75"]["WER"]))
    log("  greedy %6.2f  vs  beam-%d %6.2f   Δ %+6.2f pp" %
        (o_g["WER"], BEAM, o_b["WER"], d))
    if d < -1.556:
        log("  ⇒ ✅ beam 明显更好（超过噪声基准 1.556）⇒ 10.74pp 中有一部分是解码口径")
    elif d > 1.556:
        log("  ⇒ ❌ beam 更差 ⇒ 官方口径不是 beam 或我们的实现有差异")
    else:
        log("  ⇒ ⚪ 差异在噪声内，不可判")

    out = REPO / "artifacts/metrics/blank-gov/p147-beam-decode.json"
    out.write_text(json.dumps({
        "ckpt": CK.name, "epoch": ck.get("epoch"), "n": len(cache),
        "beam_width": BEAM, "topk": TOPK,
        "rows": rows,
        "delta_pp": d, "sanity_greedy_ok": bool(ok),
        "source": "Train.py:152 Decode(...,beam); decode.py:24 CTCBeamDecoder(beam_width=10)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
