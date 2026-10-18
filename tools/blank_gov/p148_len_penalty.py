#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P148：beam search + length penalty（**对症欠生成**）

动机（P146 实测）
  p_blank = 0.875、生成率仅 77.2%、del 占误差 43.2% ⇒ **欠生成**是主矛盾
  P147 已证：纯 beam（无长度惩罚）只值 -0.17pp ⇒ beam 本身不解决欠生成

依据【有】
  ref15 Camgoz et al., Sign Language Transformers (CVPR 2020), p.6 原文：
  "At inference time, we utilize beam search decoding with widths ranging from
   0 to 10. We also implement a length penalty [74] with alpha values ranging
   from 0 to 2. We find the best performing combination of beam width and alpha
   on the development set and use these values for the test set evaluation."
  ⇒ CSLR 的 SOTA 做法就是 beam + 长度惩罚，alpha 在 dev 上选、test 上直接用

判据（预先写死，防作弊）
  基线 = greedy + bias0.75 = 54.71（P147 实测 dev 515）
  改善 >= 1.556 pp（噪声基准）才算「可判有效」，否则记为不可判
  ⚠️ alpha 在 dev 上扫 ⇒ 有乐观风险；若要对外报数必须 test 复核

用法
  python tools/blank_gov/p148_len_penalty.py [n_dev] [beam_width]
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
    print("[P148 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def prefix_beam_search(lp, beam_width=BEAM, blank=BLANK, topk=TOPK,
                       alpha=0.0):
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
                           key=lambda kv: -(lse(kv[1][0], kv[1][1])
                                            + alpha * len(kv[0])))[:beam_width])
    if not beam:
        return []
    best = max(beam.items(),
               key=lambda kv: lse(kv[1][0], kv[1][1]) + alpha * len(kv[0]))
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

    ALPHAS = (0.0, 0.2, 0.5, 1.0, 1.5, 2.0)
    BASELINE = 54.71          # greedy + bias0.75，P147 实测 dev 515
    NOISE = 1.556             # 噪声基准

    log("网格：bias {0, 0.75} x alpha %s，beam=%d" % (list(ALPHAS), BEAM))
    rows = {}
    for bias in (0.0, 0.75):
        for al in ALPHAS:
            hyps, ntok = [], 0
            tb = time.time()
            for lp, L in zip(cache, lens):
                x = lp[:L] if lp.shape[0] > L else lp
                if bias:
                    x = x.clone()
                    x[:, BLANK] -= bias
                ids = prefix_beam_search(x, BEAM, alpha=al)
                hyps.append([idx2word[j] if 0 <= j < len(idx2word) else "<unk>"
                             for j in ids])
                ntok += len(ids)
            o = evaluate(refs, hyps)
            g = 100.0 * ntok / max(o["ref_tokens"], 1)
            rows["bias%.2f_a%.1f" % (bias, al)] = {
                "WER": o["WER_official"], "gen_rate": g}
            log("  bias %.2f  alpha %.1f  WER %6.2f%%  生成率 %5.1f%%  (%.0fs)"
                % (bias, al, o["WER_official"], g, time.time() - tb))

    print()
    log("──── 网格结果（dev %d 条，beam=%d）────" % (len(cache), BEAM))
    log("  %-8s %10s %10s %12s" % ("alpha", "bias0", "bias0.75", "生成率(b0.75)"))
    for al in ALPHAS:
        a0 = rows["bias0.00_a%.1f" % al]["WER"]
        a7 = rows["bias0.75_a%.1f" % al]["WER"]
        g7 = rows["bias0.75_a%.1f" % al]["gen_rate"]
        log("  %-8.1f %10.2f %10.2f %11.1f%%" % (al, a0, a7, g7))

    best_k = min(rows, key=lambda k: rows[k]["WER"])
    best_w = rows[best_k]["WER"]
    d = best_w - BASELINE
    print()
    log("──── 判定（判据启动前已写死）────")
    log("  基线 greedy+bias0.75 = %.2f（P147 实测）" % BASELINE)
    log("  最佳组合 %s = %.2f   Δ %+.2f pp" % (best_k, best_w, d))
    if d <= -NOISE:
        log("  ⇒ ✅ 可判有效（|Δ| > 噪声 %.3f）" % NOISE)
    elif d >= NOISE:
        log("  ⇒ ❌ 可判有害")
    else:
        log("  ⇒ ⚪ 不可判（Δ 在噪声内）")
    log("  ⚠️ alpha 在 dev 上选出 ⇒ 有乐观风险；对外报数前必须 test 复核")

    out = REPO / "artifacts/metrics/blank-gov/p148-len-penalty.json"
    out.write_text(json.dumps({
        "ckpt": CK.name, "epoch": ck.get("epoch"), "n": len(cache),
        "beam_width": BEAM, "topk": TOPK, "alphas": list(ALPHAS),
        "baseline_greedy_bias075": BASELINE, "noise": NOISE,
        "rows": rows, "best": {"key": best_k, "WER": best_w, "delta_pp": d},
        "sanity_greedy_ok": bool(ok),
        "ref": "ref15 Camgoz SLRT CVPR2020 p.6: beam width 0~10 + length penalty alpha 0~2",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
