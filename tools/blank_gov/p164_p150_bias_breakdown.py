#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P164：P150（RGB）上的 bias 重扫 + 误差构成（一次推理，两次复用）

用户点单 A + B：
  A. 在 P150 上重扫 blank bias（0.75 是在 BGR 模型上调的，新模型最优值可能不同）
  B. 重跑误差构成（sub/del/ins、生成率、按句长分桶）

设计：dev 515 只推理一次（约 111s），把 log-prob 缓存下来；
     之后每个 bias 只是从缓存重新解码，几乎免费。

⚠️ 两条已知答案校验（不过则结论作废）：
  ① bias=0 的 dev WER 必须 = 53.73
  ② ins+del+sub 必须等于 official_wer 的 edits
"""
from __future__ import annotations

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
from official_wer import split_gloss_sequence, split_glued, CFG  # noqa: E402

# ★ 关键：P150 是 RGB 训练的，评估必须同口径
P.RGB_FIX = True

CK = REPO / "artifacts/checkpoints/p150-rgbfix-best.pt"
REF_WER = 53.73                       # 已知答案：dev 全量 bias=0
BIAS_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


def log(*a):
    print("[P164 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def align(ref, hyp):
    """编辑距离 DP + 回溯 ⇒ (ins, del, sub)。"""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + c)
    ins = dele = sub = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            if D[i][j] == D[i - 1][j - 1] + c:
                if c:
                    sub += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and D[i][j] == D[i - 1][j] + 1:
            dele += 1
            i -= 1
            continue
        ins += 1
        j -= 1
    return ins, dele, sub


def main() -> int:
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
    log("载入 %s（epoch %s，official_pad=%s）" % (CK.name, ck.get("epoch"), P.OFFICIAL_PAD))

    _, tf = P.build_transforms(hflip=False, img_size=160)
    ds = P.RGBSeqDataset("dev", labels, word2idx, tf, False)
    ls = torch.nn.LogSoftmax(dim=-1)

    # ---- 只推理一次，缓存 log-prob ----
    cache, lens, refs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            batch = P.collate([ds[i]])
            vid, _t, _tl, dl, _tl_len, sids, _ids = batch
            out = model(vid.to(dev), dl, False)
            lp = ls(out[0])[:, 0, :]
            L = min(P._conv_len(int(dl.view(-1)[0])), lp.shape[0])
            cache.append(lp[:L].to("cpu", torch.float16))
            lens.append(int(dl.view(-1)[0]))
            for b in sids:
                refs.append(labels[b][1])
    log("dev %d 条推理 %.0fs（之后每个 bias 只重解码，免费）"
        % (len(ds), time.time() - t0))

    def decode(bias):
        hyps, ntok = [], 0
        P.BLANK_BIAS = float(bias)
        for lp16, L in zip(cache, lens):
            lp = lp16.float().unsqueeze(1)
            if bias:
                lp = lp.clone()
                lp[..., 0] -= bias
            t = P.greedy_decode(lp, torch.LongTensor([[L]]), idx2word)
            hyps += t
            ntok += sum(len(q) for q in t)
        P.BLANK_BIAS = 0.0
        return hyps, ntok

    # ================= A. bias 重扫 =================
    print()
    log("──── A. blank bias 重扫（dev 515）────")
    log("  %-7s %9s %11s %10s" % ("bias", "WER", "生成率", "Δ vs bias0"))
    scan, base = {}, None
    for b in BIAS_GRID:
        hyps, ntok = decode(b)
        o = evaluate(refs, hyps)
        g = 100.0 * ntok / max(o["ref_tokens"], 1)
        if base is None:
            base = o["WER_official"]
        scan["%.2f" % b] = {"WER": o["WER_official"], "gen_rate": g,
                            "delta": o["WER_official"] - base}
        log("  %-7.2f %9.2f %10.1f%% %+10.2f" % (b, o["WER_official"], g,
                                                 o["WER_official"] - base))

    got0 = scan["0.00"]["WER"]
    ok1 = abs(got0 - REF_WER) < 0.05
    log("  校验① bias=0 = %.2f（应 %.2f）⇒ %s"
        % (got0, REF_WER, "✅ 路径正确" if ok1 else "❌ 结论作废"))

    best_b = min(scan, key=lambda k: scan[k]["WER"])
    log("  ⇒ 最优 bias = %s，WER %.2f（Δ %+.2f pp）"
        % (best_b, scan[best_b]["WER"], scan[best_b]["delta"]))

    # ================= B. 误差构成 =================
    def breakdown(bias, tag):
        hyps, _ = decode(bias)
        o = evaluate(refs, hyps)
        ti = td = ts = 0
        n_h = n_r = 0
        exact = 0
        buckets = {}
        for r, h in zip(refs, hyps):
            rtok = split_glued(split_gloss_sequence(r, CFG))
            htok = split_glued(list(h))
            i_, d_, s_ = align(rtok, htok)
            ti += i_
            td += d_
            ts += s_
            n_h += len(htok)
            n_r += len(rtok)
            exact += (rtok == htok)
            L = len(rtok)
            k = "1-3" if L <= 3 else ("4-6" if L <= 6 else ("7-9" if L <= 9 else "10+"))
            b = buckets.setdefault(k, {"n": 0, "ins": 0, "del": 0, "sub": 0,
                                       "ref": 0, "hyp": 0})
            b["n"] += 1
            b["ins"] += i_
            b["del"] += d_
            b["sub"] += s_
            b["ref"] += len(rtok)
            b["hyp"] += len(htok)
        tot = ti + td + ts
        print()
        log("──── B. 误差构成（bias=%s，dev %d）────" % (tag, len(refs)))
        log("  sub 替换 : %5d (%5.1f%%)" % (ts, 100.0 * ts / max(tot, 1)))
        log("  del 漏词 : %5d (%5.1f%%)" % (td, 100.0 * td / max(tot, 1)))
        log("  ins 多说 : %5d (%5.1f%%)" % (ti, 100.0 * ti / max(tot, 1)))
        log("  合计     : %5d  （official_wer edits=%d ⇒ %s）"
            % (tot, o["edits"], "✅ 一致" if tot == o["edits"] else "❌ 不一致"))
        log("  生成率   : %.1f%%   整句全对 %d/%d = %.2f%%"
            % (100.0 * n_h / max(n_r, 1), exact, len(refs),
               100.0 * exact / len(refs)))
        log("  %-6s %5s %6s %6s %6s %9s %9s" %
            ("句长", "句数", "sub", "del", "ins", "生成率", "该桶WER"))
        for k in ("1-3", "4-6", "7-9", "10+"):
            if k not in buckets:
                continue
            b = buckets[k]
            e = b["sub"] + b["del"] + b["ins"]
            log("  %-6s %5d %6d %6d %6d %8.1f%% %8.2f%%"
                % (k, b["n"], b["sub"], b["del"], b["ins"],
                   100.0 * b["hyp"] / max(b["ref"], 1),
                   100.0 * e / max(b["ref"], 1)))
        return {"bias": bias, "WER": o["WER_official"], "edits": o["edits"],
                "sub": ts, "del": td, "ins": ti,
                "pct_sub": round(100.0 * ts / max(tot, 1), 1),
                "pct_del": round(100.0 * td / max(tot, 1), 1),
                "pct_ins": round(100.0 * ti / max(tot, 1), 1),
                "sum_ok": tot == o["edits"],
                "gen_rate": round(100.0 * n_h / max(n_r, 1), 1),
                "exact": exact, "buckets": buckets}

    bd0 = breakdown(0.0, "0.00")
    bdb = breakdown(float(best_b), best_b) if best_b != "0.00" else bd0

    ok2 = bd0["sum_ok"]
    print()
    log("校验② ins+del+sub vs edits ⇒ %s" % ("✅ 一致" if ok2 else "❌ 不一致"))

    out = REPO / "artifacts/metrics/blank-gov/p164-p150-bias-and-breakdown.json"
    out.write_text(json.dumps({
        "ckpt": CK.name, "epoch": ck.get("epoch"), "rgb_fix": True,
        "n_dev": len(refs),
        "sanity": {"bias0_wer": got0, "expected": REF_WER, "ok": bool(ok1),
                   "sum_ok": bool(ok2)},
        "bias_scan": scan, "best_bias": best_b,
        "breakdown_bias0": bd0, "breakdown_best": bdb,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
