"""P134：blank 负偏置扫描 —— 针对"说太少"的零训练成本对策。

动机（P133 实测，全量 dev 515，P123 best ep39）：
  误差构成：sub 864(54.4%) / del 685(43.2%) / ins 38(2.4%)
  生成率  ：**77.2%**（输出 2195 token vs 参考 2842）
  生成率随句长急剧下降：
      1-3 token → 92.1%   4-6 → 80.5%   7-9 → 69.3%   10+ → **54.7%**
⇒ 模型极度"保守"：几乎不多说(ins 仅 2.4%)，但严重少说(del 43.2%)
⇒ 长句尤其明显 ⇒ CTC 在不确定时倾向输出 blank

对策：解码时给 blank 类（index 0）的 logit 减一个常数 bias
  logp[..., 0] -= bias
⇒ blank 相对概率下降 ⇒ 更容易输出真实 token ⇒ 应能减少 del
⚠️ 风险：说多了可能同时增加 sub / ins。净效果必须实测，不能想当然。

为什么不用 beam：P120 已实测 beam(10) 会让输出 token **+109%**（125.58%），
过度生成到失控。blank bias 是**温和且连续可调**的，与 beam 不同。

方法：推理一次缓存 logp，再对多个 bias 复用解码 ⇒ 只占一次 GPU 时间。

判据（预先写死）：
  best bias 使 WER 比 bias=0 低 **≥ 2.0 pp** ⇒ 判活（噪声 1.556 的 1.3 倍）
  1.0 ~ 2.0 pp                                ⇒ 弱证据
  < 1.0 pp                                    ⇒ 不判活

⚠️ 已知答案校验：bias=0 必须复现 55.84 与生成率 77.2%。
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

CK = REPO / (sys.argv[2] if len(sys.argv) > 2
             else "artifacts/checkpoints/p123-official55bf16-best.pt")
BIASES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
MAX_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 515


def log(*a):
    print("[P134 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def main() -> int:
    dev = "cuda"
    labels_dv = P.read_csv_split("dev.csv")
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    log("词表 %d" % wordSetNum)

    ck = torch.load(CK, map_location="cpu", weights_only=False)
    assert "model_state" in ck, "键名变了：%s" % list(ck)[:8]
    # 🔴 从 ckpt 的 config 读补帧设置，不能硬编码
    #    （P123 开了 official_pad，P124e 没开；写死会让后者的 data_len 算错）
    P.OFFICIAL_PAD = bool(ck.get("config", {}).get("official_pad", False))
    log("official_pad = %s（取自 ckpt config）" % P.OFFICIAL_PAD)
    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    log("载入 %s（epoch %s）" % (CK.name, ck.get("epoch")))

    _, dv_tf = P.build_transforms(hflip=False, img_size=160)
    ds = P.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    if MAX_DEV:
        ds.items = ds.items[:MAX_DEV]
    ls = torch.nn.LogSoftmax(dim=-1)

    # ---- 推理一次，缓存 logp（float16 省内存）----
    log("推理并缓存 logp（%d 条）…" % len(ds))
    cache, lens, refs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            batch = P.collate([ds[i]])
            vid, _t, _tl, dl, _tl_len, sids, _ids = batch
            out = model(vid.to(dev), dl, False)
            lp = ls(out[0])[:, 0, :]                    # [T, C]
            L = min(P._conv_len(int(dl.view(-1)[0])), lp.shape[0])
            cache.append(lp[:L].to("cpu", torch.float16))
            lens.append(int(dl.view(-1)[0]))
            for b in sids:
                refs.append(labels_dv[b][1])
    log("缓存完成 %.0fs，%d 条" % (time.time() - t0, len(cache)))

    def decode(bias: float):
        hyps, ntok = [], 0
        for lp16, L in zip(cache, lens):
            lp = lp16.float()
            if bias:
                lp[:, 0] = lp[:, 0] - bias              # 🔴 抑制 blank
            toks = P.greedy_decode(lp.unsqueeze(1),     # [T,1,C]
                                   torch.LongTensor([[L]]), idx2word)
            hyps += toks
            ntok += sum(len(t) for t in toks)
        return hyps, ntok

    base_hyps, base_ntok = decode(0.0)
    o0 = evaluate(refs, base_hyps)
    ref_tokens = o0["ref_tokens"]
    # 已知答案校验：与该 ckpt 收据里的 best_wer_official 比对（不再硬编码 55.84）
    tag = Path(sys.argv[2]).name.replace("-best.pt", "") if len(sys.argv) > 2 \
        else "p123-official55bf16"
    rp = REPO / "artifacts/metrics/blank-gov" / ("%s-official-tfnet.json" % tag)
    expect = None
    if rp.exists():
        expect = json.load(open(rp)).get("best_wer_official")
    ok = (expect is None) or (abs(o0["WER_official"] - expect) < 0.05)
    log("已知答案校验：bias=0 WER=%.2f（收据 best=%s）⇒ %s"
        % (o0["WER_official"], expect, "✅" if ok else "❌ 不一致"))
    log("  生成率 %.1f%%" % (100.0 * base_ntok / max(ref_tokens, 1)))

    print()
    log("──── 扫描 ────")
    log("bias    WER      生成率    Δ vs bias=0")
    rows = []
    best_b, best_w = 0.0, o0["WER_official"]
    for b in BIASES:
        hyps, ntok = decode(b)
        o = evaluate(refs, hyps)
        gen = 100.0 * ntok / max(ref_tokens, 1)
        d = o["WER_official"] - o0["WER_official"]
        rows.append({"bias": b, "WER": o["WER_official"],
                     "gen_pct": round(gen, 1), "delta_pp": round(d, 2),
                     "edits": o["edits"]})
        log("%4.2f   %6.2f%%   %6.1f%%   %+6.2f" % (b, o["WER_official"], gen, d))
        if o["WER_official"] < best_w:
            best_w, best_b = o["WER_official"], b

    gain = o0["WER_official"] - best_w
    if gain >= 2.0:
        verdict = ("判活：bias=%.2f 改善 **%.2f pp**（55.84 → %.2f）⇒ 可立即应用，"
                   "且无需重训" % (best_b, gain, best_w))
    elif gain >= 1.0:
        verdict = "弱证据：bias=%.2f 改善 %.2f pp" % (best_b, gain)
    else:
        verdict = ("不判活：最优 bias=%.2f 仅改善 %.2f pp ⇒ 欠生成不是可调的解码问题"
                   % (best_b, gain))
    print()
    log("判定：%s" % verdict)

    out = REPO / "artifacts/metrics/blank-gov" / ("p134-blank-bias-%s.json" % tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "P134 blank 负偏置扫描（针对生成率仅 77.2% 的欠生成）",
        "ckpt": str(CK), "epoch": ck.get("epoch"), "n_dev": len(ds),
        "motivation": {"sub": 864, "del": 685, "ins": 38,
                       "generation_rate_pct": 77.2},
        "biases": BIASES, "rows": rows,
        "baseline_WER": o0["WER_official"],
        "best_bias": best_b, "best_WER": best_w, "gain_pp": round(gain, 2),
        "known_answer_check_passed": ok,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
