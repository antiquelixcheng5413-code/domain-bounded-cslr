#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P143：Model Soup —— 平均已有主配置 ckpt 的权重（**零训练成本**）

动机（零成本还能试的方向）
  超参/正则化主线已走完（P140/P141/P142 全部失败），GPU 空闲。
  手上已有 4 个**同架构、同主配置、仅超参不同**的 ckpt：
    P123 (lr 1e-4)      best 55.84  ← 当前最佳
    P140 (lr 5e-5)      best 57.14
    P141 (lr 1.5e-4)    best 57.00
    P142 (LS 0.01)      best 57.32
  ⇒ 权重平均（uniform soup）有可能比任一单独模型更好。

依据
  【有】Model Soups（Wortsman et al., ICML 2022）：平均多个 fine-tune 的权重
        可提升泛化，零额外训练成本
  【间接】权重平均抑制过拟合（与 SWA 同源，Izmailov et al. 2018）
  ⚠️【无】CSLR / CTC 上 model soup 的直接论文证据

判据（预先写死）
  soup 的 dev WER 比 **P123 单独** 低 ≥ 1.556 pp（噪声基准）⇒ 判活
  否则判"无收益"

已知答案校验
  P123 单独评估必须复现 **55.84%**（否则说明走的代码路径不对，结论作废）
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

CKDIR = REPO / "artifacts/checkpoints"
NAMES = ["p123-official55bf16", "p140-lr5e5-main",
         "p141-lr15e4-main", "p142-ls001-main"]
MAX_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 515
BIASES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

# P123 收据里的 best，用于已知答案校验
EXPECT_123 = 55.84


def log(*a):
    print("[P143 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def load_state(name):
    p = CKDIR / ("%s-best.pt" % name)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert "model_state" in ck, "%s 键名变了：%s" % (name, list(ck)[:8])
    return ck


def main() -> int:
    dev = "cuda"
    labels_dv = P.read_csv_split("dev.csv")
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    log("词表 %d" % wordSetNum)

    cks = {n: load_state(n) for n in NAMES}
    # 🔴 official_pad 从 ckpt config 读（都是主配置 ⇒ 都开）
    pads = {n: bool(ck.get("config", {}).get("official_pad", False))
            for n, ck in cks.items()}
    log("official_pad: %s" % pads)
    P.OFFICIAL_PAD = bool(pads[NAMES[0]])

    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    model.eval()
    _, dv_tf = P.build_transforms(hflip=False, img_size=160)
    ds = P.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    if MAX_DEV:
        ds.items = ds.items[:MAX_DEV]
    ls = torch.nn.LogSoftmax(dim=-1)

    def infer_and_cache():
        """推理一次缓存 logp；返回 (cache, lens, refs)"""
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
                    refs.append(labels_dv[b][1])
        log("  缓存完成 %.0fs，%d 条" % (time.time() - t0, len(cache)))
        return cache, lens, refs

    def decode(cache, lens, bias):
        """与 p78 评估段逐字一致的写法"""
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

    results = {}

    # ---------- 1. 各单独模型 ----------
    log("──── 单独模型（bias=0）────")
    for n in NAMES:
        model.load_state_dict(cks[n]["model_state"], strict=True)
        cache, lens, refs = infer_and_cache()
        hyps, ntok = decode(cache, lens, 0.0)
        o = evaluate(refs, hyps)
        results[n] = {"WER": o["WER_official"], "gen": 100.0 * ntok / max(o["ref_tokens"], 1)}
        log("  %-22s WER %6.2f%%  生成率 %.1f%%" % (n, o["WER_official"], results[n]["gen"]))
        if n == NAMES[0] and MAX_DEV == 515:
            okk = abs(o["WER_official"] - EXPECT_123) < 0.05
            log("  已知答案校验：P123 应 %.2f ⇒ %s" % (EXPECT_123, "✅" if okk else "❌ 作废"))
            if not okk:
                return 1
        del cache
        torch.cuda.empty_cache()

    # ---------- 2. soups ----------
    combos = [
        ("soup_all(4)", NAMES),
        ("soup_3_no123", NAMES[1:]),
        ("soup_123_142", [NAMES[0], NAMES[3]]),
    ]
    log("──── 权重平均（uniform soup）────")
    for label, members in combos:
        sds = [cks[m]["model_state"] for m in members]
        avg = {}
        n_bn = 0
        for k in sds[0]:
            # 🔴 BatchNorm 的 running_mean / running_var / num_batches_tracked
            #    **不能平均**：统计量是各模型在自己数据分布上累积的，
            #    平均后归一化失配 ⇒ 输出崩坏（实测生成率从 82% 掉到 18.5%）
            #    ⇒ 沿用第一个成员（P123，当前最佳）的统计量
            if ("running_" in k) or ("num_batches_tracked" in k):
                avg[k] = sds[0][k].clone()
                n_bn += 1
            else:
                avg[k] = sum(sd[k].float() for sd in sds) / len(sds)
        log("    （BN 统计量沿用 %s，共 %d 个，未参与平均）" % (members[0], n_bn))
        model.load_state_dict(avg, strict=True)
        cache, lens, refs = infer_and_cache()
        hyps, ntok = decode(cache, lens, 0.0)
        o = evaluate(refs, hyps)
        results[label] = {"WER": o["WER_official"],
                          "gen": 100.0 * ntok / max(o["ref_tokens"], 1),
                          "members": members}
        log("  %-22s WER %6.2f%%  生成率 %.1f%%  （%s）"
            % (label, o["WER_official"], results[label]["gen"], "+".join(m[:6] for m in members)))

        # 对最好的 soup 顺便扫一下 bias（缓存已就绪，几乎免费）
        if label == "soup_all(4)":
            log("    bias 扫描（复用缓存）：")
            best_b, best_w = 0.0, o["WER_official"]
            for b in BIASES[1:]:
                h2, n2 = decode(cache, lens, b)
                o2 = evaluate(refs, h2)
                log("      bias %.2f → %.2f%% (Δ %+.2f)"
                    % (b, o2["WER_official"], o2["WER_official"] - o["WER_official"]))
                if o2["WER_official"] < best_w:
                    best_b, best_w = b, o2["WER_official"]
            results[label]["best_bias"] = best_b
            results[label]["WER_with_bias"] = best_w
        del cache
        torch.cuda.empty_cache()

    # ---------- 3. 判定 ----------
    base = results[NAMES[0]]["WER"]
    print()
    log("──── 判定 ────")
    log("基线 P123 单独 = %.2f%%" % base)
    for label, _ in combos:
        w = results[label]["WER"]
        d = w - base
        verdict = ("判活：改善 %.2f pp ≥ 1.556" % (-d) if d <= -1.556
                   else ("弱证据：改善 %.2f pp" % (-d) if d < 0
                         else "无收益：恶化 %.2f pp" % d))
        log("  %-22s %.2f%%  Δ %+.2f  ⇒ %s" % (label, w, d, verdict))
        if "WER_with_bias" in results[label]:
            log("      +bias %.2f → %.2f%%" % (results[label]["best_bias"],
                                               results[label]["WER_with_bias"]))

    out = REPO / "artifacts/metrics/blank-gov/p143-soup.json"
    out.write_text(json.dumps({
        "max_dev": MAX_DEV, "results": results,
        "baseline_P123": base, "noise_baseline_pp": 1.556,
        "note": "uniform weight averaging of main-config ckpts (zero training cost)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
