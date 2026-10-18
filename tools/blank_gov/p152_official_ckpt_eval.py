#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P152：官方 TFNet 权重 -> 在**我们的 dev** 上实测（决定性对照）

为什么这是决定性的
  我们此前只能与论文数字（TFNet dev 42.1）比。现在有了官方权重本身：
    - ckpt 内 bestWerScore = **33.19**（Train.py:295 = validLoader 上的**逐句等权** WER，epoch 37）
    - classifier 输出 3516 ⇒ 是 **CE-CSL**（不是 CSL-Daily）
    - 结构含 motorAttention1-4 + 5 个 classifier ⇒ **TFNet**
  把它的权重直接跑在我们的 dev 帧上，就能把"数据/预处理差异"和"训练差异"分开。

必须对齐的三件事（否则数字没意义）
  1. **RGB**：官方链路最终是真 RGB（P151 实测）⇒ P.RGB_FIX = True
  2. **224**：官方默认分辨率（我们 160 是我们自己的工程取舍）
  3. **official collate**：官方 collate_fn 有补帧（left_pad=6）⇒ P.OFFICIAL_PAD = True

用法
  python tools/blank_gov/p152_official_ckpt_eval.py [n_dev]
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
from p147_beam_decode import prefix_beam_search  # noqa: E402

CK = Path("/mnt/c/Users/su127/Desktop/中文手语识别/官方权重/TFNet-CE-CSL-CSLDaily-32.46.pth")
OFFICIAL_DEV_PER_SENT = 33.19      # ckpt 内 bestWerScore（官方 dev 逐句等权）
OUR_DEV = 55.84                    # P123（VAC，语料级，BGR 输入）
N_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 0
IMG = 224


def log(*a):
    print("[P152 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def main() -> int:
    dev = "cuda"
    log("官方 ckpt = %s（%.0f MB）" % (CK.name, CK.stat().st_size / 1e6))
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    n_cls = wordSetNum + 1
    log("我们的 wordSetNum = %d ⇒ num_classes = %d（官方 ckpt = 3516）" % (wordSetNum, n_cls))
    if n_cls != 3516:
        log("❌ 词表不匹配，无法加载；停止")
        return 1

    model = P.Net.moduleNet(1024, n_cls, "TFNet", torch.device(dev), "CE-CSL", True).to(dev)
    sd = torch.load(str(CK), map_location="cpu", weights_only=False,
                    mmap=True)["moduleNet_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log("载入：missing %d，unexpected %d" % (len(missing), len(unexpected)))
    if len(missing) > 4:
        log("❌ missing 过多：%s" % missing[:6])
        return 1
    if missing:
        log("  missing = %s" % missing[:6])
    model.eval()

    # 🔴 三件必须对齐的事
    P.RGB_FIX = True                 # 官方是真 RGB
    P.OFFICIAL_PAD = True            # 官方 collate 有补帧
    log("对齐：RGB_FIX=True（真RGB）、OFFICIAL_PAD=True（官方collate）、img=%d" % IMG)

    labels = P.read_csv_split("dev.csv")
    _, tf = P.build_transforms(hflip=False, img_size=IMG)
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

    rows = {}
    # ---- greedy ----
    P.BLANK_BIAS = 0.0
    hyps, ntok = [], 0
    for lp, L in zip(cache, lens):
        t = P.greedy_decode(lp.unsqueeze(1), torch.LongTensor([[L]]), idx2word)
        hyps += t
        ntok += sum(len(q) for q in t)
    o = evaluate(refs, hyps)
    rows["greedy"] = {"WER_official": o["WER_official"],
                      "WER_per_sentence": o["WER_per_sentence"],
                      "gen_rate": 100.0 * ntok / max(o["ref_tokens"], 1)}
    # ---- beam-10（官方解码口径）----
    hyps_b, ntok_b = [], 0
    for lp, L in zip(cache, lens):
        ids = prefix_beam_search(lp, 10)
        hyps_b.append([idx2word[j] if 0 <= j < len(idx2word) else "<unk>" for j in ids])
        ntok_b += len(ids)
    ob = evaluate(refs, hyps_b)
    rows["beam10"] = {"WER_official": ob["WER_official"],
                      "WER_per_sentence": ob["WER_per_sentence"],
                      "gen_rate": 100.0 * ntok_b / max(ob["ref_tokens"], 1)}

    print()
    log("──── 官方 TFNet 权重跑我们的 dev（%d 条）────" % len(ds))
    log("  %-9s %10s %14s %9s" % ("解码", "语料级", "逐句等权", "生成率"))
    for k in ("greedy", "beam10"):
        r = rows[k]
        log("  %-9s %10.2f %14.2f %8.1f%%"
            % (k, r["WER_official"], r["WER_per_sentence"], r["gen_rate"]))
    log("  官方自报 bestWerScore（逐句等权，ep37） = %.2f" % OFFICIAL_DEV_PER_SENT)

    got = rows["beam10"]["WER_per_sentence"]
    d = got - OFFICIAL_DEV_PER_SENT
    print()
    log("──── 判定 ────")
    log("  官方权重在我们的帧上 = %.2f（逐句等权）vs 官方自报 %.2f ⇒ Δ %+.2f pp"
        % (got, OFFICIAL_DEV_PER_SENT, d))
    if abs(d) < 2.0:
        log("  ⇒ ✅ 复现成功：我们的帧/预处理与官方**等价**，差距全部来自训练侧")
    elif d > 0:
        log("  ⇒ ⚠️ 我们的数据侧让官方模型变差 %.2f pp ⇒ 数据/预处理有差异" % d)
    else:
        log("  ⇒ ⚠️ 官方模型在我们帧上更好 %.2f pp（可能是 ckpt 非 best 或口径差异）" % -d)
    log("  参考：我们 P123(VAC) 语料级 %.2f / 逐句等权 见上表" % OUR_DEV)

    out = REPO / "artifacts/metrics/blank-gov/p152-official-ckpt-eval.json"
    out.write_text(json.dumps({
        "ckpt": CK.name, "module": "TFNet", "img_size": IMG,
        "rgb_fix": True, "official_pad": True, "n": len(ds),
        "official_bestWerScore_dev_per_sentence": OFFICIAL_DEV_PER_SENT,
        "rows": rows,
        "delta_vs_official_per_sentence": d,
        "note": "官方 ckpt 内 bestWerScore=33.19 是 dev 逐句等权（Train.py:295）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
