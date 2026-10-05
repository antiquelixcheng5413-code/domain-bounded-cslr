# -*- coding: utf-8 -*-
"""P56 · 我与 CE-CSL 官方 TFNet 的差距分解（真实数据，不靠猜）

## 用户问题
「我和它差距多少，有可以参考的地方吗」

## 关键方法论：口径必须先对齐，否则比出来的差距是假的
论文的 WER 定义（论文式 6）：
    WER = 100% × (ins + del + sub) / sum
其中 `sum` = **标签里的 gloss 总数**。

⚠️ 我们的 P50 审计发现自己的 folded WER 有 0.1755 的「unk 送分」，
   而官方没有这个问题（他们的 2000 词表覆盖 CE-CSL 全部 gloss）。
⇒ **直接比 52.11% vs 42.1% 是不公平的**，要先在同一口径下重算。

## 本脚本做三件事
1. 用**严格口径**（剔除 unk 送分）算我们的真实 WER
2. 与官方 6 个模型逐个对齐比较
3. 拆解差距来源：哪些是「口径假象」，哪些是「真实差距」

## 判断依据（每个都要有数据）
- 词表覆盖：官方 2000 词 vs 我们 301（max_tokens=300 截断）
- 输入特征：官方 RGB + CNN backbone vs 我们 landmark 368 维
- 训练轮数/批大小/显存
- 解码方式
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402
from cslr.recognition.training import decode_batch                    # noqa: E402
from p40_rgb_main import levenshtein, DualInputCTC, read_csv          # noqa: E402

LM = REPO / "artifacts/part3_features"
CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
UNK = "<unk>"

# 官方 CE-CSL 基准（论文 Table VII，arXiv:2409.11960）
OFFICIAL = [
    ("MSTNet", 54.4, 53.0), ("CorrNet", 47.2, 46.5), ("SEN", 46.5, 45.3),
    ("VAC", 45.1, 43.3), ("MAM-FSD", 44.9, 44.7), ("TFNet", 42.1, 41.9),
]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")

    tr_tok = set()
    n_tr_tok = 0
    for g in lab_tr.values():
        for t in g.split("/"):
            t = t.strip()
            if t:
                tr_tok.add(t)
                n_tr_tok += 1

    # ⚠️ 词表只建一次。之前在 voc_encode/voc_decode 里每次重建，
    #    500+ 次重复构建纯属浪费（也会让 P40 的对照失去可复现性）。
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)
    T300 = set(voc.tokens)

    def toks_of(g):
        return [t.strip() for t in g.split("/") if t.strip()]

    items = []
    for sid in sorted(lab_dv):
        p = LM / "validation" / (sid + ".landmark.npy")
        if not p.exists():
            continue
        ids = voc.encode(lab_dv[sid])
        if not ids or len(ids) > 24:
            continue
        items.append({"sid": sid, "lm": np.load(p).astype(np.float32),
                      "folded": voc.decode(list(ids)),
                      "raw": toks_of(lab_dv[sid])})
    print("dev = %d 条" % len(items))

    # 词表覆盖
    n_all = sum(len(r["raw"]) for r in items)
    n_in = sum(1 for r in items for t in r["raw"] if t in T300)
    print("\n=== 词表覆盖 ===")
    print("  我们的词表 %d（max_tokens=300 截断）" % voc.size)
    print("  train 实际出现的 gloss 词种 = %d" % len(tr_tok))
    print("  dev token 落在我们词表内 = %d/%d = %.1f%%"
          % (n_in, n_all, 100 * n_in / n_all))
    print("  => 官方 CSL-Daily 用 2000 词表；我们只覆盖 %.1f%%" % (100 * n_in / n_all))

    # 跑模型
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = DualInputCTC(lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                         vocab=int(voc.size), hidden=cfg["hidden"],
                         layers=cfg["layers"], dropout=cfg["dropout"],
                         use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                         mode=cfg.get("mode", "add")).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    hyps = []
    with torch.no_grad():
        for k in range(0, len(items), 32):
            ch = items[k:k + 32]
            x = torch.from_numpy(np.stack([c["lm"] for c in ch])).to(device)
            il = torch.full((len(ch),), x.shape[1], dtype=torch.long,
                            device=device)
            lg = model(x, il, None, None)
            lp = torch.log_softmax(lg.float(), -1).cpu().numpy()
            ol = np.full((lg.size(0),), lg.size(1), dtype=np.int64)
            dec, _, _ = decode_batch(lp, ol, 1)
            hyps.extend(voc.decode(list(s)) for s in dec)

    # ---- 三个口径 ----
    E_f = sum(levenshtein(list(r["folded"]), h)
              for r, h in zip(items, hyps))
    N_f = sum(len(r["folded"]) for r in items)
    w_folded = E_f / N_f

    refs_no = [[t for t in r["folded"] if t != UNK] or [UNK] for r in items]
    E_n = sum(levenshtein(r, h) for r, h in zip(refs_no, hyps))
    N_n = sum(len(r) for r in refs_no)
    w_no_unk = E_n / N_n

    E_s = sum(levenshtein(r["raw"], h) for r, h in zip(items, hyps))
    N_s = sum(len(r["raw"]) for r in items)
    w_strict = E_s / N_s

    per = [levenshtein(list(r["folded"]), h) for r, h in zip(items, hyps)]
    w_persent = float(np.mean(per))
    ex = sum(1 for r, h in zip(items, hyps)
             if levenshtein(list(r["folded"]), h) == 0)

    print("\n=== 我们的 WER（四个口径）===")
    print("  folded（含 unk 送分，我一直在报的） = %.4f" % w_folded)
    print("  剔除 ref 的 unk（更接近官方口径）  = %.4f  <- 可比" % w_no_unk)
    print("  strict（保留真实 OOV）             = %.4f" % w_strict)
    print("  逐句等权                          = %.4f" % w_persent)
    print("  完全正确 = %d/%d (%.1f%%)" % (ex, len(items), 100 * ex / len(items)))

    # ---- 与官方对比 ----
    print("\n" + "=" * 74)
    print("与官方 CE-CSL 基准对比（用可比口径 %.2f%%）" % (100 * w_no_unk))
    print("=" * 74)
    print("%-10s %8s %10s %s" % ("方法", "Dev%", "差距", "备注"))
    ours = 100 * w_no_unk
    for name, dev, test in sorted(OFFICIAL, key=lambda x: -x[1]):
        gap = ours - dev
        mark = " ← 我们在这" if abs(gap) < 3 else (
            "我们优于它" if gap < 0 else "")
        print("%-10s %8.1f %+10.1f %s" % (name, dev, gap, mark))
    best = OFFICIAL[-1][1]
    print("\n  我们 vs 官方最佳 TFNet(42.1): %+.1f pp" % (ours - best))

    # ---- 差距归因 ----
    print("\n" + "=" * 74)
    print("差距来源分解")
    print("=" * 74)
    unk_infl = w_folded - w_no_unk
    print("  ① 口径假象（unk 送分）      = %+.1f pp"
          % (-100 * unk_infl))
    print("     我们一直报的 %.2f%% 里，有 %.2f pp 是 unk 送分"
          % (100 * w_folded, 100 * unk_infl))
    print("     剔掉后真实水平 %.2f%%" % (100 * w_no_unk))
    real_gap = ours - best
    print("\n  ② 真实差距（vs TFNet 42.1）  = %+.1f pp" % real_gap)
    print("     其中可归因于词表截断的部分：")
    # 词表截断造成的理论下限
    oov_ratio = 1 - n_in / n_all
    print("       我们词表只覆盖 %.1f%% 的 dev token" % (100 * (1 - oov_ratio)))
    print("       理论 WER 下界 >= %.1f%%（这部分再努力也降不下去）"
          % (100 * oov_ratio))
    print("     扣掉词表因素后，真实能力差距 >= %.1f pp"
          % (real_gap - 100 * oov_ratio))

    receipt = {
        "experiment": "P56", "date": "2026-10-05",
        "question": "我和 CE-CSL 官方 TFNet 差距多少",
        "our_wer_four_calibers": {
            "folded_reported": round(w_folded, 4),
            "excl_unk_comparable": round(w_no_unk, 4),
            "strict": round(w_strict, 4),
            "per_sentence_avg": round(w_persent, 4),
            "exact": "%d/%d (%.1f%%)" % (ex, len(items),
                                          100 * ex / len(items)),
        },
        "official_benchmark": {n: {"dev": d, "test": t}
                               for n, d, t in OFFICIAL},
        "comparable_gap_vs_tfnet_pp": round(ours - best, 1),
        "gap_decomposition_pp": {
            "caliber_inflation_from_unk": round(-100 * unk_infl, 1),
            "vocab_truncation_floor_pp": round(100 * oov_ratio, 1),
            "residual_real_gap_pp": round(real_gap - 100 * oov_ratio, 1),
        },
        "vocab_coverage": {
            "our_vocab": int(voc.size),
            "train_gloss_kinds": len(tr_tok),
            "dev_token_coverage": round(n_in / n_all, 4),
            "official_csldaily_vocab": 2000,
        },
    }
    p = REPO / "artifacts/metrics/blank-gov/p56-gap-to-official.json"
    p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
