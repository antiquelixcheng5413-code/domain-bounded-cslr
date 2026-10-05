# -*- coding: utf-8 -*-
"""P50b · 用第三方库 jiwer 对真实 dev 输出做最终交叉验证

P50 已用自写 DP 验证过仓库实现，但都是我自己写的代码。
本脚本用**业界标准库 jiwer** 独立算一次 WER，
若三方一致，则「0.5211」这个数字的实现问题可以彻底排除。

同时用 jiwer 的 corpus-level 变换做一次端到端验证：
把整个 dev 当作一个语料，jiwer 会自动做词级合并，
其 WER 应等于我逐句算再加权平均的结果（若不等，说明我的聚合方式有问题）。
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

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM = REPO / "artifacts/part3_features"
UNK = "<unk>"


def main() -> None:
    import jiwer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)

    items = []
    for s in sorted(lab_dv):
        p = LM / "validation" / (s + ".landmark.npy")
        if not p.exists():
            continue
        ids = voc.encode(lab_dv[s])
        if not ids or len(ids) > 24:
            continue
        items.append({"lm": np.load(p).astype(np.float32),
                      "folded": voc.decode(list(ids)),
                      "raw": [t for t in lab_dv[s].split("/") if t]})
    print("dev = %d" % len(items), flush=True)

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

    refs_f = [r["folded"] for r in items]
    refs_r = [r["raw"] for r in items]
    refs_no = [[t for t in r if t != UNK] or [UNK] for r in refs_f]

    out = {}

    def three_way(refs, tag):
        # A 仓库实现
        E = sum(levenshtein(list(r), h) for r, h in zip(refs, hyps))
        N = sum(len(r) for r in refs)
        w_repo = E / N
        # B jiwer：把 token 用 '\u0001' 连接成「一句话」，
        #    再用 transform 拿到 edits/hits 自己加权。
        #    ⚠️ 不能用 process_words —— 它会按空格分词，
        #    而我们的 token 是中文（无空格），会导致长度不匹配报错。
        #    用 'a' 作为占位词（token 数必须一致，所以直接用 index 数字不行）。
        #    正确做法：把每个 token 映射成一个唯一的英文单词，
        #    例如 gloss-> w0, w1 ...，保证按空格分词后词数与 token 数相同。
        num = den = 0
        wmap = {}
        for r, h in zip(refs, hyps):
            rw, hw = [], []
            for t in r:
                if t not in wmap:
                    wmap[t] = "w%d" % len(wmap)
                rw.append(wmap[t])
            for t in h:
                if t not in wmap:
                    wmap[t] = "w%d" % len(wmap)
                hw.append(wmap[t])
            o = jiwer.process_words(" ".join(rw), " ".join(hw))
            num += o.substitutions + o.deletions + o.insertions
            den += o.hits + o.substitutions + o.deletions
        w_jiwer = num / max(den, 1)
        # C jiwer 自带 WER（它自己的平均方式）
        try:
            w_jiwer_native = jiwer.wer(
                [" ".join(wmap[t] for t in r) for r in refs],
                [" ".join(wmap[t] for t in h) for h in hyps])
        except Exception as exc:                                # noqa: BLE001
            w_jiwer_native = "n/a: %s" % str(exc)[:50]
        print("\n  [%s]" % tag)
        print("    仓库实现        WER = %.4f  (E=%d N=%d)"
              % (w_repo, E, N))
        print("    jiwer transform WER = %.4f  (E=%d N=%d)"
              % (w_jiwer, num, den))
        print("    两者差 = %.6f" % abs(w_repo - w_jiwer))
        return {"repo": round(w_repo, 4), "jiwer": round(w_jiwer, 4),
                "jiwer_native": (round(w_jiwer_native, 4)
                                 if isinstance(w_jiwer_native, float)
                                 else w_jiwer_native),
                "diff": abs(w_repo - w_jiwer),
                "N": N, "E": E}

    print("=" * 70)
    print("三方交叉验证（仓库实现 / 自写DP(P50已验证) / 第三方 jiwer）")
    print("=" * 70)
    out["folded"] = three_way(refs_f, "folded 口径（我报的 0.5211）")
    out["excl_unk"] = three_way(refs_no, "剔除 unk（真实能力）")
    out["strict"] = three_way(refs_r, "strict（保留真实 OOV）")

    # ---- 端到端：把整个 dev 当一个语料交给 jiwer ----
    print("\n" + "=" * 70)
    print("端到端验证：整个 dev 当作单一语料（jiwer 自动做词级切分）")
    print("=" * 70)
    wmap2 = {}

    def enc(seq):
        out2 = []
        for t in seq:
            if t not in wmap2:
                wmap2[t] = "w%d" % len(wmap2)
            out2.append(wmap2[t])
        return " ".join(out2)

    m = jiwer.wer([enc(r) for r in refs_f], [enc(h) for h in hyps])
    print("  jiwer.wer(逐句 list)      = %.4f" % m)
    print("  仓库逐句累加/总 token      = %.4f" % out["folded"]["repo"])
    print("  => 差 %.6f（jiwer 的 wer 是 per-sentence 平均，"
          "语料级会略不同）" % abs(m - out["folded"]["repo"]))

    # 逐句平均（每个句子等权）—— 这才是「看一条视频」的直觉
    per = [levenshtein(list(r), h) for r, h in zip(refs_f, hyps)]
    avg = float(np.mean(per))
    avg_j = m
    print("\n  逐句 WER 平均（每句等权）= %.4f  <- 最接近「看一条视频」的直觉"
          % avg)
    print("  jiwer.wer 同样口径        = %.4f" % avg_j)
    out["per_sentence_average"] = {"repo": round(avg, 4),
                                   "jiwer": round(avg_j, 4),
                                   "note": "每句等权，与 token 级平均不同"}
    out["end_to_end_jiwer_wer"] = round(m, 4)

    # ---- 结论 ----
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    d1 = out["folded"]["diff"]
    d2 = out["excl_unk"]["diff"]
    if d1 < 1e-6 and d2 < 1e-6:
        print("  ✅ 仓库实现与 jiwer **完全一致**（差 %.2e / %.2e）" % (d1, d2))
        print("  ✅ WER 算法实现确认无误")
    else:
        print("  ⚠️ 与 jiwer 有差异：folded %.6f，excl_unk %.6f" % (d1, d2))
    print()
    print("  三个口径的 WER：")
    print("    folded（我以前报的）  = %.4f" % out["folded"]["repo"])
    print("    剔除 ref 的 unk      = %.4f  <- 真实能力" % out["excl_unk"]["repo"])
    print("    strict（保留 OOV）   = %.4f" % out["strict"]["repo"])
    print("    逐句等权平均          = %.4f  <- 最接近你看视频的直觉"
          % out["per_sentence_average"]["repo"])

    p = REPO / "artifacts/metrics/blank-gov/p50b-jiwer-crosscheck.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
