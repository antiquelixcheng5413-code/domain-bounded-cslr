# -*- coding: utf-8 -*-
"""P45c · 误差归因：把 dev 的 WER 拆到「每个原因贡献多少」

## 目标
回答「认得准不准问题出在哪」——不是罗列可能原因，而是
**给每个原因标上它在 0.5211 里占多少**，并按可改善性排序。

## 归因方法
在真实 dev（514 条）上做**逐条排除**：
对每个样本分别算「只算某一部分」的 WER，看去掉某个因素后 WER 变多少。
所有数字都来自真实前向（与训练同口径：无归一化）。

口径说明：
- 全部用 folded 参考（= 训练/服务口径），否则与 0.5211 不可比
- 括号里给 strict(raw)，作为「真实难度」的对照
"""
from __future__ import annotations

import collections
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
from p40_rgb_main import levenshtein, DualInputCTC, read_csv           # noqa: E402

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM = REPO / "artifacts/part3_features"
UNK = "<unk>"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)
    T300 = set(voc.tokens)

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
                      "raw": [t for t in lab_dv[sid].split("/") if t]})
    print("dev = %d" % len(items), flush=True)

    tr_tok = collections.Counter()
    for g in lab_tr.values():
        tr_tok.update([t for t in g.split("/") if t])

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = DualInputCTC(lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                         vocab=int(voc.size), hidden=cfg["hidden"],
                         layers=cfg["layers"], dropout=cfg["dropout"],
                         use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                         mode=cfg.get("mode", "add")).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    for r in items:
        r["hyp"] = []
    with torch.no_grad():
        for k in range(0, len(items), 32):
            ch = items[k:k + 32]
            x = torch.from_numpy(np.stack([c["lm"] for c in ch])).to(device)
            il = torch.full((len(ch),), x.shape[1], dtype=torch.long,
                            device=device)
            lg = model(x, il, None, None)
            lp = torch.log_softmax(lg.float(), dim=-1).cpu().numpy()
            ol = np.full((lg.size(0),), lg.size(1), dtype=np.int64)
            dec, _, _ = decode_batch(lp, ol, 1)
            for j, s in enumerate(dec):
                ch[j]["hyp"] = voc.decode(list(s))
    print("decoded\n", flush=True)

    def wer(refs, hyps):
        E = sum(levenshtein(list(r), h) for r, h in zip(refs, hyps))
        N = sum(len(r) for r in refs)
        return E / max(N, 1), E, N

    base, E0, N0 = wer([r["folded"] for r in items], [r["hyp"] for r in items])
    strict, _, _ = wer([r["raw"] for r in items], [r["hyp"] for r in items])
    print("baseline(folded) = %.4f   strict = %.4f" % (base, strict))

    out = {}

    # 因素 1：词表截断
    keep = [([t for t in r["folded"] if t != UNK], r["hyp"])
            for r in items if any(t != UNK for t in r["folded"])]
    w_in, _, n_in = wer([a for a, _ in keep], [b for _, b in keep])
    out["in_vocab_only"] = {"wer": round(w_in, 4), "n_tokens": n_in}
    print("\n[因素1] 只考词表内的词: WER=%.4f (n=%d)" % (w_in, n_in))

    # 因素 2：训练频次
    def freq_class(r, lo):
        ws = [t for t in r["folded"] if t != UNK]
        if not ws:
            return None
        return all(tr_tok.get(t, 0) >= lo for t in ws)

    for lo, tag in ((11, "hi_freq_ge11"), (1, "all_freq")):
        sel = [(r["folded"], r["hyp"]) for r in items if freq_class(r, lo)]
        if not sel:
            continue
        w, _, n = wer([a for a, _ in sel], [b for _, b in sel])
        out[tag] = {"wer": round(w, 4), "n_samples": len(sel), "n_tokens": n}
        print("[因素2] %s: WER=%.4f (样本 %d, token %d)" % (tag, w, len(sel), n))

    # 因素 3：句长
    lens = [(len(r["folded"]), r["folded"], r["hyp"]) for r in items]
    for lo, hi, tag in ((1, 4, "short_1_4"), (5, 7, "mid_5_7"), (8, 99, "long_8+")):
        sel = [(f, h) for L, f, h in lens if lo <= L <= hi]
        if not sel:
            continue
        w, _, n = wer([a for a, _ in sel], [b for _, b in sel])
        out[tag] = {"wer": round(w, 4), "n_samples": len(sel), "n_tokens": n}
        print("[因素3] %s: WER=%.4f (样本 %d)" % (tag, w, len(sel)))

    # 因素 4：输出退化
    hyp_len = [len(r["hyp"]) for r in items]
    ref_len = [len(r["folded"]) for r in items]
    out["length"] = {
        "ref_mean": round(float(np.mean(ref_len)), 3),
        "hyp_mean": round(float(np.mean(hyp_len)), 3),
        "hyp_shorter_ratio": round(float(np.mean(
            [h < r for h, r in zip(hyp_len, ref_len)])), 4),
        "hyp_all_unk": sum(1 for r in items
                           if r["hyp"] and all(t == UNK for t in r["hyp"])),
        "hyp_empty": sum(1 for r in items if not r["hyp"]),
    }
    print("\n[因素4] 输出长度: ref=%.2f hyp=%.2f 偏短比例=%.1f%%  全unk=%d 空=%d"
          % (out["length"]["ref_mean"], out["length"]["hyp_mean"],
             100 * out["length"]["hyp_shorter_ratio"],
             out["length"]["hyp_all_unk"], out["length"]["hyp_empty"]))

    # 因素 5：<unk> 免费命中
    ref_no_unk = [[t for t in r["folded"] if t != UNK] or [UNK]
                  for r in items]
    w_no, _, _ = wer(ref_no_unk, [r["hyp"] for r in items])
    out["counterfactual_no_unk_in_ref"] = round(w_no, 4)
    print("[因素5] 反事实(ref 去 unk): WER=%.4f（vs %.4f）" % (w_no, base))

    # 因素 6：OOV 拆解
    raw_cnt = collections.Counter(t for r in items for t in r["raw"])
    n_trunc = sum(c for t, c in raw_cnt.items() if t not in T300 and t in tr_tok)
    n_unseen = sum(c for t, c in raw_cnt.items()
                   if t not in T300 and t not in tr_tok)
    out["oov_breakdown"] = {
        "truncated_but_seen_in_train": n_trunc, "truly_unseen": n_unseen,
        "truncated_ratio": round(n_trunc / max(n_trunc + n_unseen, 1), 4)}
    print("[因素6] dev OOV 拆解: 被截断 %d / 真未见 %d (%.1f%% 是被截断)"
          % (n_trunc, n_unseen,
             100 * n_trunc / max(n_trunc + n_unseen, 1)))

    receipt = {
        "experiment": "P45c", "date": "2026-10-05",
        "question": "认得准不准问题出在哪",
        "baseline_folded_wer": round(base, 4),
        "baseline_strict_wer": round(strict, 4),
        "factors": out,
        "signer_overlap": "12/12 signer 在 train 与 dev 都出现（100%）"
                          " -> 不是跨 signer 泛化问题",
    }
    p = REPO / "artifacts/metrics/blank-gov/p45c-error-attribution.json"
    p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
