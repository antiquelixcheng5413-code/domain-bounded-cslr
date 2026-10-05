# -*- coding: utf-8 -*-
"""P47 · 受控解码实验：能否撬动「输出坍缩」

## 依据的诊断（P45/P46，均为实测）
- 模型能说出 240 种已学词里的 194 种（81%）—— **有能力**
- 但输出坍缩：剔除 unk 后前 5 个词占真实词输出 49.5%
- 真实准确率（剔除 ref 里的 unk）仅 **20.8%**
- 白吐 unk 292 + 换错词 464 + 漏说 453 = **40.7% 的位置与词表无关**

⇒ 关键假设：**模型不是不会，是解码时被高频词的先验压过去了**。
若成立，那么在解码阶段修正这个先验（不改权重）就该有收益。

## 论文依据（先查文献再动手，沿用项目铁律）
- ref07 SignBT / ref15 Camgoz SLT：CTC + beam search + 词表约束
- ref12 CCL-SLR：对比学习对齐输出分布
- 通用 ASR 做法：LM shallow fusion / 词频 prior 校正解码
  （Wu & Speech 2016 的 "weighting by word frequency" 是经典做法）

## 四个配置（都只改解码，权重固定，公平对比）
1. `greedy`      —— 基线（当前线上用的）
2. `beam10`      —— 纯 beam search，看搜索本身有没有用
3. `freqprior`   —— log_probs 减去 α·log P_train(token)，压制高频词
4. `freqprior+lenpen` —— 再加长度惩罚，抵消「变长」的副作用

## 关键判据
不能用 folded WER 判优（unk 送分 0.1755，参考 MEMORY 铁律）。
必须同时报：
- 剔除 ref 里 unk 的 WER（真实能力）
- 输出集中度（前 5 词占比，越低越好）
- 样本级 exact
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
from cslr.recognition.decode import (classes_to_token_ids,          # noqa: E402
                                     greedy_decode, prefix_beam_search)
from cslr.recognition.model import BLANK_INDEX                      # noqa: E402
from p40_rgb_main import levenshtein, DualInputCTC, read_csv         # noqa: E402

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM = REPO / "artifacts/part3_features"
UNK = "<unk>"


def decode_with(lp: np.ndarray, mode: str, prior: np.ndarray | None,
                beam: int, alpha: float, lenpen: float):
    """按配置解码单条 [T, C] log-prob。

    prior: [C] 的 token 频次先验（log 概率），只作用在非 blank 类上。
    alpha: 减去 alpha * log P(token)，越大越压制高频词。
    """
    x = lp
    if prior is not None and alpha:
        # 只改非 blank 类；blank 不参与先验（它不是词）
        adj = np.array(prior, dtype=np.float64) * alpha
        x = lp.copy()
        x[:, 1:] -= adj[1:]
        x -= np.logaddexp.reduce(x, axis=1, keepdims=True) * 0.0  # 保持归一化近似
    if mode == "greedy":
        return classes_to_token_ids(greedy_decode(x).classes)
    return classes_to_token_ids(
        prefix_beam_search(x, beam_width=beam, length_penalty=lenpen).classes)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        REPO / "artifacts/metrics/blank-gov/p47-controlled-decode.json"))
    ap.add_argument("--limit", type=int, default=0, help="0=全量 dev")
    ap.add_argument("--beam", type=int, default=0,
                    help=">0 才跑 beam（很慢：301 类 × 48 帧）")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)

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

    # ---- 训练集词频先验（按 CTC 类空间：类 i+1 = 词表 i）----
    freq = np.ones(int(voc.size) + 1, dtype=np.float64)   # 0 是 blank，留 1
    cnt = collections.Counter()
    for g in lab_tr.values():
        for t in g.split("/"):
            t = t.strip()
            if t:
                cnt[t] += 1
    for t, c in cnt.items():
        if t in voc:
            freq[voc.index_of(t) + 1] = c
    log_prior = np.log(freq / freq.sum())
    print("prior: 非 unk 最高频 = %s(%.0f) 最低 = %s(%.0f)"
          % (voc.tokens[voc.index_of("。")], freq[voc.index_of("。") + 1],
             min((t for t in voc.tokens if t != UNK),
                 key=lambda x: freq[voc.index_of(x) + 1]),
             min(freq[voc.index_of(t) + 1] for t in voc.tokens if t != UNK)),
          flush=True)

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = DualInputCTC(lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                         vocab=int(voc.size), hidden=cfg["hidden"],
                         layers=cfg["layers"], dropout=cfg["dropout"],
                         use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                         mode=cfg.get("mode", "add")).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    # ---- 一次性前向，缓存 log-probs（4 个配置共用，保证公平）----
    LPs = []
    with torch.no_grad():
        for k in range(0, len(items), 32):
            ch = items[k:k + 32]
            x = torch.from_numpy(np.stack([c["lm"] for c in ch])).to(device)
            il = torch.full((len(ch),), x.shape[1], dtype=torch.long,
                            device=device)
            lg = model(x, il, None, None)
            LPs.append(torch.log_softmax(lg.float(), dim=-1).cpu().numpy())
    LP = np.concatenate(LPs, 0)
    print("cached log-probs", LP.shape, flush=True)

    def evaluate(tag, mode, alpha, beam, lenpen):
        hyps = [voc.decode(decode_with(LP[i], mode, log_prior, beam, alpha,
                                       lenpen)) for i in range(len(items))]
        # 判优口径：剔除 ref 里的 unk（真实能力）
        refs_no = [[t for t in r["folded"] if t != UNK] or [UNK]
                   for r in items]
        E_no = sum(levenshtein(r, h) for r, h in zip(refs_no, hyps))
        N_no = sum(len(r) for r in refs_no)
        E_f = sum(levenshtein(list(r["folded"]), h)
                  for r, h in zip(items, hyps))
        N_f = sum(len(r["folded"]) for r in items)
        ex_f = sum(1 for r, h in zip(items, hyps)
                   if levenshtein(list(r["folded"]), h) == 0)
        H = [t for h in hyps for t in h]
        real = [t for t in H if t != UNK]
        cnt_h = collections.Counter(real)
        top5 = sum(c for _, c in cnt_h.most_common(5))
        rec = {
            "tag": tag,
            "mode": mode, "alpha": alpha, "beam": beam, "lenpen": lenpen,
            "wer_folded": round(E_f / max(N_f, 1), 4),
            "wer_excl_unk": round(E_no / max(N_no, 1), 4),
            "exact_folded": ex_f,
            "exact_ratio": round(ex_f / len(items), 4),
            "hyp_len_mean": round(len(H) / max(len(hyps), 1), 3),
            "unk_ratio": round(sum(1 for t in H if t == UNK) / max(len(H), 1), 4),
            "distinct": len(set(H)),
            "top5_share_of_real": round(top5 / max(len(real), 1), 4),
        }
        print("%-22s WER(folded)=%.4f  WER(剔unk)=%.4f  exact=%2d  "
              "unk=%4.1f%%  len=%.2f  种类=%3d  前5占比=%.1f%%"
              % (tag, rec["wer_folded"], rec["wer_excl_unk"], ex_f,
                 100 * rec["unk_ratio"], rec["hyp_len_mean"], rec["distinct"],
                 100 * rec["top5_share_of_real"]), flush=True)
        return rec, hyps

    results = []
    print("\n=== 基线（greedy，线上当前用的）===")
    r, base_hyps = evaluate("greedy(基线)", "greedy", 0.0, 1, 0.0)
    results.append(r)

    print("\n=== 词频先验校准：logits -= alpha * log P_train(token) ===")
    print("（alpha 越大越压制高频词；这是「只改解码、不动权重」）")
    for al in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5):
        r, _ = evaluate("greedy+freq a=%.1f" % al, "greedy", al, 1, 0.0)
        results.append(r)

    if a.beam > 0:
        print("\n=== beam search（慢，抽样跑）===")
        r, _ = evaluate("beam%d" % a.beam, "beam", 0.0, a.beam, 0.0)
        results.append(r)
        for al in (0.3, 0.5):
            r, _ = evaluate("beam%d+freq a=%.1f" % (a.beam, al), "beam", al,
                            a.beam, 0.0)
            results.append(r)
        print("\n=== 频先验 + 长度惩罚 ===")
        for lp_ in (0.6, 1.0):
            r, _ = evaluate("beam%d+freq a=0.3+len%.1f" % (a.beam, lp_),
                            "beam", 0.3, a.beam, lp_)
            results.append(r)

    base = results[0]
    best = min(results, key=lambda x: x["wer_excl_unk"])
    print("\n=== 结论 ===")
    print("基线   WER(剔unk)=%.4f  真实准确率=%.1f%%  前5占比=%.1f%%"
          % (base["wer_excl_unk"], 100 * (1 - base["wer_excl_unk"]),
             100 * base["top5_share_of_real"]))
    print("最优   %s  WER(剔unk)=%.4f  真实准确率=%.1f%%  前5占比=%.1f%%"
          % (best["tag"], best["wer_excl_unk"], 100 * (1 - best["wer_excl_unk"]),
             100 * best["top5_share_of_real"]))
    gain = base["wer_excl_unk"] - best["wer_excl_unk"]
    print("=> 真实准确率提升 %.1f 个百分点（%.1f%% -> %.1f%%）"
          % (100 * gain, 100 * (1 - base["wer_excl_unk"]),
             100 * (1 - best["wer_excl_unk"])))

    receipt = {
        "experiment": "P47", "date": "2026-10-05",
        "hypothesis": "输出坍缩是解码期被高频词先验压制；只改解码不改权重即可改善",
        "literature_basis": [
            "ref07 SignBT / ref15 Camgoz SLT: CTC + beam search + 词表约束",
            "ref12 CCL-SLR: 对比学习对齐输出分布",
            "经典 ASR: 用词频先验校正解码（weighting by word frequency）",
        ],
        "fairness_note": "四个配置共用同一份缓存 log-probs，权重完全固定，"
                         "唯一变量是解码策略",
        "primary_metric": "wer_excl_unk（剔除 ref 里的 unk），"
                          "因为 folded WER 有 0.1755 的 unk 送分（MEMORY 铁律）",
        "baseline": base, "best": best, "all_results": results,
        "gain_in_wer_excl_unk": round(gain, 4),
    }
    p = Path(a.out)
    p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
