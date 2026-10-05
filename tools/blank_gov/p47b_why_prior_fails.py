# -*- coding: utf-8 -*-
"""P47b · 为什么词频先验校准没用？（坍缩的机制澄清）

## P47 的结果（实测）
词频先验校准（logits -= alpha * log P_train）**单调变差**：
  alpha=0.0  WER(剔unk)=0.7919  真实准确率 20.8%  前5占比 49.5%
  alpha=0.1  WER(剔unk)=0.8171                20.8%      49.6%
  alpha=0.3  WER(剔unk)=0.8710                17.6%      49.2%
  alpha=0.5  WER(剔unk)=0.9381                13.7%      49.5%
  alpha=1.0  WER(剔unk)=1.1024                 3.5%      49.4%
**前5占比几乎不动（49.5% -> 49.1%）** —— 坍缩没有被撬动。

## 两种可能解释，必须区分
H1「解码期无能为力」：坍缩已烧进权重，log-prob 本身就错，改解码没用
H2「先验方向错了」：要压制高频不该用「减 log P」，
   因为 CTC 的 log-prob 已经包含了「这个词在这里」的证据；
   真正该做的是**对比式打分**（用 hyp 分数减去无条件 baseline）

若 H1 成立，加权再训练才是唯一出路。
若 H2 成立，可能存在不改权重的解法。

## 本脚本的判据实验
逐帧检查模型 log-prob 的真实结构：
1. 真实词位置上，模型给正确词的概率 vs 给最高频词（`。`）的概率
2. blank 占多少（若 blank 极高，说明模型在「不敢输出」而非「选错」）
3. 非 blank 帧里，argmax 是 blank 的比例
4. top-5 候选里包含正确词的比例（teacher-forcing 上界）
5. 若 #4 高而 #1 低 → 模型「知道但排序错」，可解码修复
   若 #4 也低 → 特征里根本没有这个信息，只能重训

## 这是 P20/P19 那类探针的翻版，但用真实 dev 特征
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
from p40_rgb_main import DualInputCTC, read_csv                        # noqa: E402

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM = REPO / "artifacts/part3_features"
UNK = "<unk>"


def main() -> None:
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
                      "folded": voc.decode(list(ids))})

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = DualInputCTC(lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                         vocab=int(voc.size), hidden=cfg["hidden"],
                         layers=cfg["layers"], dropout=cfg["dropout"],
                         use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                         mode=cfg.get("mode", "add")).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    LP = []
    with torch.no_grad():
        for k in range(0, len(items), 32):
            ch = items[k:k + 32]
            x = torch.from_numpy(np.stack([c["lm"] for c in ch])).to(device)
            il = torch.full((len(ch),), x.shape[1], dtype=torch.long,
                            device=device)
            lg = model(x, il, None, None)
            LP.append(torch.log_softmax(lg.float(), dim=-1).cpu().numpy())
    LP = np.concatenate(LP, 0)                     # (N, 48, C)
    N, T, C = LP.shape
    print("log-probs = %s" % (LP.shape,), flush=True)

    # ---- 1) blank 占比 ----
    blank = LP[:, :, 0]
    nb = (blank > -0.1).mean()
    print("\n=== 1) blank 结构 ===")
    print("  blank 概率 > 0.9 的帧占比 = %.1f%%" % (100 * nb))
    print("  平均 blank 概率 = %.3f" % blank.mean())
    argmax_blank = (LP.argmax(2) == 0)
    print("  argmax == blank 的帧占比 = %.1f%%" % (100 * argmax_blank.mean()))
    print("  非 blank 帧数/句 = %.1f" % ((~argmax_blank).sum(1).mean()))

    # ---- 2) top-k 里是否含正确词（teacher-forcing 上界）----
    # 用 CTC 强制对齐做不了（无帧级标注），改用「整句内是否至少一帧
    # 的 top-k 覆盖了某个正确词」这个弱上界
    top1 = LP.argmax(2)
    top5 = np.argsort(-LP, axis=2)[:, :, :5]
    print("\n=== 2) 帧级 top-k 覆盖（弱上界，参考含 OOV 故偏保守）===")
    hit1 = hit5 = tot = 0
    for i, it in enumerate(items):
        gold = {voc.index_of(t) + 1 for t in it["folded"] if t != UNK}
        if not gold:
            continue
        for t in range(T):
            if top1[i, t] in gold:
                hit1 += 1
            if gold.intersection(top5[i, t].tolist()):
                hit5 += 1
            tot += 1
    print("  帧级：argmax 命中某正确词 %.1f%%" % (100 * hit1 / tot))
    print("  帧级：top-5   命中某正确词 %.1f%%" % (100 * hit5 / tot))

    # ---- 3) 非 blank 帧里，模型在选什么 ----
    print("\n=== 3) 非 blank 帧的 argmax 分布 ===")
    cnt = collections.Counter()
    for i in range(N):
        for t in np.where(top1[i] != 0)[0]:
            cnt[top1[i, t]] += 1
    tot_nb = sum(cnt.values())
    print("  非 blank 帧总数 = %d" % tot_nb)
    for cid, c in cnt.most_common(10):
        tok = voc.tokens[cid - 1] if cid - 1 < voc.size else "?"
        print("    %-8s %5d  %5.1f%%" % (tok, c, 100 * c / tot_nb))

    # ---- 4) 关键：模型给「正确词」和给「最高频词」的分数差 ----
    # 在 ref 非 unk 的样本上，看正确词类的平均 log-prob 排名
    print("\n=== 4) 正确词的 log-prob 排名分布（能否靠重排救回）===")
    ranks = []
    for i, it in enumerate(items):
        gold = [voc.index_of(t) + 1 for t in it["folded"] if t != UNK]
        if not gold:
            continue
        for t in range(T):
            if top1[i, t] == 0:            # 只看模型想输出的时候
                continue
            order = np.argsort(-LP[i, t])
            pos = {c: r for r, c in enumerate(order)}
            for g in gold:
                ranks.append(pos.get(g, 999))
    ranks = np.array(ranks)
    if len(ranks):
        print("  样本数 = %d" % len(ranks))
        for thr in (0, 1, 3, 5, 10, 50):
            print("  正确词在 top-%d 内的比例 = %.1f%%"
                  % (thr + 1, 100 * (ranks <= thr).mean()))

    # ---- 5) 逐词看：模型给过的高频词，其 log-prob 是否真的最高 ----
    print("\n=== 5) 高频词的分数是否「真的高」还是「只是先验」===")
    cnt_tr = collections.Counter()
    for g in lab_tr.values():
        for t in g.split("/"):
            t = t.strip()
            if t:
                cnt_tr[t] += 1
    for w in ("。", "我", "你", "？"):
        if w not in voc:
            continue
        cid = voc.index_of(w) + 1
        print("  %-4s class=%3d 平均logprob=%.3f 平均概率=%.3f train频次=%d"
              % (w, cid, LP[:, :, cid].mean(),
                 np.exp(LP[:, :, cid]).mean(), cnt_tr.get(w, 0)))
    # 对比：所有非 blank 类的平均概率
    others = np.exp(LP[:, :, 1:]).mean()
    print("  全部非 blank 类平均概率 = %.4f" % others)

    receipt = {
        "experiment": "P47b", "date": "2026-10-05",
        "question": "为什么词频先验校准没用",
        "blank": {
            "frames_with_blank_gt_0.9": round(float(nb), 4),
            "mean_blank_prob": round(float(blank.mean()), 4),
            "argmax_blank_ratio": round(float(argmax_blank.mean()), 4),
            "non_blank_frames_per_sent": round(float((~argmax_blank).sum(1).mean()), 2),
        },
        "frame_topk_weak_upper_bound": {
            "top1": round(hit1 / max(tot, 1), 4),
            "top5": round(hit5 / max(tot, 1), 4),
        },
        "gold_word_rank_on_nonblank_frames": {
            "n": int(len(ranks)),
            "in_top1": round(float((ranks <= 0).mean()), 4),
            "in_top3": round(float((ranks <= 2).mean()), 4),
            "in_top10": round(float((ranks <= 9).mean()), 4),
            "in_top50": round(float((ranks <= 49).mean()), 4),
        } if len(ranks) else None,
        "nonblank_argmax_top10": [
            {"token": voc.tokens[c - 1] if 1 <= c <= voc.size else str(c),
             "count": n, "ratio": round(n / max(tot_nb, 1), 4)}
            for c, n in cnt.most_common(10)],
        "high_freq_mean_prob": {
            w: round(float(np.exp(LP[:, :, voc.index_of(w) + 1]).mean()), 5)
            for w in ("。", "我", "你", "？") if w in voc},
        "all_nonblank_mean_prob": round(float(others), 5),
        "p47_result": "alpha 扫描 0->1.5，WER(剔unk) 0.7919 -> 1.2776 单调变差；"
                      "前5占比仅 49.5% -> 50.2%，坍缩未被撬动",
    }
    p = REPO / "artifacts/metrics/blank-gov/p47b-why-prior-fails.json"
    p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
