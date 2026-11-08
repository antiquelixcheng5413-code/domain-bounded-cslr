#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P197：CTC 前缀 beam search + N-best + 词级 bigram LM 重排（零训练、零新参数）。

动机（来自误差结构）：
  - 替换错误占 67.5%，典型如 今天→现在、设计→计划、？→。
  - 这类错误在声学上"够像"，greedy 单条路径无法纠正；
    但 train 里 4973 句日常手语的自然搭配可以给出句级先验。
做法：
  1) 用前缀 beam search 出 N-best（拿同一份 log_probs，不重训）
  2) 词级 bigram LM（train 标注统计，add-α 平滑 + 与 unigram 插值）
  3) 重排打分 = logP_ctc + λ·logP_lm（λ 在 dev 网格上选）
判优口径：**只用 official_wer.evaluate（官方 corpus 级）**，与 34.34 基线可比。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402

P78.OFFICIAL_PAD = True
P78.RGB_FIX = True
P78.BLANK_BIAS = 0.0
NEG = -1e30


def _lse(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def ctc_prefix_beam(logp: np.ndarray, beam: int, topk: int, nbest: int,
                    blank: int = 0):
    """标准 CTC 前缀 beam search（log 域）。

    logp: [T, C] 的 log-prob。返回 [(tokens, logP_ctc), ...] 长度 ≤ nbest。
    """
    T = logp.shape[0]
    if T == 0:
        return [((), 0.0)]
    k = min(topk, logp.shape[1])
    if k < logp.shape[1]:
        top_idx = np.argpartition(-logp, k - 1, axis=1)[:, :k]   # [T, k]
    else:
        top_idx = np.argsort(-logp, axis=1)
    beams = {(): (0.0, NEG)}                             # pre -> (p_b, p_nb)
    for t in range(T):
        lp = logp[t]
        new = {}
        for pre, (pb, pnb) in beams.items():
            tot = _lse(pb, pnb)
            for c in top_idx[t]:
                c = int(c)
                v = float(lp[c])
                if c == blank:
                    b0, n0 = new.get(pre, (NEG, NEG))
                    new[pre] = (_lse(b0, tot + v), n0)
                    continue
                # 从 l 延展出一个字符 c：
                #   若 c == l 的末字符 ⇒ 只有「上一帧是 blank」的路径能延展（p_b），
                #   而非 blank 路径重复 c 会退回**同一个前缀 l**（p_nb 自指）；
                #   否则两条路径都能延展（p_b + p_nb）。
                # 🔴 之前把 tot 和 p_b 同时算进 l+c，导致同字符场景概率翻倍。
                if pre and pre[-1] == c:
                    nb0, nn0 = new.get(pre + (c,), (NEG, NEG))
                    new[pre + (c,)] = (nb0, _lse(nn0, pb + v))
                    b1, n1 = new.get(pre, (NEG, NEG))
                    new[pre] = (b1, _lse(n1, pnb + v))
                else:
                    nb0, nn0 = new.get(pre + (c,), (NEG, NEG))
                    new[pre + (c,)] = (nb0, _lse(nn0, tot + v))
        ranked = sorted(new.items(), key=lambda kv: -_lse(kv[1][0], kv[1][1]))
        beams = dict(ranked[:beam])
    out = sorted(((pre, _lse(pb, pnb)) for pre, (pb, pnb) in beams.items()),
                 key=lambda x: -x[1])
    return out[:nbest]


class BigramLM:
    """词级 bigram（train 标注统计；绝对折扣 + unigram 回退）。

    🔴 一开始用 add-α 插值，未登录 bigram 会被分母（几百）压成 1e-7，
    log 惩罚 ≈ -16/词，重排必然越调越差。改成绝对折扣（d=0.75）：
      p(t|ctx) = max(c-d,0)/N + (d·|T(ctx)|/N)·p_uni(t)
    未登录 bigram 只剩一个温和的折扣因子。
    """

    def __init__(self, sents, alpha=0.75):
        self.d = float(alpha)
        self.uni = Counter()
        self.bi = defaultdict(Counter)
        for toks in sents:
            seq = ["<s>"] + list(toks) + ["</s>"]
            for a, b in zip(seq, seq[1:]):
                self.bi[a][b] += 1
            for t in seq[1:]:
                self.uni[t] += 1
        self.total = float(sum(self.uni.values())) or 1.0
        self.V = max(1, len(self.uni))

    def logp(self, toks) -> float:
        s = 0.0
        ctx = "<s>"
        for t in list(toks) + ["</s>"]:
            c = self.bi.get(ctx)
            p_uni = (self.uni.get(t, 0) + 0.5) / (self.total + 0.5 * self.V)
            if not c:
                p = p_uni
            else:
                denom = float(sum(c.values()))
                p = (max(c.get(t, 0) - self.d, 0.0) / denom
                     + (self.d * len(c) / denom) * p_uni)
            s += np.log(max(p, 1e-12))
            ctx = t
        return float(s)


def greedy_fast(logp: np.ndarray, idx2word, blank_bias: float = 0.0,
                blank: int = 0):
    """在已裁好的 [T, C] log-prob 上做 argmax+去重（等价 greedy_decode）。

    blank_bias>0 ⇒ 压低 blank 概率（抑制「吞词」/删除型错误）。
    """
    lp = logp
    if blank_bias:
        lp = logp.copy()
        lp[:, blank] -= blank_bias
    arg = lp.argmax(axis=1)
    toks, prev = [], -1
    for k in arg:
        k = int(k)
        if k != prev and k != blank and 0 <= k < len(idx2word):
            toks.append(idx2word[k])
        prev = k
    return toks


def build_model(ckpt, hidden, word_set_num, device):
    mdl = P78.Net.moduleNet(hidden, word_set_num * 1 + 1, "TFNet",
                            device, "CE-CSL", True)
    ck = torch.load(str(ckpt), map_location=device, weights_only=False)
    sd = ck.get("model_state", ck)
    mdl.load_state_dict(sd, strict=True)
    return mdl.to(device).eval()


@torch.no_grad()
def collect_logps(mdl, loader, device, labels_dv):
    """一次前向拿到所有 dev 样本的 [T_conv, C] log-prob 与真值/参考。"""
    items = []
    for vid, _tgt, _tl, dl, _tll, sids, _ids in loader:
        out = mdl(vid.to(device), dl, False)
        logp = F.log_softmax(out[0], dim=-1).cpu().numpy()   # [T,B,C]
        lens = dl.view(-1).tolist()
        for b, sid in enumerate(sids):
            T = min(P78._conv_len(int(lens[b])), logp.shape[0])
            items.append({
                "sid": str(sid),
                "logp": logp[:T, b, :].astype(np.float32),
                "ref": labels_dv[sid][1],
            })
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/p184-tfnet-freeze-best.pt")
    ap.add_argument("--tag", default="p197")
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--nbest", type=int, default=8)
    ap.add_argument("--lm-alpha", type=float, default=0.75)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3])
    ap.add_argument("--blank-bias-grid", type=float, nargs="+",
                    default=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_dv = P78.read_csv_split("dev.csv")
    labels_tr = P78.read_csv_split("train.csv")
    word2idx, word_set_num, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")

    _, dv_tf = P78.build_transforms(hflip=False, img_size=a.img_size,
                                    y_shift=0, temporal_rescale=0.0)
    dv_ds = P78.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    loader = DataLoader(dv_ds, batch_size=a.batch, shuffle=False,
                        collate_fn=P78.collate, num_workers=0)
    print("dev items = %d, vocab = %d" % (len(dv_ds), word_set_num + 1),
          flush=True)

    mdl = build_model(REPO / a.ckpt, 1024, word_set_num, device)
    items = collect_logps(mdl, loader, device, labels_dv)
    print("前向完成：%d 条" % len(items), flush=True)

    refs = [it["ref"] for it in items]

    # ── 0. greedy 基线（必须先复现官方 34.34 才说明管线对齐）
    g_hyps = [greedy_fast(it["logp"], idx2word) for it in items]
    g_res = P78.evaluate(refs, g_hyps)
    print("[greedy]       WER_official %.2f%%" % g_res["WER_official"],
          flush=True)

    # ── 0b. blank-bias 扫描（只改解码，不重训；针对 24.5% 删除型错误）
    bias_grid = {}
    for bb in a.blank_bias_grid:
        if bb == 0.0:
            bias_grid[bb] = g_res["WER_official"]
            continue
        hyps = [greedy_fast(it["logp"], idx2word, blank_bias=bb) for it in items]
        bias_grid[bb] = P78.evaluate(refs, hyps)["WER_official"]
        print("[blank bias %.1f] WER_official %.2f%%" % (bb, bias_grid[bb]),
              flush=True)

    # ── 1. beam N-best + oracle 上界
    nb_list, oracle_e, oracle_n = [], 0, 0
    for it in items:
        nb = ctc_prefix_beam(it["logp"], a.beam, a.topk, a.nbest)
        nb_list.append(nb)
    ref_tok = [OW.split_glued(OW.split_gloss_sequence(g, OW.CFG))
               for g in refs]
    for rt, nb in zip(ref_tok, nb_list):
        best = min(OW.levenshtein(rt, OW.split_glued([idx2word[c] for c in pre]))
                   for pre, _ in nb)
        oracle_e += best
        oracle_n += len(rt)
    print("[beam oracle]  WER_official %.2f%%  (N=%d 的头寸)"
          % (100.0 * oracle_e / max(1, oracle_n), a.nbest), flush=True)
    b1 = [[idx2word[c] for c in nb[0][0]] for nb in nb_list]
    b1_res = P78.evaluate(refs, b1)
    print("[beam 1-best]  WER_official %.2f%%" % b1_res["WER_official"],
          flush=True)

    # ── 2. LM 重排
    # 🔴 LM 词表必须与官方 WER 口径一致：ref 先按 "/" 切，再 split_glued 拆粘连词，
    #    否则 ref 里的「可以支持」与模型输出的「可以」「支持」在 LM 里是两套词，
    #    未登录惩罚会假性把重排分数打偏（实测 λ 越大越差就是这个原因）。
    lm_sents = []
    for _s, (_tr, seq) in labels_tr.items():
        toks = [t for t in seq.split("/") if t]
        lm_sents.append(OW.split_glued(toks))
    lm = BigramLM(lm_sents, alpha=a.lm_alpha)
    print("LM: %d 句, %d 词, alpha=%.2f"
          % (len(lm_sents), lm.V, a.lm_alpha), flush=True)
    print("长度诊断：greedy 平均 %.2f 词 / beam 平均 %.2f 词"
          % (np.mean([len(h) for h in g_hyps]),
             np.mean([len(h) for h in b1]),), flush=True)
    grid = {}
    for lam in a.lambdas:
        hyps = []
        for nb in nb_list:
            best_pre, best_score = nb[0][0], NEG
            for pre, lp_ctc in nb:
                toks = [idx2word[c] for c in pre]
                sc = lp_ctc + lam * lm.logp(toks)
                if sc > best_score:
                    best_score, best_pre = sc, pre
            hyps.append([idx2word[c] for c in best_pre])
        r = P78.evaluate(refs, hyps)
        grid[lam] = r["WER_official"]
        print("[LM λ=%.1f]      WER_official %.2f%%" % (lam, r["WER_official"]),
              flush=True)

    best_lam = min(grid, key=lambda k: grid[k])
    out = {
        "ckpt": a.ckpt,
        "beam": a.beam, "topk": a.topk, "nbest": a.nbest,
        "greedy": g_res, "beam_1best": b1_res,
        "blank_bias_grid": {str(k): v for k, v in bias_grid.items()},
        "beam_oracle_wer": round(100.0 * oracle_e / max(1, oracle_n), 2),
        "lm_grid": {str(k): v for k, v in grid.items()},
        "best_lambda": best_lam,
        "best_wer": grid[best_lam],
        "n_sentences": len(items),
    }
    out_path = REPO / (a.out or "artifacts/metrics/blank-gov/%s-beam-lm-dev.json" % a.tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("收据 -> %s" % out_path, flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
