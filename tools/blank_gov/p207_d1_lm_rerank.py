#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P207 / D1 阶段二：用本地中文预训练 LM 对 8-best gloss 序列重排（零训练）。

与 P197 的区别：P197 用 train 4973 句统计出的 bigram（语料稀疏，越调越差）；
这里用 Qwen2.5-1.5B-Instruct 的**纯语言模型 logP**（自带语言知识，不吃稀疏统计）。
gloss 序列按 base-LM 续写方式打分，不加 chat template。

预注册判据（写死，不得事后改）：
  - 主判据：dev WER_official <= 33.60（比 34.34 好 >= 0.7pp）才算过线；
  - 过线还需配对 bootstrap 95% CI 不含 0；
  - 否则判负，不跑 test。
  - 同时报 oracle 利用率，作为「候选集合信息量」诊断（非性能）。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for _p in (REPO / "tools/blank_gov", REPO / "src"):
    sys.path.insert(0, str(_p))

import official_wer as OW  # noqa: E402

QWEN = ("/mnt/d/part3_models/ms_cache/models/"
        "Qwen--Qwen2.5-1.5B-Instruct/snapshots/master")


def dedup(nb_list):
    """同一 collapse 前缀可能被多条对齐路径命中，去重保留 lp_ctc 最大者。"""
    out = []
    for nb in nb_list:
        seen, keep = set(), []
        for c in nb:
            key = tuple(c["tokens"])
            if key in seen:
                continue
            seen.add(key)
            keep.append(c)
        out.append(keep)
    return out


def render(tokens, join_mode):
    if join_mode == "space":
        return " ".join(tokens)
    if join_mode == "plain":
        return "".join(tokens)
    raise ValueError(join_mode)


@torch.no_grad()
def lm_logp_batched(model, tok, texts, device, batch_size=32):
    """teacher-forcing 序列 logP（base LM，无 chat template）。

    返回 (sum_logp, n_tok)：sum_logp = Σ log p(x_i | x_<i)，
    n_tok 为被统计的 token 数（排除 padding 与首 token）。
    """
    sums, counts = [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        ids = enc["input_ids"]
        am = enc["attention_mask"]
        out = model(input_ids=ids, attention_mask=am, use_cache=False)
        lg = torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
        lab = ids[:, 1:]
        lp = lg.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
        mask = am[:, 1:].float()
        sums.append((lp * mask).sum(-1).cpu().numpy())
        counts.append(mask.sum(-1).cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print("  LM %d/%d" % (min(i + batch_size, len(texts)), len(texts)),
                  flush=True)
    return np.concatenate(sums), np.concatenate(counts)


def paired_bootstrap(base_e, base_n, cand_e, cand_n, n_boot=20000, seed=42):
    """逐句编辑距离配对的 bootstrap CI（语料级 WER 差）。

    返回 (delta_pp, ci_lo, ci_hi, better, worse, tie)；delta = base - cand，
    delta > 0 表示候选更好。
    """
    rng = np.random.default_rng(seed)
    b = np.asarray(base_e, dtype=float)
    c = np.asarray(cand_e, dtype=float)
    n = np.asarray(base_n, dtype=float)
    d = b - c
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].sum(1) / n.sum() * 100.0
    delta = d.sum() / n.sum() * 100.0
    return (float(delta), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
            int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest-pkl", default="artifacts/cache/p207_d1_nbest.pkl")
    ap.add_argument("--tag", default="p207-d1")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                             0.1, 0.3, 1.0],
                    help="raw 模式用（LM 累积 logP 量级远大于 lp_ctc）")
    ap.add_argument("--z-lambdas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5],
                    help="zscore 模式用（句内标准化后两种分数同量级）")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    with open(REPO / a.nbest_pkl, "rb") as f:
        data = pickle.load(f)
    print("载入 %s: n=%d, greedy=%.2f, beam1=%.2f, oracle=%.2f"
          % (a.nbest_pkl, data["n"], data["greedy_wer"],
             data["beam1_wer"], data["oracle_wer"]), flush=True)

    items = data["items"]
    nbs = dedup([it["nbest"] for it in items])
    print("去重后平均候选数 %.2f" % np.mean([len(x) for x in nbs]), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("加载 Qwen: %s" % QWEN, flush=True)
    tok = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        QWEN, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    print("Qwen 加载完成", flush=True)

    # 展平所有候选，一次性算 LM 分数；spans 对两种拼接方式结构相同，只算一次
    flat_texts, spans, join_modes = {}, [], ["plain", "space"]
    start = 0
    for nb in nbs:
        spans.append((start, start + len(nb)))
        start += len(nb)
    for jm in join_modes:
        flat_texts[jm] = [render(c["tokens"], jm) for nb in nbs for c in nb]
    scores = {}
    for jm in join_modes:
        s, n = lm_logp_batched(model, tok, flat_texts[jm], device, a.batch_size)
        scores[jm] = (s, n)
        print("[%s] logP mean %.2f, tok mean %.1f"
              % (jm, s.mean(), n.mean()), flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    refs = [it["ref"] for it in items]
    ref_tok = [OW.split_glued(OW.split_gloss_sequence(g, OW.CFG)) for g in refs]
    base_hyps = [it["greedy"] for it in items]
    base_res = OW.evaluate(refs, base_hyps)
    base_e = [OW.levenshtein(r, OW.split_glued(h))
              for r, h in zip(ref_tok, base_hyps)]
    base_n = [len(r) for r in ref_tok]
    print("[基线 greedy] WER_official %.2f%%" % base_res["WER_official"],
          flush=True)

    results = {}
    # 三种打分模式：
    #  raw    = lp_ctc + lam * sum_logp         （P197 原式，便于直接对照）
    #  norm   = lp_ctc/L + lam * sum_logp/n_tok （长度归一，消除短句偏置）
    #  zscore = z(lp_ctc) + lam * z(logP/n_tok) （句内标准化，两种分数同量级，最稳健）
    for jm in join_modes:
        s_all, n_all = scores[jm]
        for mode in ("raw", "norm", "zscore"):
            grid = {}
            lams = a.z_lambdas if mode == "zscore" else a.lambdas
            for lam in lams:
                hyps, edits = [], []
                for i, nb in enumerate(nbs):
                    st, ed = spans[i][0], spans[i][1]
                    ctc = np.array([nb[k - st]["lp_ctc"] for k in range(st, ed)])
                    lms = np.array([float(s_all[k]) / max(float(n_all[k]), 1.0)
                                    for k in range(st, ed)])
                    if mode == "raw":
                        sc = ctc + lam * np.array(
                            [float(s_all[k]) for k in range(st, ed)])
                    elif mode == "norm":
                        L = np.array([max(len(nb[k - st]["tokens"]), 1)
                                      for k in range(st, ed)])
                        sc = ctc / L + lam * lms
                    else:
                        zc = (ctc - ctc.mean()) / max(ctc.std(), 1e-8)
                        zl = (lms - lms.mean()) / max(lms.std(), 1e-8)
                        sc = zc + lam * zl
                    j = int(np.argmax(sc))
                    pre = nb[j]["tokens"]
                    hyps.append(list(pre))
                    edits.append(OW.levenshtein(
                        ref_tok[i], OW.split_glued(list(pre))))
                r = OW.evaluate(refs, hyps)
                delta, lo, hi, bw, ww, tie = paired_bootstrap(
                    base_e, base_n, edits, base_n)
                grid[str(lam)] = {
                    "WER_official": r["WER_official"],
                    "delta_pp": round(delta, 3),
                    "ci95": [round(lo, 3), round(hi, 3)],
                    "better": bw, "worse": ww, "tie": tie,
                    "oracle_util": round(
                        (base_res["WER_official"] - r["WER_official"])
                        / (base_res["WER_official"] - data["oracle_wer"]), 4),
                }
                print("[%s/%s lam=%.4g] WER %.2f  Δ%+.2f CI[%.2f,%.2f] %d/%d/%d"
                      % (jm, mode, lam, r["WER_official"], delta, lo, hi,
                         bw, ww, tie), flush=True)
            results["%s/%s" % (jm, mode)] = grid

    best = None
    for key, grid in results.items():
        for lam, v in grid.items():
            if best is None or v["WER_official"] < best[1]["WER_official"]:
                best = ("%s lam=%s" % (key, lam), v)
    print("\n[best] %s -> WER %.2f  Δ%+.2f  CI %s"
          % (best[0], best[1]["WER_official"], best[1]["delta_pp"],
             best[1]["ci95"]), flush=True)
    verdict = ("PASS" if (best[1]["WER_official"] <= 33.60
                          and best[1]["ci95"][0] > 0) else "FAIL")
    print("[预注册判据] dev<=33.60 且 CI 不含 0 => %s" % verdict, flush=True)

    out = {
        "tag": a.tag,
        "nbest_pkl": a.nbest_pkl,
        "n": len(items),
        "lm": "Qwen2.5-1.5B-Instruct (local, base-LM logP, no chat template)",
        "baseline_greedy_wer": base_res["WER_official"],
        "beam1_wer": data["beam1_wer"],
        "oracle_wer": data["oracle_wer"],
        "pre_registered": {
            "threshold": 33.60, "noise_floor_pp": 0.77,
            "rule": "dev<=33.60 且配对 bootstrap 95% CI 不含 0 才过线，否则判负不进 test",
        },
        "grid": results,
        "best": {"setting": best[0], **best[1]},
        "verdict": verdict,
        "note": ("oracle_wer 为用参考标签挑最优候选的作弊上界，不可达；"
                 "oracle_util 仅表示候选集合信息被利用的比例，非性能增益"),
    }
    out_path = REPO / (a.out or "artifacts/metrics/blank-gov/%s-lm-rerank.json" % a.tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("收据 -> %s" % out_path, flush=True)


if __name__ == "__main__":
    main()
