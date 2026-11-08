#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P226 / H1-b：kNN 后验精修（RAC 分布插值）+ λ 扫描 + 全指标评估。

记忆库只来自 train；dev 段查询；λ 选择规则与判据见 H1-PREREG（启动前写死）。
只做 dev，不碰 test；主干/适配器/分类头全部不动。
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402
import DataProcessMoudle as DPM  # noqa: E402
from p197_beam_lm import greedy_fast  # noqa: E402
from p210_d4_eval import align_ops  # noqa: E402

M = REPO / "artifacts/metrics/blank-gov"
CACHE = REPO / "artifacts/cache"
TAG = "p226-h1"
PCA_DIM, KNN_K = 128, 32
LAMBDAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    tr = load_pkl(CACHE / "p224_train_segments.pkl")
    dv = load_pkl(CACHE / "p224_dev_segments.pkl")
    _, n, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    C = n + 1
    print("train segs=%s  dev segs=%s" % (tr["Z"].shape, dv["Z"].shape), flush=True)

    labels_dv = P78.read_csv_split("dev.csv")
    labels_tr = P78.read_csv_split("train.csv")
    freq = defaultdict(int)
    for _sid, (_p, seq) in labels_tr.items():
        for t in DPM.PreWords(str(seq).split("/")):
            freq[t] += 1

    # PCA 只在 train 库上 fit（防泄漏）
    t0 = time.time()
    pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=42)
    Ztr = pca.fit_transform(tr["Z"]).astype(np.float32)
    Zdv = pca.transform(dv["Z"]).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=KNN_K, metric="cosine", algorithm="brute")
    nn.fit(Ztr)
    dist, nbr = nn.kneighbors(Zdv)
    knn_ms = time.time() - t0
    print("PCA+KNN index %.1fs  query %.3fs/segment"
          % (knn_ms, knn_ms / max(len(Zdv), 1)), flush=True)

    Pk = np.zeros((len(Zdv), C), dtype=np.float32)
    for i in range(len(Zdv)):
        w = np.clip(1.0 - dist[i], 0.0, None)
        for j, ww in zip(nbr[i], w):
            Pk[i, int(tr["Y"][j])] += ww
        s = float(Pk[i].sum())
        if s > 0:
            Pk[i] /= s

    # dev 后验（已落盘，同一 p184 模型）
    d = torch.load(M / "p188_logp_p184.pt", map_location="cpu", weights_only=False)
    sids = [s for s in dv["NT"] if s in labels_dv and s in d["items"]]
    print("dev sentences with posterior =", len(sids), flush=True)
    # OW.evaluate 要原始 gloss 字符串；align_ops 要切分后的 token list
    raw_refs = [labels_dv[s][1] for s in sids]
    refs = [OW.split_glued(OW.split_gloss_sequence(labels_dv[s][1], OW.CFG))
            for s in sids]
    base_lp = {}
    for s in sids:
        lp = d["items"][s]["logp"]
        base_lp[s] = (lp.numpy() if hasattr(lp, "numpy") else np.asarray(lp)
                      ).astype(np.float32).copy()
    # 段索引按句子分组
    seg_by_sid = defaultdict(list)
    for i, sid in enumerate(dv["SID"]):
        if sid in base_lp:
            seg_by_sid[sid].append(i)

    def metrics(hyps):
        per_cls = defaultdict(lambda: {"n": 0, "err": 0})
        lo, hi = {"n": 0, "err": 0}, {"n": 0, "err": 0}
        ins = sub = dele = nref = 0
        per_item = []
        for r, h in zip(refs, hyps):
            hh = OW.split_glued(list(h))
            ops, ii = align_ops(r, hh)
            for t, op in zip(r, ops):
                bk = lo if freq.get(t, 0) <= 10 else hi
                per_cls[t]["n"] += 1
                per_cls[t]["err"] += (op != "match")
                bk["n"] += 1
                bk["err"] += (op != "match")
            s_ = sum(o == "sub" for o in ops)
            d_ = sum(o == "del" for o in ops)
            ins += ii; sub += s_; dele += d_; nref += len(r)
            _ln = sum(1 for t in r if freq.get(t, 0) <= 10)
            _le = sum(1 for t, op in zip(r, ops)
                      if freq.get(t, 0) <= 10 and op != "match")
            per_item.append({"sid": "", "ref_n": len(r),
                             "edits": OW.levenshtein(r, hh), "ins": ii,
                             "sub": s_, "del": d_, "low_n": _ln, "low_err": _le})
        cls_rates = [v["err"] / v["n"] for v in per_cls.values() if v["n"] > 0]
        return {
            "WER_official": OW.evaluate(raw_refs, hyps)["WER_official"],
            "uWER_pct": round(float(np.mean(cls_rates)) * 100, 2),
            "low_bucket_WER": round(100 * lo["err"] / max(lo["n"], 1), 2),
            "high_bucket_WER": round(100 * hi["err"] / max(hi["n"], 1), 2),
            "ins": int(ins), "sub": int(sub), "del": int(dele),
            "micro_subdel_pct": round(100 * (sub + dele) / max(nref, 1), 2),
            "ins_rate_pct": round(100 * ins / max(nref, 1), 2),
            "low_n": lo["n"], "high_n": hi["n"], "n_ref": nref,
        }, per_item

    grid = {}
    best_hyps = {}
    for lam in LAMBDAS:
        hyps = []
        for s in sids:
            lp = base_lp[s].copy()
            if lam > 0:
                for k in seg_by_sid.get(s, []):
                    s0, e0 = dv["SE"][k]
                    e0 = min(e0, lp.shape[0])
                    if s0 >= e0:
                        continue
                    p = np.exp(lp[s0:e0])
                    fused = (1.0 - lam) * p + lam * Pk[k][None, :]
                    lp[s0:e0] = np.log(np.maximum(fused, 1e-8))
            hyps.append(greedy_fast(lp, idx2word))
        r, _ = metrics(hyps)
        grid["%.2f" % lam] = r
        best_hyps[lam] = hyps
        print("lam=%.2f WER %.2f uWER %.2f low %.2f high %.2f ins %d micro %.2f"
              % (lam, r["WER_official"], r["uWER_pct"], r["low_bucket_WER"],
                 r["high_bucket_WER"], r["ins"], r["micro_subdel_pct"]), flush=True)

    b = grid["0.00"]
    guard = []
    for key, r in grid.items():
        if (r["high_bucket_WER"] - b["high_bucket_WER"] <= 1.0
                and r["WER_official"] - b["WER_official"] <= 0.30
                and (r["ins"] - b["ins"]) / max(b["ins"], 1) <= 0.10
                and r["micro_subdel_pct"] - b["micro_subdel_pct"] <= 0.30):
            guard.append((r["uWER_pct"], key))
    chosen = min(guard)[1] if guard else "0.00"
    lam = float(chosen)
    print("chosen (pre-registered rule) = %.2f  guard-passing = %d"
          % (lam, len(guard)), flush=True)
    rep, per_item = metrics(best_hyps[lam])

    uwer = {
        "tag": TAG, "ckpt": "p184-tfnet-freeze-best.pt + kNN posterior refinement",
        "n": len(sids), "WER_official": rep["WER_official"],
        "uWER_pct": rep["uWER_pct"],
        "uWER_n_classes": len(set(t for r in refs for t in r)),
        "low_freq_max": 10,
        "low_bucket": {"n_tokens": rep["low_n"], "WER_pct": rep["low_bucket_WER"]},
        "high_bucket": {"n_tokens": rep["high_n"], "WER_pct": rep["high_bucket_WER"]},
        "edits": {"ins": rep["ins"], "sub": rep["sub"], "del": rep["del"],
                  "n_ref_tokens": rep["n_ref"],
                  "micro_subdel_pct": rep["micro_subdel_pct"],
                  "ins_rate_pct": rep["ins_rate_pct"]},
        "nonblank_frame_ratio_pct": None,
        "per_item": per_item,
        "note": "H1 代表组合 lam=%.2f（护栏内 uWER 最低）；记忆库仅 train，PCA%d/K%d"
                % (lam, PCA_DIM, KNN_K),
    }
    (M / (TAG + "-uwer.json")).write_text(
        json.dumps(uwer, ensure_ascii=False, indent=2), encoding="utf-8")
    (M / (TAG + "-scan.json")).write_text(json.dumps({
        "grid": grid, "chosen": chosen, "guard_count": len(guard),
        "lambdas": LAMBDAS, "pca_dim": PCA_DIM, "knn_k": KNN_K,
        "train_segments": int(tr["Z"].shape[0]),
        "dev_segments": int(dv["Z"].shape[0]),
        "knn_index_seconds": round(knn_ms, 2),
        "selection_rule": "guard R3/R4/R5/R6 first, then min uWER; "
                          "if none pass -> baseline lam=0 (auto FAIL)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("收据 -> %s / %s-scan.json" % (TAG, TAG), flush=True)


if __name__ == "__main__":
    main()
