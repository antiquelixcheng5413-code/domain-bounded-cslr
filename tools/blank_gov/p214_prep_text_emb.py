#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P214/F2 离线预处理：用本地 Qwen2.5-1.5B 生成 3516 个 gloss 的文本语义目标。

输出：
  artifacts/cache/p214_vta_target.npy  [C,1024] float32，L2 归一化
  artifacts/cache/p214_vta_mask.npy    [C]      float32，低频类 1.0 / 高频类 0.2 / blank 0
做法：last_hidden_state 均值池化 -> 中心化 -> PCA(1024) -> L2 归一化。
投影固定，训练时不引入任何可训练参数（避免 6 轮内欠拟合）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78  # noqa: E402
import DataProcessMoudle as DPM  # noqa: E402

QWEN = Path("/mnt/d/part3_models/ms_cache/models/"
            "Qwen--Qwen2.5-1.5B-Instruct/snapshots/master")
OUT_T = REPO / "artifacts/cache/p214_vta_target.npy"
OUT_M = REPO / "artifacts/cache/p214_vta_mask.npy"


def main():
    from transformers import AutoModel, AutoTokenizer
    word2idx, n, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    C = n + 1
    print("vocab C =", C, flush=True)

    # 用 word2idx 反查，保证文本向量的行号与模型分类头列号严格一致
    id2word = {int(v): str(k) for k, v in word2idx.items()}
    texts = ["" if i == 0 else id2word.get(i, "") for i in range(C)]
    print("sample gloss:", texts[1:9],
          " non-empty=%d" % sum(1 for t in texts[1:] if t), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(QWEN), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained(str(QWEN), trust_remote_code=True).to(dev).eval().float()
    H = mdl.config.hidden_size
    emb = np.zeros((C, H), dtype=np.float32)
    bs = 64
    with torch.no_grad():
        for st in range(0, C, bs):
            batch = [t if t else " " for t in texts[st:st + bs]]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=16).to(dev)
            out = mdl(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * m).sum(1) / m.sum(1).clamp_min(1.0)
            emb[st:st + len(batch)] = pooled.cpu().numpy()
            if st % 512 == 0:
                print("embedded %d/%d" % (st, C), flush=True)
    del mdl
    if dev == "cuda":
        torch.cuda.empty_cache()

    # 中心化 + PCA 到 1024 + L2 归一化
    X = emb.copy()
    X[0] = 0.0
    mean = X[1:].mean(0, keepdims=True)
    X[1:] -= mean
    X = torch.from_numpy(X[1:]).to(dev)
    U, S, V = torch.pca_lowrank(X, q=1024, center=False, niter=2)
    Z = (X @ V).cpu().numpy()                 # [C-1,1024]
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    target = np.zeros((C, 1024), dtype=np.float32)
    target[1:] = Zn.astype(np.float32)

    # 低频 mask（与 D4/E0 频次表同口径）
    tr = P78.read_csv_split("train.csv")
    cnt = np.zeros(C, dtype=np.float64)
    for _sid, (_p, seq) in tr.items():
        for t in DPM.PreWords(str(seq).split("/")):
            g = word2idx.get(t)
            if g is not None and 0 <= g < C:
                cnt[g] += 1.0
    mask = np.where(cnt <= 10, 1.0, 0.2).astype(np.float32)
    mask[0] = 0.0
    OUT_T.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_T, target)
    np.save(OUT_M, mask)
    print("saved %s %s" % (OUT_T, target.shape), flush=True)
    print("saved %s %s  low-freq classes=%d" % (OUT_M, mask.shape,
                                                int((mask == 1.0).sum())), flush=True)
    # 自检：语义邻域是否非空（相邻 gloss 的 cosine 应显著高于随机对）
    rng = np.random.default_rng(42)
    i = rng.integers(1, C, size=512)
    j = rng.integers(1, C, size=512)
    rnd = float(np.mean((target[i] * target[j]).sum(1)))
    near = float(np.mean((target[1:100] * target[2:101]).sum(1)))
    print("self-check: random-pair cos=%.4f  adjacent-index cos=%.4f  norm=%.4f"
          % (rnd, near, float(np.linalg.norm(target[1]))), flush=True)


if __name__ == "__main__":
    main()
