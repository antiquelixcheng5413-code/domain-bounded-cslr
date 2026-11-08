#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P230 / K1 语料合成流水线（A align / B sentences / C synth / report）。

预注册见 docs/planning/K1-PREREG-2026-11-08.md（20:45 写死）。
A: train 全量 fp32 前向 + CTC Viterbi 强制对齐 -> per-gloss 片段库（分片断点）
B: train 统计 bigram + 低频优先 -> 1500 条新 gloss 句（本地替代 LLM，标注【无】）
C: 随机取片段、硬链接落盘合成 RGB 样本 + manifest
report: 读 judge 收据生成阶段报告（WSL + 桌面镜像）
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
DESK = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
for p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(p))

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402
import DataProcessMoudle as DPM  # noqa: E402
from p197_beam_lm import build_model  # noqa: E402

P78.OFFICIAL_PAD = True
P78.RGB_FIX = True
P78.BLANK_BIAS = 0.0

CACHE = REPO / "artifacts/cache"
M = REPO / "artifacts/metrics/blank-gov"
CLIPS = CACHE / "p230_k1_clips.json"
SENTS = CACHE / "p231_k1_sentences.json"
MANIFEST = CACHE / "p232_k1_manifest.json"
PARTS = CACHE / "p230_k1_align_parts"
SYNTH_TR = "k1syn"
N_SENT = 1500
SEED = 42
LOW_FREQ_MAX = 10
MIN_LEN, MAX_LEN = 3, 10
CHUNK = 25
MAX_T = 400


# --------------------------------------------------------------- forced align
def forced_align(logp, wid):
    """标准 CTC Viterbi 强制对齐，与 p229_j1j2_probe.py 同一实现。"""
    L = len(wid)
    S = 2 * L + 1
    ext = [0] * S
    for i, w in enumerate(wid):
        ext[2 * i + 1] = w
    T = logp.shape[0]
    if T < S:
        return None
    dp = np.full((T, S), -np.inf, dtype=np.float64)
    bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0], dp[0, 1] = logp[0, ext[0]], logp[0, ext[1]]
    for t in range(1, T):
        emis = np.asarray([logp[t, e] for e in ext], dtype=np.float64)
        for s in range(S):
            best, arg = dp[t - 1, s], 0
            if s >= 1 and dp[t - 1, s - 1] > best:
                best, arg = dp[t - 1, s - 1], 1
            if (s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2]
                    and dp[t - 1, s - 2] > best):
                best, arg = dp[t - 1, s - 2], 2
            dp[t, s] = best + emis[s]
            bp[t, s] = arg
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = s
    for t in range(T - 1, 0, -1):
        s -= int(bp[t, s])
        path[t - 1] = s
    return path


def spans_from_path(path, L, T_conv):
    """相邻 spike 中点切分，返回 L 个 [s_step, e_step)。缺失 spike 线性插值。"""
    pos = []
    for i in range(L):
        f = np.where(path == 2 * i + 1)[0]
        pos.append(float(f[0]) if len(f) else -1.0)
    good = [i for i, p in enumerate(pos) if p >= 0]
    if not good:
        return None
    for i in range(L):
        if pos[i] >= 0:
            continue
        left = max([j for j in good if j < i], default=None)
        right = min([j for j in good if j > i], default=None)
        if left is None and right is None:
            pos[i] = T_conv / 2
        elif left is None:
            pos[i] = max(0.0, pos[right] - (right - i))
        elif right is None:
            pos[i] = pos[left] + (i - left)
        else:
            pos[i] = pos[left] + (pos[right] - pos[left]) * (i - left) / (right - left)
    edges = [0.0] + [(pos[i] + pos[i + 1]) / 2 for i in range(L - 1)] + [float(T_conv)]
    return [(edges[i], edges[i + 1]) for i in range(L)]


# --------------------------------------------------------------- stage A
def stage_align(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels_tr = P78.read_csv_split("train.csv")
    word2idx, n, _ = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    _, dv_tf = P78.build_transforms(hflip=False, img_size=160,
                                    y_shift=0, temporal_rescale=0.0)
    ds = P78.RGBSeqDataset("train", labels_tr, word2idx, dv_tf, False)
    if a.limit:
        ds.items = ds.items[:a.limit]
        PARTS.mkdir(parents=True, exist_ok=True)
        for f in PARTS.glob("part*.pkl"):
            f.unlink()
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=P78.collate, num_workers=0)
    print("train items=%d vocab=%d" % (len(ds), n + 1), flush=True)

    PARTS.mkdir(parents=True, exist_ok=True)
    parts = sorted(PARTS.glob("part*.pkl"))
    done = 0
    for pp in parts:
        with open(pp, "rb") as f:
            done += len(pickle.load(f))
    if done:
        print("分片断点恢复 %d 片 / %d 句" % (len(parts), done), flush=True)
    part_no = len(parts)

    mdl = build_model(str(REPO / a.ckpt), 1024, n, dev).eval()
    t0 = time.time()
    buf = []
    skipped = 0
    for bi, (vid, _tgt, _tl, dl, true_len, sids, _ids) in enumerate(loader):
        if bi < done:
            continue
        sid = str(sids[0])
        if int(vid.shape[2]) > MAX_T:
            skipped += 1
            buf.append([])          # 占位，保持分片与句序一致
        else:
            with torch.no_grad():
                out = mdl(vid.to(dev), dl, False)
                lp = F.log_softmax(out[0], dim=-1).float()
            if dev == "cuda":
                torch.cuda.synchronize()
            lp = lp[:, 0, :].cpu().numpy().astype(np.float64)
            del out
            ref = labels_tr[sid][1]
            toks = DPM.PreWords(str(ref).split("/"))
            pairs = [(t, word2idx[t]) for t in toks if t in word2idx]
            if not pairs:
                buf.append([])
            else:
                wid = [w for _t, w in pairs]
                path = forced_align(lp, wid)
                if path is None:
                    buf.append([])
                else:
                    sp = spans_from_path(path, len(wid), lp.shape[0])
                    ratio = float(int(true_len[0])) / max(lp.shape[0], 1)
                    tr = labels_tr[sid][0]
                    d = "%s/%s" % (tr, sid)
                    row = []
                    for (t, _w), (s0, e0) in zip(pairs, sp):
                        s = int(round(s0 * ratio))
                        e = int(round(e0 * ratio))
                        s = max(0, min(s, int(true_len[0])))
                        e = max(s + 1, min(e, int(true_len[0])))
                        row.append([t, d, s, e])
                    buf.append(row)
        if len(buf) >= CHUNK:
            with open(PARTS / ("part%04d.pkl" % part_no), "wb") as f:
                pickle.dump(buf, f, protocol=4)
            buf = []
            part_no += 1
        if dev == "cuda" and (bi + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if bi % 100 == 0:
            print("align %d/%d elapsed=%.1fmin skipped=%d"
                  % (bi, len(loader), (time.time() - t0) / 60, skipped), flush=True)
    if buf:
        with open(PARTS / ("part%04d.pkl" % part_no), "wb") as f:
            pickle.dump(buf, f, protocol=4)

    clips = defaultdict(list)
    span_frames = []
    for pp in sorted(PARTS.glob("part*.pkl")):
        with open(pp, "rb") as f:
            for row in pickle.load(f):
                for t, d, s, e in row:
                    clips[t].append([d, s, e])
                    span_frames.append(e - s)
    sf = np.asarray(span_frames, float)
    out = {
        "tag": "p230-k1", "source": "train only (P184 fp32 forward + CTC Viterbi forced alignment)",
        "n_clips": int(len(sf)), "n_gloss": len(clips),
        "n_skipped_long": skipped, "seed": SEED,
        "span_frames": {"median": float(np.median(sf)) if len(sf) else 0,
                        "mean": round(float(sf.mean()), 2) if len(sf) else 0,
                        "p10": float(np.percentile(sf, 10)) if len(sf) else 0,
                        "p90": float(np.percentile(sf, 90)) if len(sf) else 0},
        "clips": {k: v for k, v in clips.items()},
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    CLIPS.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("A DONE clips=%d gloss=%d span_median=%.1f frames -> %s"
          % (len(sf), len(clips), np.median(sf) if len(sf) else 0, CLIPS), flush=True)
    if a.limit:
        shutil.rmtree(PARTS, ignore_errors=True)   # 冒烟不保留分片


# --------------------------------------------------------------- stage B
def stage_sentences(a):
    labels_tr = P78.read_csv_split("train.csv")
    labels_dv = P78.read_csv_split("dev.csv")
    labels_te = P78.read_csv_split("test.csv")
    seqs = [[t for t in DPM.PreWords(str(g).split("/")) if t]
            for _p, g in labels_tr.values()]
    freq = defaultdict(int)
    for sq in seqs:
        for t in sq:
            freq[t] += 1
    low = {t for t, c in freq.items() if c <= LOW_FREQ_MAX}

    bigram = defaultdict(lambda: defaultdict(int))
    for sq in seqs:
        prev = "<s>"
        for t in sq:
            bigram[prev][t] += 1
            prev = t
        bigram[prev]["</s>"] += 1

    def draw(words):
        tot = sum(words.values())
        r = rng.random() * tot
        acc = 0
        for w, c in words.items():
            acc += c
            if acc >= r:
                return w
        return next(iter(words))

    def canon(sq):
        return "/".join(sq)

    existing = set()
    for lab in (labels_tr, labels_dv, labels_te):
        for _p, g in lab.values():
            existing.add(canon(DPM.PreWords(str(g).split("/"))))

    rng = random.Random(SEED)
    out, tries = [], 0
    max_tries = N_SENT * 50
    while len(out) < N_SENT and tries < max_tries:
        tries += 1
        sq, prev = [], "<s>"
        for _ in range(MAX_LEN):
            w = draw(bigram[prev])
            if w == "</s>":
                break
            sq.append(w)
            prev = w
        if not (MIN_LEN <= len(sq) <= MAX_LEN):
            continue
        if not any(t in low for t in sq):
            continue
        if canon(sq) in existing:
            continue
        out.append(sq)
        existing.add(canon(sq))
    res = {
        "tag": "p231-k1", "n": len(out), "target": N_SENT, "tries": tries,
        "seed": SEED, "length_range": [MIN_LEN, MAX_LEN],
        "low_freq_max": LOW_FREQ_MAX,
        "generator": "train gloss bigram + low-frequency priority (local substitute for LLM; evidence level [无])",
        "leakage": "exact-sequence duplicates with train/dev/test excluded; dev/test used ONLY for duplicate check",
        "sentences": out,
    }
    SENTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("B DONE n=%d / target=%d tries=%d -> %s"
          % (len(out), N_SENT, tries, SENTS), flush=True)
    if len(out) < N_SENT:
        raise RuntimeError("生成句不足 %d（实际 %d），按预注册不得补，链停止" % (N_SENT, len(out)))


# --------------------------------------------------------------- stage C
def stage_synth(a):
    clips = json.loads(CLIPS.read_text(encoding="utf-8"))["clips"]
    sents = json.loads(SENTS.read_text(encoding="utf-8"))["sentences"]
    rng = random.Random(SEED)
    root = P78.RGB_ROOT / "train" / SYNTH_TR
    if root.exists():
        shutil.rmtree(root)          # 只清 K1 自己的合成目录
    root.mkdir(parents=True, exist_ok=True)
    manifest, n_missing, n_frames = {}, 0, 0
    for i, sq in enumerate(sents):
        if any(g not in clips or not clips[g] for g in sq):
            n_missing += 1
            continue
        sid = "SYNTH%06d" % i
        d = root / sid
        d.mkdir(parents=True, exist_ok=True)
        idx = 1
        for g in sq:
            c = rng.choice(clips[g])
            src_dir = P78.RGB_ROOT / "train" / c[0]
            files = sorted(f for f in os.listdir(src_dir) if f.endswith(".jpg"))
            for f in files[c[1]:c[2]]:
                os.link(str(src_dir / f), str(d / ("%06d.jpg" % idx)))
                idx += 1
        n_frames += idx - 1
        manifest[sid] = [SYNTH_TR, "/".join(sq)]
    res = {"tag": "p232-k1", "n": len(manifest), "n_missing_clip": n_missing,
           "n_frames_linked": n_frames, "seed": SEED,
           "link": "hard link (no pixel copy)", "manifest": manifest}
    MANIFEST.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("C DONE samples=%d missing=%d frames=%d -> %s"
          % (len(manifest), n_missing, n_frames, MANIFEST), flush=True)
    if len(manifest) < N_SENT * 0.9:
        raise RuntimeError("可用合成样本不足 90%%，按预注册停止（实际 %d）" % len(manifest))


# --------------------------------------------------------------- stage report
def stage_report(a):
    jp = M / "p233-k1-judge.json"
    if not jp.exists():
        raise RuntimeError("judge 收据不存在，不得生成报告")
    j = json.loads(jp.read_text(encoding="utf-8"))
    lines = []
    for k, v in j.get("checks", {}).items():
        lines.append("| %s | %s |" % (k, "✓" if v else "✗"))
    checks = "\n".join(lines) if lines else "（收据未含 checks 明细，见 json）"
    concl = ("**PASS**（R1–R8 全过）⇒ 仅代表值得 2 seed 复跑，不进 test"
             if j.get("verdict") == "PASS" else
             "**FAIL** ⇒ 按预注册不调比例、不换生成器、不重跑、不进 test；"
             "累计 15 连败下接受天花板收尾")
    txt = """# P233 / K1 语料合成 — 阶段报告

> 日期：%s｜判定：**%s**｜预注册：`docs/planning/K1-PREREG-2026-11-08.md`（20:45 写死）
> 上位方案：`OPT-DIRECTIONS-6-2026-11-08.md`（J1 片段库前提已过线）

## 1. 做了什么
本项目**第一次改训练数据的组织方式**：train 全量 CTC Viterbi 强制对齐切出
per-gloss 片段库 -> 统计 bigram + 低频优先生成 %d 条新 gloss 句 ->
随机取片段硬链接拼接成新 RGB 训练样本 -> 按 manifest 注入 `labels_tr`，
其余逐项沿用 P184 配方 6 轮 PEFT。**模型/损失/解码/评估管线零改动**。

## 2. 判据核查（统一 R1–R8）
%s

## 3. 结论
%s

## 4. 诚实边界
- 生成器是**本地统计 bigram**，非文献的 LLM（启动前写死的工程妥协，证据等级【无】）；
  判负时无法区分「语料合成无效」与「统计生成质量不足」。
- 文献收益全部在 SLT（BLEU）测得；CTC gloss 识别的条件独立假设可能使
  「新上下文」收益打折，这是本方向最大的机制风险。
- 用 P184 切分 + 继续 P184 PEFT，存在文献承认的边界偏差自我强化风险。
- 合成样本只来自 train、只进 train；dev/test 目录零改动；合成句与 dev/test
  无 gloss 级精确重复；词表 `wordSetNum` 未变（strict load 校验）。
- 未跑 test；判据/数量/种子/比例/混合方式均未事后修改。

## 5. 依据
【有】arXiv:2606.11925、ACM MM20 arXiv:2010.05264、arXiv:2401.05336、
arXiv:2506.09643、本地 J1；【无】统计 bigram 替代 LLM；
【有】反证据：仅重组原句 −1.34 BLEU-4、合成数据不可用于预训练、L2 边界平滑负收益。

## 6. 产物
`p230_k1_clips.json` / `p231_k1_sentences.json` / `p232_k1_manifest.json` /
`p233-k1-judge.json`；日志 `artifacts/logs_P233_k1_train.log`。
""" % (datetime.now().strftime("%Y-%m-%d %H:%M"), j.get("verdict", "?"),
       N_SENT, checks, concl)
    for base in (REPO, DESK):
        p = base / "docs/experiments/stage-P233-K1-corpus-synthesis.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> stage-P233-K1-corpus-synthesis.md (+桌面镜像)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["align", "sentences", "synth", "report"])
    ap.add_argument("--ckpt", default="artifacts/checkpoints/p184-tfnet-freeze-best.pt")
    ap.add_argument("--limit", type=int, default=0, help=">0 时 A 阶段冒烟只取前 N 句")
    a = ap.parse_args()
    {"align": stage_align, "sentences": stage_sentences,
     "synth": stage_synth, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
