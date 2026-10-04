# -*- coding: utf-8 -*-
"""P8 · WER 分解：错误归因到词频与 OOV。

回答 FYP 里最该讲清的问题：WER 0.8506 到底错在哪。
把 dev 的编辑错误按三轴切开：
  1. 词频段（是否高频词）
  2. OOV（是否在 cap300 词表内）
  3. 错误类型（替换/插入/删除）

只读 dev，不训练，不触碰 test。
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# `levenshtein` / `align_ops` 是纯 numpy 函数，可脱离 cslr 单独测试
# （tests/test_error_attribution.py 在 Windows 侧导入本模块时，仓库的
#  cslr 包不在 sys.path 上）。故 cslr 依赖改为容错导入。
_CSLR_IMPORT_ERROR = None
try:
    from cslr.recognition.model import CTCRecognizer, ctc_config_from_dict
    from cslr.recognition.gloss_sequence import (
        build_ordered_vocabulary, GlossVocabulary, GlossSequenceConfig,
    )
    from cslr.recognition.dataset import GlossSequenceDataset, FeatureNormalizer
    from cslr.recognition.training import (
        iterate_batches, to_torch_batch, resolve_device, decode_batch,
    )
    from cslr.recognition.dataset import collate_samples
except ModuleNotFoundError as exc:
    _CSLR_IMPORT_ERROR = exc


def read_split(split):
    fname = "dev.csv" if split == "validation" else "train.csv"
    p = REPO / "data/raw/CE-CSL/label" / fname
    with open(p, newline="", encoding="utf-8") as f:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(f)}


def ensure_link_dir(split):
    root = REPO / ".vocab_link" / split
    root.mkdir(parents=True, exist_ok=True)
    src_dir = REPO / "artifacts/part3_features" / split
    for p in sorted(src_dir.glob("*.landmark.npy")):
        sid = p.name[: -len(".landmark.npy")]
        dst = root / (sid + ".npy")
        if not dst.exists():
            try:
                os.symlink(p, dst)
            except OSError:
                pass
    return root


def levenshtein(ref, hyp):
    """标准编辑距离 + 带回溯的操作类型计数。

    返回 (dist, n_sub, n_ins, n_del)
    - sub: 替换（同时计 1 del + 1 ins）
    - ins:  纯插入（hyp 多出）
    - del:  纯删除（ref 有而 hyp 缺）
    """
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=match 1=sub 2=del 3=ins
    for i in range(1, n + 1):
        d[i, 0] = i
        bt[i, 0] = 2
    for j in range(1, m + 1):
        d[0, j] = j
        bt[0, j] = 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            best = d[i - 1, j - 1] + cost
            op = 1 if cost else 0
            if d[i - 1, j] + 1 < best:
                best = d[i - 1, j] + 1
                op = 2
            if d[i, j - 1] + 1 < best:
                best = d[i, j - 1] + 1
                op = 3
            d[i, j] = best
            bt[i, j] = op
    # 回溯
    i, j = n, m
    n_sub = n_ins = n_del = 0
    while i > 0 or j > 0:
        op = bt[i, j]
        if op == 0:
            i -= 1; j -= 1
        elif op == 1:
            n_sub += 1; i -= 1; j -= 1
        elif op == 2:
            n_del += 1; i -= 1
        else:
            n_ins += 1; j -= 1
    return int(d[n, m]), n_sub, n_ins, n_del


def align_ops(ref, hyp):
    """与 ref 逐 token 对齐的操作列表（回溯版）。

    返回 (ops, dist)：
    - ops 长度 == len(ref)，元素 ∈ {0: match, 1: sub, 2: del}
    - hyp 多出的部分（纯插入）不计入 ops，只体现在 dist 里

    恒等式（测试已验证）：sum(o != 0 for o in ops) + n_ins == dist
    """
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        d[i, 0] = i
        bt[i, 0] = 2
    for j in range(1, m + 1):
        d[0, j] = j
        bt[0, j] = 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            best = d[i - 1, j - 1] + cost
            op = 1 if cost else 0
            if d[i - 1, j] + 1 < best:
                best = d[i - 1, j] + 1
                op = 2
            if d[i, j - 1] + 1 < best:
                best = d[i, j - 1] + 1
                op = 3
            d[i, j] = best
            bt[i, j] = op
    ops = [0] * n
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i, j]
        if op == 0:
            i -= 1
            j -= 1
        elif op == 1:
            ops[i - 1] = 1
            i -= 1
            j -= 1
        elif op == 2:
            ops[i - 1] = 2
            i -= 1
        else:
            j -= 1
    return ops, int(d[n, m])


def freq_bands(voc, tr_labels, nrm=None):
    """按 train 频次把**真实 token** 分成 4 段。

    刻意排除 `<unk>`：它是 encode 时给 OOV 打的占位符，不是真实词。
    若不排除，OOV token 会因为「在词表内」而被误分到某个频段，
    导致 OOV 归因失效（P8 首版就是这么错的）。
    """
    cnt = Counter()
    for g in tr_labels.values():
        for t in g.split("/"):
            if t.strip():
                cnt[t.strip()] += 1
    vset = set(voc.tokens) - {"<unk>"}
    bands = [(">=100", 100, 10**9), ("20-99", 20, 99),
             ("5-19", 5, 19), ("1-4", 1, 4)]
    assign = {}
    for name, lo, hi in bands:
        for t, c in cnt.items():
            if lo <= c <= hi and t in vset:
                assign[t] = name
    return assign, cnt


def main():
    if _CSLR_IMPORT_ERROR is not None:
        raise SystemExit(
            "cslr 包不可用（{}）。本脚本必须在仓库 venv 下、"
            "或已把 <repo>/src 加入 PYTHONPATH 时运行。".format(_CSLR_IMPORT_ERROR))
    from cslr.contracts import SampleRecord

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/ctc-landmark48-cap300.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p8-error-attribution.json")
    a = ap.parse_args()

    device = resolve_device("auto")
    print("device = {}".format(device))

    tr_labels = read_split("train")
    va_labels = read_split("validation")
    va_root = ensure_link_dir("validation")
    tr_root = ensure_link_dir("train")

    payload = torch.load(REPO / a.ckpt, map_location="cpu", weights_only=False)
    cfg = ctc_config_from_dict(payload["model_config"])
    model = CTCRecognizer(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    vc = payload.get("vocabulary_config") or {}
    voc = GlossVocabulary(tokens=tuple(payload["vocabulary"]),
                          counts=dict(payload.get("vocabulary_counts") or {}),
                          config=GlossSequenceConfig(**vc) if vc else GlossSequenceConfig())
    nrm_raw = payload.get("feature_normalizer")
    nrm = FeatureNormalizer(mean=np.asarray(nrm_raw["mean"], np.float32),
                            std=np.asarray(nrm_raw["std"], np.float32) + 1e-8)
    print("词表 {} 类  input_size {}".format(voc.size, cfg.input_size))

    recs = []
    for sid, g in va_labels.items():
        if not (va_root / (sid + ".npy")).exists():
            continue
        if not [t for t in g.split("/") if t.strip()]:
            continue
        recs.append(SampleRecord(sample_id=sid, video=Path(sid + ".mp4"), label=g,
                                 signer=sid.split("-")[0], session="x", split="validation"))
    recs.sort(key=lambda r: r.sample_id)
    print("dev 样本 {}".format(len(recs)))

    ds = GlossSequenceDataset(recs, va_root, voc, nrm, feature_view="full")
    band_of, train_cnt = freq_bands(voc, tr_labels, nrm)
    # 真实词集合：排除 `<unk>` 占位符，否则 OOV 会被当成词表内词。
    # P8 首版就是因为漏了这一条，864 个 OOV 全部被误判为 in_vocab。
    vset = set(voc.tokens) - {"<unk>"}

    # ⚠️ ref 必须用**原始 CSV 标签**，不能用 batch 里的 b["tokens"]。
    # GlossVocabulary.encode 会把 OOV token 替换成 `<unk>`（词表里有 unk）
    # 且**长度不变**，所以 b["tokens"] 里根本判不出 OOV —— 实测 864/2842
    # 个 OOV 会全部退化成 "in_vocab"，OOV/词汇内分桶彻底失效。
    # （b["tokens"] 与 CSV 标签 token 数差 4：2838 vs 2842，是 CSV 里
    #  有 4 个 token 在 encode 后被丢弃，属另一处口径差，暂不影响分桶。）
    ref_by_id = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()]
                 for r in recs}

    pairs = []  # (sample_id, ref_tokens_from_csv, hyp_tokens)
    for samples in iterate_batches(ds, a.batch, shuffle=False, seed=a.seed):
        b = to_torch_batch(collate_samples(samples), device)
        with torch.no_grad():
            logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        dec, _, _ = decode_batch(lp, ol.cpu().tolist(), 1)
        for row, ids in enumerate(dec):
            sid = samples[row].sample_id
            # hyp 侧的 `<unk>` 是模型预测出的 unk 类，是真实的预测错误，
            # 不能从列表里删掉（删掉会改变编辑距离）。但它不该被当成
            # 「模型说了某个具体词」来统计 —— 在频段归因时按 OOV 处理。
            hyp = voc.decode(ids)
            pairs.append((sid, ref_by_id[sid], hyp))

    # ---- 汇总 ----
    tot_ref = sum(len(r) for _, r, _ in pairs)
    tot_sub = tot_ins = tot_del = 0
    sent_ok = 0
    wrong_tokens = Counter()   # 出现在 hyp 但不在 ref 的 token
    missed_tokens = Counter()  # 出现在 ref 但不在 hyp 的 token

    for sid, ref, hyp in pairs:
        d, ns, ni, nd = levenshtein(ref, hyp)
        tot_sub += ns; tot_ins += ni; tot_del += nd
        for t in set(ref) - set(hyp):
            missed_tokens[t] += 1
        for t in set(hyp) - set(ref):
            wrong_tokens[t] += 1
        if d == 0:
            sent_ok += 1

    band_stat = defaultdict(lambda: {"n": 0, "match": 0, "sub": 0, "del": 0})
    oov_tok_stat = {"in_vocab": {"n": 0, "err": 0}, "has_oov": {"n": 0, "err": 0}}
    tot_tok_err = 0
    for sid, ref, hyp in pairs:
        ops, d = align_ops(ref, hyp)
        for t, op in zip(ref, ops):
            key = band_of.get(t, "OOV" if t not in vset else "vocab_only")
            st = band_stat[key]
            st["n"] += 1
            if op == 0:
                st["match"] += 1
            else:
                st[ {1: "sub", 2: "del"}[op] ] += 1
            tot_tok_err += (1 if op else 0)
            k = "has_oov" if t not in vset else "in_vocab"
            oov_tok_stat[k]["n"] += 1
            if op:
                oov_tok_stat[k]["err"] += 1

    print("\n" + "=" * 70)
    print("P8 WER 错误归因（dev {} 条 / {} token）".format(len(pairs), tot_ref))
    print("=" * 70)
    print("完全正确句子: {} ({:.1%})".format(sent_ok, sent_ok / len(pairs)))
    print("编辑距离分解: sub={} ins={} del={}  总计={}".format(
        tot_sub, tot_ins, tot_del, tot_sub + tot_ins + tot_del))
    print("整体 WER = {}".format(
        (tot_sub + tot_ins + tot_del) / max(tot_ref, 1)))

    print("\n--- 按词频段（逐 token 精确对齐）---")
    print("频段        token   正确率    错误数   贡献占全部错误")
    rows = []
    for key in (">=100", "20-99", "5-19", "1-4", "OOV", "vocab_only"):
        st = band_stat.get(key)
        if not st or st["n"] == 0:
            continue
        err = st["sub"] + st["del"]
        acc = st["match"] / st["n"]
        share = err / max(tot_tok_err, 1)
        rows.append({"band": key, "n_tokens": st["n"], "accuracy": float(acc),
                     "errors": err, "error_share": float(share),
                     "sub": st["sub"], "del": st["del"]})
        print("  {:<8}  {:>5}   {:>6.1%}   {:>6}   {:>6.1%}".format(
            key, st["n"], acc, err, share))

    print("\n--- 按 OOV / 词汇内（逐 token）---")
    for key in ("in_vocab", "has_oov"):
        st = oov_tok_stat[key]
        if st["n"] == 0:
            continue
        acc = 1 - st["err"] / st["n"]
        share = st["err"] / max(tot_tok_err, 1)
        print("  {:<9} token={:>5}  正确率={:>6.1%}  错误={:>5}  占全部错误 {:>6.1%}".format(
            key, st["n"], acc, st["err"], share))

    print("\n--- 最常被漏掉的词（ref 有 hyp 无，top 15）---")
    for t, c in missed_tokens.most_common(15):
        print("  {:<10} 漏 {} 次   训练频次 {}".format(
            t, c, train_cnt.get(t, 0)))
    print("\n--- 最常被凭空输出的词（hyp 有 ref 无，top 15）---")
    for t, c in wrong_tokens.most_common(15):
        print("  {:<10} 多 {} 次   训练频次 {}".format(t, c, train_cnt.get(t, 0)))

    receipt = {
        "ckpt": a.ckpt,
        "n_samples": len(pairs),
        "n_ref_tokens": tot_ref,
        "sent_exact_match": sent_ok,
        "sent_exact_rate": float(sent_ok / len(pairs)),
        "edit_breakdown": {"sub": tot_sub, "ins": tot_ins, "del": tot_del},
        "wer": float((tot_sub + tot_ins + tot_del) / max(tot_ref, 1)),
        "by_frequency_band": rows,
        "by_oov": {k: {"n_tokens": v["n"], "errors": v["err"],
                        "accuracy": float(1 - v["err"] / v["n"])}
                   for k, v in oov_tok_stat.items() if v["n"]},
        "top_missed": [{"token": t, "count": c, "train_freq": int(train_cnt.get(t, 0))}
                       for t, c in missed_tokens.most_common(25)],
        "top_spurious": [{"token": t, "count": c, "train_freq": int(train_cnt.get(t, 0))}
                         for t, c in wrong_tokens.most_common(25)],
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
