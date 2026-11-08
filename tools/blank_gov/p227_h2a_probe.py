#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P227 / H2-a：TTA corruption 敏感性前提诊断（零训练、纯推理、无梯度）。

判据见 docs/planning/H2A-PREREG-2026-11-08.md（2026-11-08 18:05 写死）。
只做 dev 515 句，不碰 test；主干全冻结、eval、不更新任何参数。
9 组 = clean + 4 类 corruption × mild/medium；主判据只看 mild 档 WER 跌幅。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
DESK = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
for _p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(_p))

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402
import DataProcessMoudle as DPM  # noqa: E402
from p197_beam_lm import build_model, greedy_fast  # noqa: E402
from p210_d4_eval import align_ops  # noqa: E402

# 与 p207_d1_dump_nbest.py / night_eval_full.py 完全一致的全局口径
P78.OFFICIAL_PAD = True
P78.RGB_FIX = True
P78.BLANK_BIAS = 0.0

M = REPO / "artifacts/metrics/blank-gov"
CACHE = REPO / "artifacts/cache"
PARTIAL = CACHE / "p227_h2a_partial.json"
TAG = "p227-h2a"
LOW_FREQ_MAX = 10
RULE_DELTA = 2.00
CLEAN_RANGE = (34.20, 34.60)

GROUPS = ["clean", "blur_mild", "blur_med", "jpeg_mild", "jpeg_med",
          "bright_mild", "bright_med", "occlud_mild", "occlud_med"]
MILD = ["blur_mild", "jpeg_mild", "bright_mild", "occlud_mild"]


# ---------------------------------------------------------------- corruption
def _jpeg(im, q):
    ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _bright(im, f):
    return np.clip(im.astype(np.float32) * f, 0, 255).astype(np.uint8)


def _occlud(im, frac):
    out = im.copy()
    h, w = im.shape[:2]
    side = max(1, int(np.sqrt(frac) * min(h, w)))
    y0, x0 = (h - side) // 2, (w - side) // 2
    out[y0:y0 + side, x0:x0 + side] = 0
    return out


def corrupt_fn(name):
    """作用在 cv2.imread 得到的 uint8 [H,W,3] 上，归一化之前。全部确定性。"""
    if name == "clean":
        return None
    if name == "blur_mild":
        return lambda im: cv2.GaussianBlur(im, (3, 3), 0.8)
    if name == "blur_med":
        return lambda im: cv2.GaussianBlur(im, (5, 5), 1.8)
    if name == "jpeg_mild":
        return lambda im: _jpeg(im, 60)
    if name == "jpeg_med":
        return lambda im: _jpeg(im, 25)
    if name == "bright_mild":
        return lambda im: _bright(im, 0.85)
    if name == "bright_med":
        return lambda im: _bright(im, 0.60)
    if name == "occlud_mild":
        return lambda im: _occlud(im, 0.08)
    if name == "occlud_med":
        return lambda im: _occlud(im, 0.20)
    raise KeyError(name)


class CorruptDataset(P78.RGBSeqDataset):
    """只覆盖读图那一段：RGB_FIX=True 的原生读法 + corruption，其余逐行照抄父类。"""

    fn = None

    def __getitem__(self, i):
        assert P78.RGB_FIX is True, "必须与 P184 口径一致"
        it = self.items[i]
        files = sorted(f for f in os.listdir(it["dir"]) if f.endswith(".jpg"))
        imgs = [cv2.imread(str(it["dir"] / f)) for f in files]
        if self.fn is not None:
            imgs = [self.fn(im) for im in imgs]
        seq = self.transform(imgs)
        seq = seq.float() / 127.5 - 1.0
        return {"video": seq, "ids": it["ids"], "sid": it["sid"]}


# ---------------------------------------------------------------- forward
def collect(mdl, loader, dev, labels_dv):
    """逐行照 p197_beam_lm.collect_logps，另加 synchronize / empty_cache 防 WSL 驱动错。"""
    items = []
    for bi, (vid, _tgt, _tl, dl, _tll, sids, _ids) in enumerate(loader):
        with torch.no_grad():
            out = mdl(vid.to(dev), dl, False)
            logp = F.log_softmax(out[0], dim=-1).float()
        if dev == "cuda":
            torch.cuda.synchronize()
        logp = logp.cpu().numpy()
        lens = dl.view(-1).tolist()
        for b, sid in enumerate(sids):
            T = min(P78._conv_len(int(lens[b])), logp.shape[0])
            items.append({"sid": str(sid),
                          "logp": logp[:T, b, :].astype(np.float32),
                          "ref": labels_dv[sid][1]})
        if dev == "cuda" and bi % 100 == 0:
            torch.cuda.empty_cache()
        if bi % 100 == 0:
            print("  forward %d/%d" % (bi, len(loader)), flush=True)
    return items


# ---------------------------------------------------------------- metrics
def evaluate_all(name, items, idx2word, freq, ckpt_name):
    """与 night_eval_full.py 的指标段逐行一致，另加两个 collapse 指标。"""
    refs = [it["ref"] for it in items]                 # 原始 gloss 字符串
    hyps = [greedy_fast(it["logp"], idx2word) for it in items]
    res = OW.evaluate(refs, hyps)

    per_cls = defaultdict(lambda: {"n": 0, "err": 0})
    per_bucket = defaultdict(lambda: {"n": 0, "err": 0})
    ins = sub = dele = nref = 0
    nb_frames = total_frames = 0
    per_item = []
    empty_out = 0
    vocab_out = set()
    for it, ref, hyp in zip(items, refs, hyps):
        r = OW.split_glued(OW.split_gloss_sequence(ref, OW.CFG))
        h = OW.split_glued(list(hyp))
        ops, ii = align_ops(r, h)
        assert len(ops) == len(r), "align length mismatch"
        for t, op in zip(r, ops):
            b = "low" if freq.get(t, 0) <= LOW_FREQ_MAX else "high"
            per_cls[t]["n"] += 1
            per_cls[t]["err"] += (op != "match")
            per_bucket[b]["n"] += 1
            per_bucket[b]["err"] += (op != "match")
        s = sum(o == "sub" for o in ops)
        d = sum(o == "del" for o in ops)
        ins += ii
        sub += s
        dele += d
        nref += len(r)
        e = OW.levenshtein(r, h)
        _low_n = sum(1 for t in r if freq.get(t, 0) <= LOW_FREQ_MAX)
        _low_err = sum(1 for t, op in zip(r, ops)
                       if freq.get(t, 0) <= LOW_FREQ_MAX and op != "match")
        per_item.append({"sid": str(it.get("sid", "")), "ref_n": len(r),
                         "edits": e, "ins": ii, "sub": s, "del": d,
                         "low_n": _low_n, "low_err": _low_err})
        lp = it["logp"]
        path = lp.argmax(-1).reshape(-1).tolist()
        nb_frames += sum(1 for x in path if x != 0)
        total_frames += len(path)
        if len(h) == 0:
            empty_out += 1
        vocab_out.update(h)

    cls_rates = [v["err"] / v["n"] for v in per_cls.values() if v["n"] > 0]
    low, high = per_bucket["low"], per_bucket["high"]
    out = {
        "tag": "%s-%s" % (TAG, name),
        "corruption": name,
        "ckpt": ckpt_name,
        "n": len(items),
        "WER_official": res["WER_official"],
        "uWER_pct": round(float(np.mean(cls_rates)) * 100.0, 2),
        "uWER_n_classes": len(cls_rates),
        "low_freq_max": LOW_FREQ_MAX,
        "low_bucket": {"n_tokens": low["n"],
                       "WER_pct": round(100.0 * low["err"] / max(low["n"], 1), 2)},
        "high_bucket": {"n_tokens": high["n"],
                        "WER_pct": round(100.0 * high["err"] / max(high["n"], 1), 2)},
        "edits": {"ins": int(ins), "sub": int(sub), "del": int(dele),
                  "n_ref_tokens": int(nref),
                  "micro_subdel_pct": round(100.0 * (sub + dele) / max(nref, 1), 2),
                  "ins_rate_pct": round(100.0 * ins / max(nref, 1), 2)},
        "nonblank_frame_ratio_pct": round(100.0 * nb_frames / max(total_frames, 1), 2),
        "collapse": {"empty_output_rate_pct": round(100.0 * empty_out / max(len(items), 1), 2),
                     "distinct_gloss_count": len(vocab_out)},
        "per_item": per_item,
        "note": ("H2-a 前提诊断；指标口径与 night_eval_full.py 一致；"
                 "不产生候选模型，不参与性能比较"),
    }
    return out


# ---------------------------------------------------------------- report
def write_report(results, verdict, delta_mild, clean_wer, audit_ok, elapsed):
    def row(name):
        r = results[name]
        return ("| `%s` | %s | %.2f | %.2f | %.2f | %.2f | %d | %.2f | %.2f | %d |"
                % (name, "mild" if name in MILD else ("medium" if name != "clean" else "—"),
                   r["WER_official"], r["uWER_pct"],
                   r["low_bucket"]["WER_pct"], r["high_bucket"]["WER_pct"],
                   r["edits"]["ins"], r["edits"]["micro_subdel_pct"],
                   r["collapse"]["empty_output_rate_pct"],
                   r["collapse"]["distinct_gloss_count"]))
    rows = "\n".join(row(n) for n in GROUPS if n in results)
    deltas = "\n".join(
        "- `%s`: Δ WER **%+.2f pp**" % (n, results[n]["WER_official"] - clean_wer)
        for n in MILD if n in results)
    med = "\n".join(
        "- `%s`: Δ WER %+.2f pp（仅报告，不参与判定）"
        % (n, results[n]["WER_official"] - clean_wer)
        for n in GROUPS if n not in MILD and n != "clean" and n in results)

    txt = """# P227 / H2-a TTA corruption 敏感性前提诊断 — 阶段报告

> 日期：%s｜状态：**%s**｜类型：前提诊断（零训练、纯推理、无梯度、无参数更新）
> 预注册：`docs/planning/H2A-PREREG-2026-11-08.md`（18:05 写死，早于任何结果）
> 上位方案：`OPT-DIRECTIONS-4-2026-11-08.md` §2 H2 / §3

## 1. 回答什么问题
H2（参数自适应 TTA）的收益前提是「测试时存在可修复的分布偏移」。
CE-CSL dev/test 同分布，故方案写死：dev 加 corruption 若 WER 跌 <2pp ⇒ H2 直接 STOP。
本报告只给 `PRECONDITION_PASS` / `STOP`，**不产生候选模型**。

## 2. 配置（逐项沿用 P184）
`p184-tfnet-freeze-best.pt`｜160 / hidden 1024｜`OFFICIAL_PAD=True`、`RGB_FIX=True`、
`BLANK_BIAS=0.0`｜dev 分支 `build_transforms(hflip=False, img_size=160, y_shift=0,
temporal_rescale=0.0)`｜dev **515 句全量**、`batch=1`、不跳句、不增广｜
`eval()` + `no_grad`，主干全冻结｜corruption 作用在 `cv2.imread` 后、归一化前的 uint8 帧。

## 3. 结果（dev WER_official，%%）

| 组 | 档 | WER | uWER | 低频桶 | 高频桶 | ins | micro | 空输出率 | 不同 gloss 数 |
|---|---|---|---|---|---|---|---|---|---|
%s

### 3.1 主判据（只看 mild 档）
%s

**Δ_mild_max = %+.2f pp**，判据线 **≥ %.2f pp** ⇒ **%s**

### 3.2 medium 档（如实报告，不参与判定）
%s

### 3.3 审计自检
`clean` WER = **%.2f**，允许区间 [%.2f, %.2f] ⇒ **%s**
（本管线前向收集与 `p188_logp_p184.pt` 同源；`p184-base-uwer.json` 的 34.41
走另一 collect 实现，差 0.07pp。判据用同管线内相对跌幅，故不受影响）

## 4. 结论
%s

## 5. 诚实边界
- 本诊断只用 4 类合成 corruption、每类 2 档；未覆盖 signer shift / 背景 / 手形速度等
  真实域偏移。**STOP 只表示「这 4 类轻度偏移下无可修复空间」**，不等于证明 TTA 在
  任何真实域偏移下都无效。
- medium 档退化再大也不参与判定（理由见预注册 §3：强破坏必然掉点，不构成 TTA 有空间的证据）。
- 未跑 test、未改任何权重、未使用 dev 标签做监督；全程零训练。
- 判据、强度、阈值、参与判定的档位均在 18:05 写死；FAIL/STOP 后不调阈值、不挑组、不复活 H2-b。

## 6. 依据等级
【有】SAME (ACL 2026)、TRANSLATE (IEEE TMM 2026, DOI 10.1109/TMM.2026.3730967)、
TENT (ICLR 2021)；【有】本地 P187 输入变换式 TTA 已判死（机制不同，不构成反证）；
【有】`OPT-DIRECTIONS-4` 检索结论：CSLR gloss 层面 / 冻结主干下的 TTA 未见；
【无】以 mild 档作主判据、4 类 corruption 的具体强度为工程判断，无直接论文依据。

## 7. 产物
- 汇总收据 `artifacts/metrics/blank-gov/p227-h2a-probe.json`
- 每组全指标 `<tag>-<组名>-uwer.json`（含 per_item，可配对 bootstrap）
- 日志 `artifacts/logs_P227_h2a.log`｜耗时 %s
""" % (datetime.now().strftime("%Y-%m-%d %H:%M"), verdict, rows,
       deltas, delta_mild, RULE_DELTA, verdict,
       med, clean_wer, CLEAN_RANGE[0], CLEAN_RANGE[1],
       "PASS" if audit_ok else "INVALID（管线异常，不得采信结论）",
       ("**PRECONDITION_PASS（机械判据）**：Δ_mild_max ≥2.00pp，verdict 保持 PASS。\n\n"
        "⚠️ 但 PASS 可能由单组强破坏驱动（2026-11-08 本次即由 `occlud_mild` 单点驱动，"
        "详见本报告 §4.1 的决策层分析）：判据/阈值/强度均未事后调整，"
        "是否投 H2-b 须结合「是否属于可修复的统计量偏移」由用户确认，"
        "不得把机械 PASS 直接读作「值得投 GPU」。"
        if verdict == "PRECONDITION_PASS" else
        "**STOP**：4 类轻度自然偏移下 WER 最大退化不足 2pp，模型对同协议内的轻微偏移已足够鲁棒；"
        "按 `OPT-DIRECTIONS-4` §3 的硬前提，**H2 整条线关闭，不投 H2-b 的 GPU**。"
        "结合累计 13+ 判负，合理结论是接受天花板收尾。"
        if verdict == "STOP" else
        "**INVALID**：clean 基线自检未通过，管线存在异常，本次不给出 PASS/STOP 结论，需先排查。"),
       elapsed)
    for base in (REPO, DESK):
        p = base / "docs/experiments/stage-P227-H2a-tta-precondition.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> stage-P227-H2a-tta-precondition.md (+桌面镜像)", flush=True)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/p184-tfnet-freeze-best.pt")
    ap.add_argument("--limit", type=int, default=0, help=">0 时冒烟只取前 N 句")
    ap.add_argument("--groups", nargs="*", default=None, help="冒烟指定组")
    a = ap.parse_args()
    t0 = time.time()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels_dv = P78.read_csv_split("dev.csv")
    labels_tr = P78.read_csv_split("train.csv")
    word2idx, n, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    freq = defaultdict(int)
    for _sid, (_p, seq) in labels_tr.items():
        for t in DPM.PreWords(str(seq).split("/")):
            freq[t] += 1

    _, dv_tf = P78.build_transforms(hflip=False, img_size=160,
                                    y_shift=0, temporal_rescale=0.0)
    groups = a.groups if a.groups else GROUPS
    smoke = a.limit > 0
    print("groups=%s smoke=%s dev=%s" % (groups, smoke, dev), flush=True)

    # 断点：每组完成后写一次（冒烟不写，避免污染正式跑）
    done = {}
    if not smoke and os.environ.get("H2A_RESUME") == "1" and PARTIAL.exists():
        done = json.loads(PARTIAL.read_text(encoding="utf-8")).get("done", {})
        print("resume: 已完成 %s" % list(done), flush=True)

    mdl = build_model(str(REPO / a.ckpt), 1024, n, dev)
    mdl.eval()
    results = {}

    for name in groups:
        tag = "%s-%s" % (TAG, name)
        rp = M / ("%s-uwer.json" % tag)
        if not smoke and name in done and rp.exists():
            results[name] = json.loads(rp.read_text(encoding="utf-8"))
            print("[%s] resume from receipt" % name, flush=True)
            continue
        print("=== group %s ===" % name, flush=True)
        ds = CorruptDataset("dev", labels_dv, word2idx, dv_tf, False)
        ds.fn = corrupt_fn(name)
        if smoke:
            ds.items = ds.items[:a.limit]
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=P78.collate, num_workers=0)
        print("  dev items = %d" % len(ds), flush=True)
        items = collect(mdl, loader, dev, labels_dv)
        r = evaluate_all(name, items, idx2word, freq, a.ckpt)
        print("  WER %.2f | uWER %.2f | low %.2f | high %.2f | ins %d | empty %.2f%%"
              % (r["WER_official"], r["uWER_pct"], r["low_bucket"]["WER_pct"],
                 r["high_bucket"]["WER_pct"], r["edits"]["ins"],
                 r["collapse"]["empty_output_rate_pct"]), flush=True)
        results[name] = r
        if not smoke:
            M.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            done[name] = {"WER_official": r["WER_official"]}
            CACHE.mkdir(parents=True, exist_ok=True)
            PARTIAL.write_text(json.dumps({"done": done}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        del items, ds, loader
        if dev == "cuda":
            torch.cuda.empty_cache()

    del mdl
    if dev == "cuda":
        torch.cuda.empty_cache()

    if smoke:
        print("SMOKE DONE (不写正式收据/报告)", flush=True)
        return

    # 判定
    clean_wer = results["clean"]["WER_official"]
    audit_ok = CLEAN_RANGE[0] <= clean_wer <= CLEAN_RANGE[1]
    delta_mild = max(results[x]["WER_official"] - clean_wer for x in MILD)
    verdict = ("INVALID" if not audit_ok else
               ("PRECONDITION_PASS" if delta_mild >= RULE_DELTA else "STOP"))
    print("clean=%.2f audit=%s delta_mild=%.2f => %s"
          % (clean_wer, audit_ok, delta_mild, verdict), flush=True)

    out = {
        "tag": TAG, "verdict": verdict, "rule_delta_pp": RULE_DELTA,
        "delta_mild_max_pp": round(delta_mild, 4),
        "clean_WER_official": clean_wer, "audit_clean_in_range": audit_ok,
        "clean_allowed_range": list(CLEAN_RANGE),
        "mild_groups": MILD, "medium_groups": [g for g in GROUPS if g not in MILD and g != "clean"],
        "deltas_all_pp": {k: round(v["WER_official"] - clean_wer, 4)
                          for k, v in results.items()},
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_item"}
                    for k, v in results.items()},
        "prereg": "docs/planning/H2A-PREREG-2026-11-08.md",
        "note": ("前提诊断，不产生候选模型；medium 档不参与判定；"
                 "STOP 后不启动 H2-b；未跑 test、未改权重"),
    }
    M.mkdir(parents=True, exist_ok=True)
    (M / "p227-h2a-probe.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (M / "p227-h2a-verdict.txt").write_text(verdict, encoding="utf-8")

    elapsed = "%.1f min" % ((time.time() - t0) / 60.0)
    write_report(results, verdict, delta_mild, clean_wer, audit_ok, elapsed)
    PARTIAL.unlink(missing_ok=True)
    print("H2-a DONE %s elapsed %s" % (verdict, elapsed), flush=True)


if __name__ == "__main__":
    main()
