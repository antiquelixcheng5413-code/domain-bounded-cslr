#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P228 / I0：推理期输入时间分辨率扫描（零训练、参数无关、纯推理、fp32）。

判据见 docs/planning/I0-PREREG-2026-11-08.md（19:05 写死）。
四档 step2 / step1(基线) / repeat2 / step3，dev 515 句全量，主干全冻结。
指标计算复用 p227_h2a_probe.evaluate_all（与 night_eval_full.py 逐行一致）。
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import shutil
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
for p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(p))

import p78_train_official_tfnet as P78  # noqa: E402
import official_wer as OW  # noqa: E402
import DataProcessMoudle as DPM  # noqa: E402
from p197_beam_lm import build_model  # noqa: E402
from p227_h2a_probe import collect, evaluate_all  # noqa: E402

M = REPO / "artifacts/metrics/blank-gov"
CACHE = REPO / "artifacts/cache"
PARTIAL = CACHE / "p228_i0_partial.json"
TAG = "p228-i0"
GROUPS = ["step2", "step1", "repeat2", "step3"]
BASE = "step1"
CANDIDATES = ["step2", "repeat2", "step3"]
RULE_DELTA = 2.00
CLEAN_RANGE = (34.20, 34.60)
N_BOOT = 20000
SEED = 42


def resample_frames(imgs, mode):
    """确定性帧序列重采样，作用在 cv2.imread 得到的 uint8 帧列表上。"""
    if mode == "step1":
        return imgs
    if mode == "step2":
        return imgs[::2]
    if mode == "step3":
        return imgs[::3]
    if mode == "repeat2":
        return [im for im in imgs for _ in (0, 1)]
    raise KeyError(mode)


class ResampleDataset(P78.RGBSeqDataset):
    """只覆盖读图段：帧序列重采样 + 短序列补帧；其余逐行照父类（含 /127.5-1）。"""

    mode = "step1"
    n_padded = 0

    def __getitem__(self, i):
        assert P78.RGB_FIX is True, "必须与 P184 口径一致"
        it = self.items[i]
        files = sorted(f for f in os.listdir(it["dir"]) if f.endswith(".jpg"))
        imgs = [cv2.imread(str(it["dir"] / f)) for f in files]
        imgs = resample_frames(imgs, self.mode)
        if len(imgs) < P78.MIN_FRAMES:
            ResampleDataset.n_padded += 1
            imgs = imgs + [imgs[-1]] * (P78.MIN_FRAMES - len(imgs))
        seq = self.transform(imgs)
        seq = seq.float() / 127.5 - 1.0
        return {"video": seq, "ids": it["ids"], "sid": it["sid"]}


def collect_resumable(mdl, loader, dev, labels_dv, group, chunk=25):
    """逐行照 p227_h2a_probe.collect，另加**组内分片断点**：每 chunk 句写一个小分片。

    纯工程加固（应对 WSL 驱动在 repeat2 长样本后累积性崩溃）：
    不改档位/精度/判据/句序；重启按序加载分片，最多重跑 chunk-1 句。
    分片而非整表反复 dump：repeat2 全部后验约 670MB，整表写 20 次不可接受。
    """
    pdir = CACHE / ("p228_i0_%s_parts" % group)
    pdir.mkdir(parents=True, exist_ok=True)
    parts = sorted(pdir.glob("part*.pkl"))
    items = []
    for pp in parts:
        with open(pp, "rb") as f:
            items.extend(pickle.load(f))
    seen = len(items)
    if seen:
        print("  [%s] 组内分片恢复 %d 片 / items=%d"
              % (group, len(parts), seen), flush=True)
    part_no = len(parts)
    buf = []
    for bi, (vid, _tgt, _tl, dl, _tll, sids, _ids) in enumerate(loader):
        if bi < seen:
            continue                       # 保留 dataset 顺序，只跳过前向
        with torch.no_grad():
            out = mdl(vid.to(dev), dl, False)
            logp = F.log_softmax(out[0], dim=-1).float()
        if dev == "cuda":
            torch.cuda.synchronize()
        logp = logp.cpu().numpy()
        lens = dl.view(-1).tolist()
        for b, sid in enumerate(sids):
            T = min(P78._conv_len(int(lens[b])), logp.shape[0])
            buf.append({"sid": str(sid),
                        "logp": logp[:T, b, :].astype(np.float32),
                        "ref": labels_dv[sid][1]})
        del out, logp, vid
        if len(buf) >= chunk:
            with open(pdir / ("part%04d.pkl" % part_no), "wb") as f:
                pickle.dump(buf, f, protocol=4)
            items.extend(buf)
            buf = []
            part_no += 1
        if dev == "cuda" and (bi + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if bi % 100 == 0:
            print("  forward %d/%d items=%d"
                  % (bi, len(loader), len(items) + len(buf)), flush=True)
    if buf:
        with open(pdir / ("part%04d.pkl" % part_no), "wb") as f:
            pickle.dump(buf, f, protocol=4)
        items.extend(buf)
    return items, pdir


def align_stats(items):
    """每词编码器时间步、每词非 blank 帧数（解释性，不参与判定）。"""
    tl, nbpg = [], []
    for it in items:
        L = len(OW.split_glued(OW.split_gloss_sequence(it["ref"], OW.CFG)))
        path = it["logp"].argmax(-1)
        tl.append(it["logp"].shape[0] / max(L, 1))
        nbpg.append(float((path != 0).sum()) / max(L, 1))
    return {
        "T_conv_per_gloss_median": round(float(np.median(tl)), 2),
        "T_conv_per_gloss_mean": round(float(np.mean(tl)), 2),
        "nonblank_frames_per_gloss_mean": round(float(np.mean(nbpg)), 3),
    }


def paired_bootstrap(base, cand):
    """Δ = 100*(Σbase_edits − Σcand_edits)/Σref_n；配对逐句重采样。"""
    b = np.asarray([x["edits"] for x in base["per_item"]], float)
    c = np.asarray([x["edits"] for x in cand["per_item"]], float)
    n = np.asarray([x["ref_n"] for x in base["per_item"]], float)
    bs = [str(x.get("sid", "")) for x in base["per_item"]]
    cs = [str(x.get("sid", "")) for x in cand["per_item"]]
    assert len(b) == len(c) == len(n) == 515, "配对样本数不为 515"
    if all(bs) and all(cs):
        assert bs == cs, "配对 bootstrap：sid 必须逐句同序"
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(n), size=(N_BOOT, len(n)))
    deltas = 100.0 * (b[idx].sum(1) - c[idx].sum(1)) / n[idx].sum(1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return (float(100.0 * (b.sum() - c.sum()) / n.sum()),
            float(lo), float(hi))


def write_report(results, checks, verdict, base_wer, audit_ok,
                 representative, elapsed, i1):
    lines = []
    lines.append("| 档 | WER | Δ vs step1 | uWER | 低频桶 | 高频桶 | ins | sub | del | micro | 非blank帧% | T_conv/L | 每词非blank帧 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in GROUPS:
        r = results[g]
        e = r["edits"]
        lines.append("| `%s` | %.2f | %+.2f | %.2f | %.2f | %.2f | %d | %d | %d | %.2f | %.2f | %.2f | %.3f |"
                     % (g, r["WER_official"], base_wer - r["WER_official"],
                        r["uWER_pct"], r["low_bucket"]["WER_pct"],
                        r["high_bucket"]["WER_pct"], e["ins"], e["sub"], e["del"],
                        e["micro_subdel_pct"], r["nonblank_frame_ratio_pct"],
                        r["T_conv_per_gloss_median"],
                        r["nonblank_frames_per_gloss_mean"]))
    table = "\n".join(lines)

    crows = []
    for g in CANDIDATES:
        c = checks[g]
        crows.append("| `%s` | %+.2f pp | %s | [%.3f, %.3f] | %s | %s | %s | %s | %s | **%s** |"
                     % (g, c["delta_pp"], "✓" if c["R1_delta"] else "✗",
                        c["ci95"][0], c["ci95"][1],
                        "✓" if c["R2_ci"] else "✗",
                        "✓" if c["R3_high"] else "✗",
                        "✓" if c["R5_ins"] else "✗",
                        "✓" if c["R6_micro"] else "✗",
                        "✓" if c["R7_low_bucket"] else "✗",
                        "PASS" if c["pass"] else "FAIL"))
    ctable = "\n".join(crows)

    i1txt = ("I1 `rho(T/L, del_rate) = %.4f`，bootstrap CI `[%.3f, %.3f]` ⇒ **%s**；"
             "低/高 T/L 半区删除率 %.2f%% / %.2f%%。"
             % (i1["rho_TL_vs_del_rate"], i1["ci95"][0], i1["ci95"][1],
                i1["verdict"], i1["buckets"]["low_TL_half"]["del_rate_pct"],
                i1["buckets"]["high_TL_half"]["del_rate_pct"])) if i1 else "I1 收据未生成。"

    if verdict == "SIGNAL":
        concl = ("**SIGNAL**：`%s` 档过线（Δ %+.2f pp，CI 下界 >0，三项护栏全过）。"
                 "按预注册 ⇒ 值得进入 2 seed 复跑；**仍不等于确认提升，也不进 test**。"
                 "I2（训练期对齐增密）须另写预注册后才可启动。"
                 % (representative, checks[representative]["delta_pp"]))
    elif verdict == "INVALID":
        concl = "**INVALID**：`step1` 审计自检未通过，管线异常，不给 SIGNAL/FAIL，需先排查。"
    else:
        concl = ("**FAIL**：三档均未同时满足 Δ≥2.00pp、CI 下界>0 与三项护栏 ⇒ "
                 "**输入时间分辨率这条线判死**，不投 I2、不进 test、不调档位重跑。"
                 "按累计 13+ 判负与 `OPT-DIRECTIONS-5` §6，合理结论是接受天花板收尾。")

    txt = """# P228 / I0 推理期输入时间分辨率扫描 — 阶段报告

> 日期：%s｜状态：**%s**｜类型：零训练、参数无关、纯推理、fp32、无梯度
> 预注册：`docs/planning/I0-PREREG-2026-11-08.md`（19:05 写死，早于任何结果）
> 上位方案：`OPT-DIRECTIONS-5-2026-11-08.md`

## 1. 回答什么问题
方案 §3 给出两个相反假设：**H_low**（输入帧/词 32.8 偏高 ⇒ `step2` 应改善）与
**H_high**（`T_conv/L` 8.50、每词 1.06 非 blank 帧 ⇒ `repeat2` 应改善）。
本扫描不预设立场，四档同时覆盖，由数据裁决。

## 2. 配置（逐项沿用 P184 / H2-a）
`p184-tfnet-freeze-best.pt`｜160 / hidden1024｜`OFFICIAL_PAD=True`、`RGB_FIX=True`、
`BLANK_BIAS=0.0`｜dev 分支 transform｜**dev 515 句全量**、`batch=1`、不跳句、不增广｜
`eval()`+`no_grad`+**fp32**、主干全冻结｜帧重采样作用在 `cv2.imread` 后、`transform` 前。
重采样后 <48 帧的样本重复末帧补到 `MIN_FRAMES`（确定性，各组 n_padded 见收据）。

## 3. 结果（dev WER_official，%%）

%s

`step1` 审计自检：WER **%.2f** ∈ [%.2f, %.2f] ⇒ **%s**
（已知 p188/p197 同源 fp32 = 34.34；`p184-base-uwer.json` 34.41 走 p207 collect，差 0.07pp）

### 3.1 判据逐档核查

| 档 | Δ | R1 ≥2.00pp | 配对 CI 95%% | R2 CI下界>0 | R3 高频桶 | R5 插入 | R6 micro | R7 低频桶 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
%s

> 另有 (6) 低频桶 Δ ≥ −1.00pp 护栏，逐档取值见 `p228-i0-probe.json`。
> 配对 bootstrap 20000 次、双方 sid 逐句同序断言；多档过线取 Δ 最大者。

### 3.2 I1 解释性诊断（不改变 I0 判定）
%s

## 4. 结论
%s

## 5. 诚实边界
- `repeat2` 的**重复帧不增加视觉信息**，只增加 CTC 可发射时间位置；
  不得把其结果解读为「数据增广有效」。
- 文献的 17/18/58 是**输入帧/词**、本地 8.50 是**编码器时间步/词**，两口径不可直接比大小。
- 本地主矛盾是替换 67.5%%，删除仅 24.5%%；帧密度主要作用于对齐/删除，收益上限有限。
  MTCNet 的「deletion 主导 54.3%%」前提在本地不成立。
- 全部 4 档共用同一 collect / evaluate 实现，横向可比；判据是同管线内相对 Δ。
- 未跑 test、未改任何权重、未用 dev 标签做监督；判据/档位/阈值/失败处理均在 19:05 写死，
  未事后增删或调整（预注册 §7 禁止）。

## 6. 依据等级
【有】ICCV2025W（downsampling ratio 0.5 最优）、SignFlow IEEE Access25（6fps、
rapid signing→deletion）、RAIDAL BMVC26、arXiv:2406.02560；
【有】本地 D3 / H2-a / P191 / P120；【间接】Zhao&Bell SLT24；
【无】推理期帧步长扫描无先例、U 型曲线属推导、四档选择属工程判断。

## 7. 产物
- `artifacts/metrics/blank-gov/p228-i0-probe.json`（汇总 + 逐档 checks + verdict）
- 每档 `p228-i0-<档>-uwer.json`（含 per_item）、`p228-i1-correlation.json`
- 日志 `artifacts/logs_P228_i0.log`｜耗时 %s
""" % (datetime.now().strftime("%Y-%m-%d %H:%M"), verdict, table,
       base_wer, CLEAN_RANGE[0], CLEAN_RANGE[1],
       "PASS" if audit_ok else "INVALID", ctable, i1txt, concl, elapsed)

    for base in (REPO, DESK):
        p = base / "docs/experiments/stage-P228-I0-temporal-resolution.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> stage-P228-I0-temporal-resolution.md (+桌面镜像)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/p184-tfnet-freeze-best.pt")
    ap.add_argument("--limit", type=int, default=0, help=">0 时冒烟只取前 N 句")
    ap.add_argument("--groups", nargs="*", default=None)
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

    done = {}
    if not smoke and os.environ.get("I0_RESUME") == "1" and PARTIAL.exists():
        done = json.loads(PARTIAL.read_text(encoding="utf-8")).get("done", {})
        print("resume: 已完成 %s" % list(done), flush=True)

    mdl = build_model(str(REPO / a.ckpt), 1024, n, dev).eval()
    results = {}

    for g in groups:
        rp = M / ("%s-%s-uwer.json" % (TAG, g))
        if not smoke and g in done and rp.exists():
            results[g] = json.loads(rp.read_text(encoding="utf-8"))
            print("[%s] resume from receipt" % g, flush=True)
            continue
        print("=== group %s ===" % g, flush=True)
        ResampleDataset.n_padded = 0
        ds = ResampleDataset("dev", labels_dv, word2idx, dv_tf, False)
        ds.mode = g
        if smoke:
            ds.items = ds.items[:a.limit]
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=P78.collate, num_workers=0)
        print("  dev items = %d" % len(ds), flush=True)
        if smoke:
            items, item_pkl = collect(mdl, loader, dev, labels_dv), None
        else:
            items, item_pkl = collect_resumable(mdl, loader, dev, labels_dv, g)
        r = evaluate_all(g, items, idx2word, freq, a.ckpt)
        r["tag"] = "%s-%s" % (TAG, g)
        r["resample"] = g
        r.pop("corruption", None)
        r["alignment"] = align_stats(items)
        r["T_conv_per_gloss_median"] = r["alignment"]["T_conv_per_gloss_median"]
        r["nonblank_frames_per_gloss_mean"] = r["alignment"]["nonblank_frames_per_gloss_mean"]
        r["n_padded_short"] = int(ResampleDataset.n_padded)
        r["note"] = "I0 推理期时间分辨率扫描；指标口径与 night_eval_full.py 一致"
        print("  WER %.2f | uWER %.2f | low %.2f | high %.2f | ins %d | "
              "T/L %.2f | nb/gloss %.3f | padded %d"
              % (r["WER_official"], r["uWER_pct"], r["low_bucket"]["WER_pct"],
                 r["high_bucket"]["WER_pct"], r["edits"]["ins"],
                 r["T_conv_per_gloss_median"],
                 r["nonblank_frames_per_gloss_mean"], r["n_padded_short"]),
              flush=True)
        results[g] = r
        if not smoke:
            M.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            done[g] = {"WER_official": r["WER_official"]}
            CACHE.mkdir(parents=True, exist_ok=True)
            PARTIAL.write_text(json.dumps({"done": done}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
            if item_pkl is not None:
                shutil.rmtree(item_pkl, ignore_errors=True)  # 组成功，清掉组内分片
        del items, ds, loader
        if dev == "cuda":
            torch.cuda.empty_cache()

    del mdl
    if dev == "cuda":
        torch.cuda.empty_cache()
    if smoke:
        print("SMOKE DONE (不写正式收据/报告)", flush=True)
        return

    # ---------------- 判定 ----------------
    base = results[BASE]
    base_wer = base["WER_official"]
    audit_ok = CLEAN_RANGE[0] <= base_wer <= CLEAN_RANGE[1]
    checks = {}
    for g in CANDIDATES:
        r = results[g]
        delta = base_wer - r["WER_official"]
        point, lo, hi = paired_bootstrap(base, r)
        c = {
            "delta_pp": round(delta, 4),
            "bootstrap_point_pp": round(point, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "R1_delta": bool(delta >= RULE_DELTA),
            "R2_ci": bool(lo > 0),
            "R3_high": bool(r["high_bucket"]["WER_pct"]
                            - base["high_bucket"]["WER_pct"] <= 1.00),
            "R5_ins": bool(r["edits"]["ins"]
                           <= base["edits"]["ins"] * 1.10),
            "R6_micro": bool(r["edits"]["micro_subdel_pct"]
                             - base["edits"]["micro_subdel_pct"] <= 0.30),
            "R7_low_bucket": bool(base["low_bucket"]["WER_pct"]
                                  - r["low_bucket"]["WER_pct"] >= -1.00),
        }
        c["pass"] = bool(c["R1_delta"] and c["R2_ci"] and c["R3_high"]
                         and c["R5_ins"] and c["R6_micro"] and c["R7_low_bucket"])
        checks[g] = c
        print("  %-8s Δ%+.2f CI[%.3f,%.3f] R1%s R2%s R3%s R5%s R6%s R7%s => %s"
              % (g, delta, lo, hi, c["R1_delta"], c["R2_ci"], c["R3_high"],
                 c["R5_ins"], c["R6_micro"], c["R7_low_bucket"],
                 "PASS" if c["pass"] else "FAIL"), flush=True)

    passed = [g for g in CANDIDATES if checks[g]["pass"]]
    representative = max(passed, key=lambda g: checks[g]["delta_pp"]) if passed else None
    verdict = ("INVALID" if not audit_ok else ("SIGNAL" if passed else "FAIL"))
    print("base=%.2f audit=%s => %s %s"
          % (base_wer, audit_ok, verdict, representative or ""), flush=True)

    i1p = M / "p228-i1-correlation.json"
    i1 = json.loads(i1p.read_text(encoding="utf-8")) if i1p.exists() else None

    out = {
        "tag": TAG, "verdict": verdict, "representative": representative,
        "rule_delta_pp": RULE_DELTA, "base_group": BASE,
        "base_WER_official": base_wer, "audit_clean_in_range": audit_ok,
        "clean_allowed_range": list(CLEAN_RANGE),
        "groups": GROUPS, "candidates": CANDIDATES,
        "checks": checks,
        "deltas_all_pp": {g: round(base_wer - r["WER_official"], 4)
                          for g, r in results.items()},
        "results": {g: {k: v for k, v in r.items() if k != "per_item"}
                    for g, r in results.items()},
        "I1": i1,
        "prereg": "docs/planning/I0-PREREG-2026-11-08.md",
        "note": ("零训练、参数无关；FAIL 后不投 I2、不进 test；"
                 "repeat2 的重复帧不增加视觉信息，只增加 CTC 可发射位置"),
    }
    M.mkdir(parents=True, exist_ok=True)
    (M / "p228-i0-probe.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (M / "p228-i0-verdict.txt").write_text(verdict, encoding="utf-8")

    if all(g in done for g in GROUPS):
        elapsed = ("四档前向已在首次运行完成（见 `artifacts/logs_P228_i0.log` / "
                   "`..._first3attempts.log`）；本次为判定与报告重生成，耗时 %.1f min"
                   % ((time.time() - t0) / 60.0))
    else:
        elapsed = "%.1f min" % ((time.time() - t0) / 60.0)
    write_report(results, checks, verdict, base_wer, audit_ok,
                 representative, elapsed, i1)
    PARTIAL.unlink(missing_ok=True)
    print("I0 DONE %s elapsed %s" % (verdict, elapsed), flush=True)


if __name__ == "__main__":
    main()
