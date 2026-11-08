#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P222 / G0 完整版：train 集前向挖混淆对，验证 G1 的硬前提。

方案 OPT-DIRECTIONS-3 §5 规定：G1（用真实混淆对做 hard negatives）启动前，
必须先在 train 集上重挖混淆对并验证与 dev 混淆表的重叠度。
train WER 仅 1.03%（强过拟合），train 上可能根本不犯错 ⇒ 本项同时挖两类信号：
  (a) greedy 实际替换错误对（严格 Levenshtein DP 回溯）
  (b) 帧级 top-k 竞争对：CTC greedy 路径每个非 blank 段，取该段平均 logp 里
      最高的 k 个「非自身、非 blank」类 ⇒ 即使全对，也能揭示模型觉得哪些词相近。

预注册判据（08:25 写死，看到结果前，禁止事后调整）：
  主：(b) 在 k=2 时对 dev 混淆表（严格回溯的 606 个唯一对）的 recall >= 20%
      且 相对「条件随机对照（固定 ref、随机 hyp）」的 lift >= 3.0
      ⇒ 两条同时满足才判 G1 前提成立（PRECONDITION_PASS），否则 STOP。
  次：报 (a) 的对数；若 < 50 则记录「train 过拟合坐实，greedy 混淆表不可用」。
  ⚠️ 本项只作前提诊断，不报 WER 性能、不判模型好坏、不进 test。

用法：venv/bin/python -u tools/blank_gov/p222_g0_train_confusion.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

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
from p197_beam_lm import greedy_fast, build_model  # noqa: E402
from p219_subpair_semantics import dp_pairs  # noqa: E402

P78.OFFICIAL_PAD = True
P78.RGB_FIX = True
P78.BLANK_BIAS = 0.0
M = REPO / "artifacts/metrics/blank-gov"
CKPT = REPO / "artifacts/checkpoints/p184-tfnet-freeze-best.pt"
K = 2
RULE_RECALL = 0.20
RULE_LIFT = 3.0
# 10:35 续跑修复：08:30 首跑后半程因显存碎片 + OOM 重试慢了约 5 倍。
# 实测 train 集最大 520 帧、>400 帧仅 18 条(0.36%)，故阈值 400 影响极小。
BATCH = int(os.environ.get("G0_BATCH", "1"))
MAX_T = int(os.environ.get("G0_MAX_T", "400"))
PARTIAL = REPO / "artifacts/cache/p222_g0_partial.json"
RESUME = os.environ.get("G0_RESUME", "0") == "1"


def segments(logp, k=K):
    """CTC greedy 路径按连续相同非 blank 类切段，返回每段的 top-k 竞争类 id。"""
    path = logp.argmax(1)
    segs, cur, st = [], None, 0
    for t, x in enumerate(path):
        x = int(x)
        if x != cur:
            if cur is not None and cur != 0:
                segs.append((cur, st, t))
            cur, st = x, t
    if cur is not None and cur != 0:
        segs.append((cur, st, len(path)))
    out = []
    for cls, s, e in segs:
        m = logp[s:e].mean(0).copy()
        m[0] = -1e9
        idx = np.argpartition(m, -(k + 2))[-(k + 2):]
        idx = idx[np.argsort(m[idx])[::-1]]
        got = 0
        for c in idx:
            c = int(c)
            if c == cls or c == 0:
                continue
            out.append((cls, c))
            got += 1
            if got >= k:
                break
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels_tr = P78.read_csv_split("train.csv")
    labels_dv = P78.read_csv_split("dev.csv")
    word2idx, n, idx2word = P78.DPM.Word2Id(
        str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
        str(P78.CSV_DIR / "test.csv"), "CE-CSL")
    C = n + 1

    # ---- dev 混淆表（严格 DP 回溯，作为被覆盖目标）----
    dd = torch.load(M / "p188_logp_p184.pt", map_location="cpu", weights_only=False)
    dev_pairs = set()
    for s in dd["items"]:
        if s not in labels_dv:
            continue
        r = OW.split_glued(OW.split_gloss_sequence(labels_dv[s][1], OW.CFG))
        lp = dd["items"][s]["logp"]
        lp = lp.numpy() if hasattr(lp, "numpy") else np.asarray(lp)
        h = OW.split_glued(list(greedy_fast(lp, idx2word)))
        ps, _ = dp_pairs(r, h)
        for a, b in ps:
            if a in word2idx and b in word2idx:
                dev_pairs.add((int(word2idx[a]), int(word2idx[b])))
    print("dev unique confusion pairs =", len(dev_pairs), flush=True)

    # ---- train 前向（边算边统计，不落盘 4GB 后验）----
    _, tf = P78.build_transforms(hflip=False, img_size=160, y_shift=0,
                                 temporal_rescale=0.0)
    ds = P78.RGBSeqDataset("train", labels_tr, word2idx, tf, False)
    _lim = int(os.environ.get("G0_LIMIT", "0"))   # 冒烟用，正式跑不设
    if _lim > 0:
        ds.items = ds.items[:_lim]
    print("train usable samples =", len(ds.items), flush=True)
    greedy_pairs = []
    comp_pairs = [[] for _ in range(3)]   # k=1,2,3
    n_edit, n_ref, seen, skipped = 0, 0, 0, 0
    if RESUME and PARTIAL.exists():
        _st = json.loads(PARTIAL.read_text(encoding="utf-8"))
        greedy_pairs = [tuple(x) for x in _st["greedy_pairs"]]
        comp_pairs = [[tuple(x) for x in c] for c in _st["comp_pairs"]]
        n_edit, n_ref, seen, skipped = (_st["n_edit"], _st["n_ref"],
                                        _st["seen"], _st.get("skipped", 0))
        ds.items = ds.items[seen:]
        print("RESUME partial: seen=%d skipped=%d remaining=%d"
              % (seen, skipped, len(ds.items)), flush=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        collate_fn=P78.collate, num_workers=0,
                        pin_memory=False)
    mdl = build_model(CKPT, 1024, n, dev)
    mdl.eval()
    t_start = datetime.now()
    with torch.no_grad():
        for bi, (vid, _tgt, _tl, dl, _tll, sids, _ids) in enumerate(loader):
            # 跳过超长视频（显存峰值元凶）。dl 是 collate 给出的真实帧长张量，
            # 比猜 vid 的维度顺序可靠；official-pad 下含约 +12 padding，影响可忽略。
            _lens_raw = [int(x) for x in dl.view(-1).tolist()]
            if _lens_raw and max(_lens_raw) > MAX_T:
                skipped += len(sids)
                seen += len(sids)
                continue
            out = mdl(vid.to(dev), dl, False)
            # 10:52 首跑在 cuMemcpyDtoHAsync 上撞 CUDA_ERROR_UNKNOWN；
            # 显式同步后再做 D2H，规避 WSL 上的异步拷贝驱动错误（batch=1 开销很小）
            if dev == "cuda":
                torch.cuda.synchronize()
            logp = F.log_softmax(out[0], dim=-1).float().cpu().numpy()
            lens = dl.view(-1).tolist()
            for b, sid in enumerate(sids):
                T = min(P78._conv_len(int(lens[b])), logp.shape[0])
                lp = logp[:T, b, :].astype(np.float32)
                r = OW.split_glued(OW.split_gloss_sequence(labels_tr[sid][1], OW.CFG))
                h = OW.split_glued(list(greedy_fast(lp, idx2word)))
                ps, dist = dp_pairs(r, h)
                n_edit += dist
                n_ref += len(r)
                for a, bb in ps:
                    if a in word2idx and bb in word2idx:
                        greedy_pairs.append((int(word2idx[a]), int(word2idx[bb])))
                # 逐段取 top-3 候选，按前 k 个分别归入 k=1/2/3 集合
                path = lp.argmax(1)
                cur, st = None, 0
                seg_defs = []
                for t, x in enumerate(path):
                    x = int(x)
                    if x != cur:
                        if cur is not None and cur != 0:
                            seg_defs.append((cur, st, t))
                        cur, st = x, t
                if cur is not None and cur != 0:
                    seg_defs.append((cur, st, len(path)))
                for cls, s0, e0 in seg_defs:
                    m = lp[s0:e0].mean(0).copy()
                    m[0] = -1e9
                    idxs = np.argpartition(m, -5)[-5:]
                    idxs = idxs[np.argsort(m[idxs])[::-1]]
                    cand = [int(c) for c in idxs if int(c) != cls and int(c) != 0]
                    for kk in (1, 2, 3):
                        for c in cand[:kk]:
                            comp_pairs[kk - 1].append((cls, c))
                seen += 1
            if bi % 100 == 0 and dev == "cuda":
                torch.cuda.empty_cache()
            if bi % 400 == 0:
                # 断点存盘（项目铁律：密集定期存盘，避免再次被叫停就全丢）
                PARTIAL.parent.mkdir(parents=True, exist_ok=True)
                PARTIAL.write_text(json.dumps({
                    "seen": seen, "skipped": skipped,
                    "n_edit": n_edit, "n_ref": n_ref,
                    "greedy_pairs": [list(x) for x in greedy_pairs],
                    "comp_pairs": [[list(x) for x in c] for c in comp_pairs],
                }), encoding="utf-8")
                _el = (datetime.now() - t_start).total_seconds()
                print("forward %d/%d samples=%d skipped=%d elapsed=%.0fs partial_saved"
                      % (bi, len(loader), seen, skipped, _el), flush=True)
    del mdl
    if dev == "cuda":
        torch.cuda.empty_cache()

    # 导出完整 train 竞争对表（k=2），供 G1 判据过线时直接用作 hard negatives。
    # 注意：periodic partial 最后一次存盘会漏掉末尾不足 400 batch 的部分，
    # 故这里在循环结束后单独导出完整表，避免 G1 再跑一次 40 分钟前向。
    _k2 = set(comp_pairs[1])
    _tbl = {}
    for _pa, _pb in _k2:
        _tbl.setdefault(str(_pa), []).append(int(_pb))
    (M / "p222_confusion_map.json").write_text(json.dumps({
        "source": "train 帧级 CTC greedy 段 top-2 竞争类（p184-tfnet-freeze-best）",
        "k": 2, "n_unique_pairs": len(_k2),
        "n_gloss_with_confusers": len(_tbl),
        "map": _tbl,
        "note": "仅当 G0 判 PRECONDITION_PASS 时才可用于 G1；dev 混淆表不得进训练",
    }, ensure_ascii=False), encoding="utf-8")
    print("confusion map -> p222_confusion_map.json (%d pairs)" % len(_k2), flush=True)

    train_wer = 100.0 * n_edit / max(n_ref, 1)
    print("train greedy WER = %.3f%%  greedy sub pairs = %d"
          % (train_wer, len(greedy_pairs)), flush=True)

    rng = np.random.default_rng(42)
    vocab = np.arange(1, C)
    results = {}
    for kk in (1, 2, 3):
        s = set(comp_pairs[kk - 1])
        size = len(s)
        hit = len(dev_pairs & s)
        recall = hit / max(len(dev_pairs), 1)
        # 同规模随机对集合的期望 recall（20 次）
        # 条件随机对照（与 P219 一致）：固定 dev 对的 ref，随机替换 hyp，
        # 看这些伪混淆对有多少落在 train 竞争对集合里（20 次重复）。
        # 08:27 冒烟发现「均匀随机词对」在 3516 类词表下期望命中≈0，
        # lift 会退化成天文数字；改条件随机，判据阈值 recall>=0.20 / lift>=3.0 不变。
        rnd = []
        for _ in range(20):
            pseudo = set()
            for (pa, _pb) in dev_pairs:
                pseudo.add((pa, int(rng.choice(vocab))))
            rnd.append(len(pseudo & s) / max(len(dev_pairs), 1))
        rnd_recall = max(float(np.mean(rnd)), 1.0 / max(len(dev_pairs), 1))
        lift = recall / max(rnd_recall, 1e-9)
        results["k%d" % kk] = {
            "n_pairs_raw": len(comp_pairs[kk - 1]),
            "unique_pairs": size,
            "dev_hit": hit,
            "recall": round(recall, 4),
            "random_recall": round(rnd_recall, 6),
            "lift": round(lift, 2),
        }
        print("k=%d unique=%d hit=%d recall=%.4f random=%.6f lift=%.2f"
              % (kk, size, hit, recall, rnd_recall, lift), flush=True)

    gs = set(greedy_pairs)
    ghit = len(dev_pairs & gs)
    k2 = results["k2"]
    passed = bool(k2["recall"] >= RULE_RECALL and k2["lift"] >= RULE_LIFT)
    verdict = "PRECONDITION_PASS" if passed else "STOP"
    out = {
        "tag": "p222", "kind": "g0_precondition_diagnostic",
        "train_samples": seen, "skipped_long_samples": skipped,
        "batch_size": BATCH, "max_frames": MAX_T,
        "train_greedy_WER_pct": round(train_wer, 3),
        "n_train_greedy_sub_pairs": len(greedy_pairs),
        "greedy_unique": len(gs), "greedy_dev_hit": ghit,
        "greedy_recall": round(ghit / max(len(dev_pairs), 1), 4),
        "dev_unique_confusion_pairs": len(dev_pairs),
        "topk": results,
        "pre_registered": {
            "fixed_at": "2026-11-08 08:25 CST (before forward pass)",
            "rule": "k=2 recall>=0.20 AND lift>=3.0 -> PRECONDITION_PASS else STOP",
        },
        "verdict": verdict,
    }
    (M / "p222-g0-train-confusion.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("topk",)},
                     ensure_ascii=False, indent=2), flush=True)

    txt = """# P222 / G0 完整版：train 混淆对能否覆盖 dev 混淆对 — 阶段报告（%s）

> 生成：%s｜train 集 %d 条前向（GPU）｜**前提诊断，不报性能、不进 test**
> 预注册判据 08:25 写死：k=2 的 recall ≥20%% 且 lift ≥3.0 ⇒ PRECONDITION_PASS，否则 STOP。

## 1. 为什么必须做
第三轮方案 G1 想用「train 集挖出的混淆对」做最小对立体 hard negatives。
但 train WER 仅 1.03%%（强过拟合），train 上可能根本不犯错；
且 dev 混淆对**不得进训练**（数据泄漏）。因此 train 混淆表的覆盖能力是 G1 的一票否决。

## 2. 方法
- 模型：`p184-tfnet-freeze-best.pt`，train split 无增广前向（与 dev 评估同口径）。
- (a) greedy 实际替换错误：严格 Levenshtein DP 回溯（复用 P219，自检已通过）。
- (b) 帧级 top-k 竞争对：CTC greedy 路径按连续相同非 blank 类切段，
  每段取平均 logp 里最高的 k 个「非自身、非 blank」类 ⇒ 即使 train 全对，
  也能揭示模型觉得哪些 gloss 最容易互相混淆。
- 覆盖度：与 dev 严格混淆表（%d 个唯一对）求交；
  对照 = **条件随机**（固定 dev 对的 ref、随机替换 hyp，与 P219 一致）重复 20 次的期望 recall，
  lift = recall / 随机 recall。
  ⚠️ 08:27 冒烟时原定的「均匀随机词对」对照在 3516 类词表下期望命中≈0，
  使 lift 退化成 10^7 量级 ⇒ 正式跑前改为条件随机；**判据阈值 recall≥0.20 / lift≥3.0 未变**，
  这是对照实现的修正，不是看到结果后调判据。

## 3. 结果

| 指标 | 值 |
|---|---|
| train 样本 / greedy WER | %d / **%.3f%%** |
| (a) greedy 替换对（唯一 / dev 命中） | %d / %d（recall %.4f） |
| dev 混淆表规模 | %d |

| top-k | 唯一竞争对 | 命中 dev 对 | recall | 随机 recall | lift |
|---|---|---|---|---|---|
%s

## 4. 判定：**%s**

%s

## 5. 诚实边界
- 帧级竞争对是「模型当前觉得相近」的近似，不是标注的最小对立体；
  top-k 越大覆盖越广但噪声越多，故主判据固定在 k=2，不事后挑 k。
- 随机对照按均匀词表采样，未控制词频；长尾先验可能抬高 lift，
  但 lift 只作相对量级参考，主判据同时要求 recall 绝对过线。
- 本项只回答「train 混淆表能否覆盖 dev 混淆对」，不回答 G1 训练是否一定有效。
- 未跑 test、未 commit、未 push。

数据来源 `artifacts/metrics/blank-gov/p222-g0-train-confusion.json`。
""" % (
        verdict, datetime.now().strftime("%Y-%m-%d %H:%M"), seen,
        len(dev_pairs), seen, train_wer, len(gs), ghit,
        ghit / max(len(dev_pairs), 1), len(dev_pairs),
        "\n".join("| %d | %d | %d | %.4f | %.6f | %.2f |" %
                  (kk, results["k%d" % kk]["unique_pairs"],
                   results["k%d" % kk]["dev_hit"],
                   results["k%d" % kk]["recall"],
                   results["k%d" % kk]["random_recall"],
                   results["k%d" % kk]["lift"]) for kk in (1, 2, 3)),
        verdict,
        ("过线 ⇒ G1 的 train 挖掘前提成立，可用 train 竞争对做 hard negatives；"
         "仍须新写预注册判据并做低频桶单独 bootstrap 后再启动 6 轮训练。"
         if passed else
         "未过线 ⇒ STOP：train 混淆表无法覆盖 dev 的真实混淆，"
         "G1 不能照搬；若要继续走最小对立体路线，需改为交叉验证/留出折挖掘或外部标注，"
         "并重新评估成本。按累计证据，合理结论是接受天花板收尾。"),
    )
    txt = txt.replace(
        "数据来源 `artifacts/",
        ("补充说明：本次为 **10:35 修复显存问题后的完整重跑**（08:30 首跑在 1500/2487 "
         "batch 被用户叫停、未产出判定）。修复项：batch=%d、每 100 batch `empty_cache()`、"
         "跳过 >%d 帧的超长视频 **%d 条**（占 train 0.36%%，实测最大 520 帧）、"
         "每 400 batch 断点存盘。**跳过的样本已计入 seen 但未参与统计**，"
         "判据阈值与对照算法均未改动。\n\n数据来源 `artifacts/" %
         (BATCH, MAX_T, skipped)))
    for p in (REPO / "docs/experiments/stage-P222-G0-train-confusion.md",
              DESK / "docs/experiments/stage-P222-G0-train-confusion.md"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("报告 -> stage-P222-G0-train-confusion.md (+桌面镜像)", flush=True)
    # 供自动链读取
    (M / "p222-g0-verdict.txt").write_text(verdict, encoding="utf-8")
    # 保留 partial 供审计/断点回溯（成功完成后不会自动续跑，保留无害）


if __name__ == "__main__":
    main()
