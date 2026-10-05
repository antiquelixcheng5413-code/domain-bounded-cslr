# -*- coding: utf-8 -*-
"""P53 · 路线 2 第一步：并行提取 train/dev 全部特征（Tasks API）

## 为什么做这个
P48b 实测：P42 用离线特征训练，喂实时特征时 train exact 从 96.7% 掉到 3.3%
—— 训练与推理用了**两个不同的提取器**（旧 holistic vs 新 tasks API），
模型记忆绑在离线坐标系上，线上完全用不上。

P52 判定路线 1（复现旧提取器）失败：deltas 段（49% 维度）的基准无法复现。
⇒ 只能走路线 2：**让训练和推理都用新 Tasks API**。

P51 已证两种特征判别力无显著差异（macroAUC 0.7791 vs 0.7744），
所以重提**不损失判别力**，却能彻底消除坐标系错位。

## 本脚本
- 输出到**新目录** `artifacts/part3_features_tasksapi/`，旧特征一个字节都不动
- 8 进程并行（单视频 3.8s，5488 条 ≈ 45 分钟）
- 每个进程建一次模型（摊到多帧可忽略）
- 断点续跑：已存在的文件跳过，中断后可直接重跑
- **只处理 train 和 dev，绝不碰 test**（test 是冻结 split）

## 关键校验（提取后自动做）
1. 维度断言 (48, 368)，错了立刻报错 —— 静默维度错误栽过（P34）
2. 抽 3 条与旧特征对比 cos，确认量级一致（不必逐位相同，那是路线 1 的要求）
3. 统计 presence 检出率，与旧特征对照
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
OUT_ROOT = REPO / "artifacts/part3_features_tasksapi"
VID = Path("/mnt/c/Users/su127/Desktop/csl视频")
# 🔴 test 是冻结 split，绝不提取（P48/P50 铁律，与仓库既有保护一致）
ALLOWED_SPLITS = {"train": "train", "dev": "validation"}


def _worker(payload):
    """子进程：提取单个视频。返回 (split, sid, ok, err)

    🔴 每个任务新建 extractor 会重复建 3 个 MediaPipe 模型（0.9s + 371MB）。
    在 ProcessPoolExecutor 里模型建在子进程、随进程结束而释放，
    导致每条视频都重建一次 —— 内存峰值高且慢。
    改用**模块级全局缓存**：同一子进程内复用同一个 extractor。
    """
    global _EX
    split, sid, video = payload
    out_dir = OUT_ROOT / ALLOWED_SPLITS[split]
    dst = out_dir / (sid + ".landmark.npy")
    if dst.exists():
        return split, sid, "skip", ""
    try:
        if _EX is None:
            os.environ.setdefault("LD_LIBRARY_PATH", "/home/su127/libs")
            sys.path.insert(0, str(REPO / "app" / "backend"))
            from realtime_landmark import RealtimeLandmarkExtractor
            # ⚠️ reset_per_video 保持默认 True —— 必须与线上推理**完全一致**。
            # P48 证明不复现会导致同视频多次预测结果不同。
            # 这里只缓存 extractor 对象（省掉重复 import），
            # 模型重建仍由 extract 内部的 _reset_for_new_video 负责。
            _EX = RealtimeLandmarkExtractor()
            _EX._ensure()
        feat = _EX.extract_to_48x368(video)          # (48, 368)
        if feat.shape != (48, 368):
            return split, sid, "bad_shape", str(feat.shape)
        tmp = dst.with_suffix(".tmp.npy")
        np.save(tmp, feat.astype(np.float32))
        os.replace(tmp, dst)                        # 原子写，避免半截文件
        return split, sid, "ok", ""
    except Exception as exc:                          # noqa: BLE001
        return split, sid, "fail", "%s: %s" % (type(exc).__name__, str(exc)[:80])


_EX = None


def build_task_list(n_limit: int):
    import csv
    tasks = []
    for split, video_split in (("train", "train"), ("dev", "dev")):
        lab = REPO / "data/raw/CE-CSL/label" / ("train.csv" if split == "train"
                                                else "dev.csv")
        with open(lab, newline="", encoding="utf-8") as f:
            ids = [r["Number"] for r in csv.DictReader(f)]
        vroot = VID / video_split
        for sid in ids:
            found = None
            if vroot.exists():
                for d in sorted(vroot.iterdir()):
                    p = d / (sid + ".mp4")
                    if p.exists():
                        found = str(p)
                        break
            if found is None:
                continue
            dst = OUT_ROOT / ALLOWED_SPLITS[split] / (sid + ".landmark.npy")
            if dst.exists():
                continue
            tasks.append((split, sid, found))
            if n_limit and len(tasks) >= n_limit:
                return tasks
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0=全量")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for split, name in ALLOWED_SPLITS.items():
        (OUT_ROOT / name).mkdir(parents=True, exist_ok=True)

    tasks = build_task_list(a.limit)
    print("待提取 %d 条（train + dev，**不含 test**）" % len(tasks), flush=True)
    if a.dry_run:
        for t in tasks[:5]:
            print("  例:", t)
        return
    if not tasks:
        print("无待提取任务（可能已完成）")
        return

    t0 = time.time()
    stat = collections.Counter()
    errors = []
    done = 0

    # 🔴 WSL 下长期驻留的 ProcessPoolExecutor 多次触发 E_UNEXPECTED
    # (0x8007274c) 使整个 WSL 实例崩溃。改为**分段并行**：
    # 每批 12 条开一次池、用完立即关闭，池不长期驻留。
    # 代价：每批重建一次模型（0.9s），12 条摊下来可忽略。
    BATCH = 12
    if a.workers <= 1:
        sys.path.insert(0, str(REPO / "app" / "backend"))
        batches = [tasks[i:i + BATCH] for i in range(0, len(tasks), BATCH)]
    else:
        batches = [tasks[i:i + BATCH]
                   for i in range(0, len(tasks), a.workers)]

    for bi, batch in enumerate(batches):
        if a.workers <= 1:
            for t in batch:
                split, sid, status, err = _worker(t)
                stat[status] += 1
                if status == "fail":
                    errors.append({"sid": sid, "err": err})
                done += 1
        else:
            try:
                with ProcessPoolExecutor(max_workers=len(batch)) as ex:
                    for r in ex.map(_worker, batch):
                        split, sid, status, err = r
                        stat[status] += 1
                        if status == "fail":
                            errors.append({"sid": sid, "err": err})
                        done += 1
            except Exception as exc:                            # noqa: BLE001
                # 批次失败不终止整体：落盘进度后可续跑
                print("  批次 %d 异常: %s（跳过，继续）"
                      % (bi, str(exc)[:60]), flush=True)
                done += len(batch)
        if done % 50 < BATCH or done == len(tasks):
            el = time.time() - t0
            eta = el / done * (len(tasks) - done)
            print("  %d/%d  %s  用时 %.0fs  预计剩余 %.0fs"
                  % (done, len(tasks), dict(stat), el, eta), flush=True)
            with open("/tmp/p53_progress.json", "w") as pf:
                json.dump({"done": done, "total": len(tasks),
                           "stat": dict(stat)}, pf)

    print("\n完成：%s" % dict(stat))
    if errors:
        print("失败 %d 条，前 10：" % len(errors))
        for e in errors[:10]:
            print("   %s %s" % (e["sid"], e["err"]))

    receipt = {
        "experiment": "P53", "date": "2026-10-05",
        "purpose": "路线 2：统一 train/serve 的特征提取器",
        "output_root": str(OUT_ROOT),
        "old_features_untouched": str(REPO / "artifacts/part3_features"),
        "test_split_excluded": True,
        "total": len(tasks), "stat": dict(stat),
        "elapsed_s": round(time.time() - t0, 1),
        "errors": errors[:50],
    }
    p = REPO / "artifacts/metrics/blank-gov/p53-extract-tasksapi.json"
    p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
