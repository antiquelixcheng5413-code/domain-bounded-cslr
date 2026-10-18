#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P149：抽帧链路端到端一致性核验（帧数 + 像素）

背景
  官方 CE-CSLDataPreProcess.py 用 imageio.get_reader().count_frames() 抽帧，
  cvtColor(BGR2RGB) → resize(256) → cv2.imencode(".jpg") 写盘；
  我们的 p77 用 cv2.VideoCapture 逐帧读（p77 注释自承"帧数可能差 1~2"）。
  ⇒ 这是最后一个**没核过的数据侧环节**。

核验两件事
  1. 帧数：我们落盘的 jpg 数 vs 视频元数据 CAP_PROP_FRAME_COUNT
  2. 像素：训练时真正喂给模型的图（cv2.imread + cvtColor，即 p78:168 的路径）
           vs 直接从视频解码的第 0 帧（cvtColor + resize）
     ⇒ 端到端覆盖「抽帧→写盘→读回」，含 jpg 压缩与 resize 插值

用法
  python tools/blank_gov/p149_frame_probe.py [n_samples]
"""
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
OUT = REPO / "artifacts/official_rgb"
VID = Path("/mnt/c/Users/su127/Desktop/csl视频")

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def log(*a):
    print("[P149 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    dirs = sorted((OUT / "dev").glob("*/*"))[:n]
    log("样本 %d 个" % len(dirs))
    diffs, maes, miss = [], [], 0
    for d in dirs:
        trans, sid = d.parent.name, d.name
        jpgs = sorted(d.glob("*.jpg"))
        n_ours = len(jpgs)
        v = VID / "dev" / trans / ("%s.mp4" % sid)
        if not v.exists():
            miss += 1
            continue
        cap = cv2.VideoCapture(str(v))
        n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, bgr = cap.read()
        cap.release()
        mae = None
        if ok and jpgs:
            ref = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (256, 256))
            ours = cv2.cvtColor(cv2.imread(str(jpgs[0])), cv2.COLOR_BGR2RGB)
            if ours is not None and ours.shape == ref.shape:
                mae = float(np.abs(ref.astype(float) - ours.astype(float)).mean())
                maes.append(mae)
        if n_meta > 0:
            diffs.append(n_ours - n_meta)
    log("视频缺失 %d（可能是 train 侧路径不同）" % miss)
    if diffs:
        d = np.array(diffs)
        log("──── 帧数：我们 − 视频元数据 ────")
        log("  n=%d  mean %+.2f  min %+d  max %+d  非零占比 %.1f%%"
            % (len(d), d.mean(), d.min(), d.max(), 100.0 * (d != 0).mean()))
        uniq, cnt = np.unique(d, return_counts=True)
        log("  分布：" + "  ".join("%+d×%d" % (u, c) for u, c in zip(uniq, cnt)))
        if abs(d.mean()) < 0.5 and (d != 0).mean() < 0.2:
            log("  ⇒ ✅ 抽帧帧数与视频一致（差异 ≤1 帧且占比小）")
        else:
            log("  ⇒ ⚠️ 存在系统性帧数差异，需进一步查（可能影响 CTC 的 T 长度）")
    if maes:
        m = np.array(maes)
        log("──── 首帧像素：训练读取 vs 视频直解 ────")
        log("  n=%d  MAE mean %.2f  max %.2f（0=完全一致，jpg 压缩一般 <5）"
            % (len(m), m.mean(), m.max()))
        if m.mean() < 5:
            log("  ⇒ ✅ 端到端链路一致（抽帧→写盘→读回）")
        elif m.mean() > 40:
            log("  ⇒ ❌ 差异过大 ⇒ 极可能是**通道顺序**错了（R/B 互换的量级）")
        else:
            log("  ⇒ ⚪ 中等差异，可能是 jpg 质量/resize 插值，需在报告里说明")
    return 0


if __name__ == "__main__":
    sys.exit(main())
