#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P151：决定性验证 —— 官方端到端链路最终喂给模型的是 RGB 还是 BGR

我们的链路（P149 实测）
  cv2.VideoCapture -> bgr -> cvtColor(BGR2RGB) -> resize -> imencode -> 落盘
  训练读图：imread -> cvtColor(BGR2RGB)        => 实测喂进去的是 **BGR**
  （逐通道 MAE vs 真RGB：R 15.90 / G 1.04 / B 16.28）

官方链路（照抄源码，不假设）
  抽帧 CE-CSLDataPreProcess.py
    vid = imageio.get_reader(videoPath)         # 非 cv2！
    image = vid.get_data(i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    cv2.imencode('.jpg', image)[1].tofile(path)
  读图 DataProcessMoudle.py:335
    cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (256, 256))

本脚本分两步（都不依赖任何假设）
  step1：用 cv2 作为基准，判定 **imageio.get_data() 到底输出哪种顺序**
  step2：完整跑一遍官方链路，把结果与「真 RGB」比较
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
VID = Path("/mnt/c/Users/su127/Desktop/csl视频")
OUT = REPO / "artifacts/official_rgb"

try:
    import imageio.v2 as iio
except Exception as e:  # pragma: no cover
    print("imageio 不可用：%s" % e)
    sys.exit(2)


def mae(a, b):
    return float(np.abs(a.astype(float) - b.astype(float)).mean())


def main() -> int:
    dirs = sorted((OUT / "dev").glob("*/*"))[:3]
    print("样本 %d 个\n" % len(dirs))
    for d in dirs:
        trans, sid = d.parent.name, d.name
        v = VID / "dev" / trans / ("%s.mp4" % sid)
        if not v.exists():
            continue

        # ---- 基准：cv2 解码第 0 帧 ----
        cap = cv2.VideoCapture(str(v))
        ok, bgr0 = cap.read()
        n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not ok:
            continue
        A = cv2.resize(cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB), (256, 256))  # 真 RGB

        # ---- step1：imageio 输出顺序 ----
        vid = iio.get_reader(str(v))
        f0 = vid.get_data(0)
        n_iio = vid.count_frames()
        vid.close()
        f0r = cv2.resize(f0, (256, 256))
        m_same = mae(f0r, A)                 # imageio == 真RGB ?
        m_swap = mae(f0r, A[:, :, ::-1])     # imageio == 真RGB 的 R/B 互换 ?
        order = "RGB" if m_same < m_swap else "BGR"
        print("── %s ──" % sid)
        print("  帧数：cv2 元数据 %d / imageio.count_frames %d" % (n_meta, n_iio))
        print("  step1 imageio 输出顺序判定 = %s  (vs真RGB %.2f / vs互换 %.2f)"
              % (order, m_same, m_swap))

        # ---- step2：完整官方链路 ----
        img = cv2.cvtColor(f0, cv2.COLOR_BGR2RGB)          # 官方 CE-CSLDataPreProcess
        img = cv2.resize(img, (256, 256))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tmp = tf.name
        cv2.imencode(".jpg", img)[1].tofile(tmp)
        off = cv2.resize(cv2.cvtColor(cv2.imread(tmp), cv2.COLOR_BGR2RGB),
                         (256, 256))                        # 官方 DataProcessMoudle:335
        Path(tmp).unlink(missing_ok=True)

        o_same = mae(off, A)
        o_swap = mae(off, A[:, :, ::-1])
        verdict = "真RGB" if o_same < o_swap else "BGR(互换)"
        print("  step2 官方链路最终 = %s  (vs真RGB %.2f / vs互换 %.2f)"
              % (verdict, o_same, o_swap))
        # ---- 我们的链路（P149 已知 = BGR）----
        ours = sorted(d.glob("*.jpg"))
        if ours:
            our = cv2.cvtColor(cv2.imread(str(ours[0])), cv2.COLOR_BGR2RGB)
            print("  我们链路最终     = %s  (vs真RGB %.2f / vs互换 %.2f)"
                  % ("真RGB" if mae(our, A) < mae(our, A[:, :, ::-1]) else "BGR(互换)",
                     mae(our, A), mae(our, A[:, :, ::-1])))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
