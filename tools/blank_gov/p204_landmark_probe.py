"""P204 / E4 前置探针：landmark 数据对齐与质量检查（零训练成本）。

回答三个问题：
1) landmark48/96 的数组形状、数值范围、是否有 NaN；
2) landmark 帧数与 RGB jpg 帧数是否一致（不一致就不能逐帧辅助监督）；
3) 预提取的特征（artifacts/part3_features*）里是否已有可用的 landmark 序列。

只读，不训练。
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
LM48 = REPO / "data/processed/ce-csl-landmark48"
LM96 = REPO / "data/processed/ce-csl-landmark96"
RGB = REPO / "artifacts/official_rgb"
CSV_DEV = REPO / "external/TFNet/data/CE-CSL/dev.csv"


def read_dev(n: int = 8):
    """dev.csv 首行是表头（Number,...），按 sample_id 形如 dev-00001 过滤。"""
    out = []
    with open(CSV_DEV, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or not row[0]:
                continue
            if not (row[0].startswith("dev-") or row[0].startswith("train-")
                    or row[0].startswith("test-")):
                continue
            out.append((row[0], row[1]))
            if len(out) >= n:
                break
    return out


def main() -> None:
    pairs = read_dev(8)
    print("dev.csv 前 %d 条：sid / translator" % len(pairs))
    for sid, tr in pairs:
        a48 = np.load(LM48 / ("%s.npy" % sid))
        a96_path = LM96 / ("%s.npy" % sid)
        a96 = np.load(a96_path) if a96_path.exists() else None
        d = RGB / "dev" / tr / sid
        n_jpg = (len([f for f in os.listdir(d) if f.endswith(".jpg")]
                     ) if d.exists() else -1)
        print("-" * 68)
        print("%s  translator=%s  jpg=%d" % (sid, tr, n_jpg))
        print("  lm48 shape=%s dtype=%s  T=%d" % (a48.shape, a48.dtype, a48.shape[0]))
        print("  lm48 min=%.4f max=%.4f mean=%.4f  NaN=%d"
              % (np.nanmin(a48), np.nanmax(a48), np.nanmean(a48),
                 int(np.isnan(a48).sum())))
        if a96 is not None:
            print("  lm96 shape=%s dtype=%s  T=%d  NaN=%d"
                  % (a96.shape, a96.dtype, a96.shape[0],
                     int(np.isnan(a96).sum())))
        if n_jpg > 0:
            print("  T(48)-jpg = %d   T(96)-jpg = %s"
                  % (a48.shape[0] - n_jpg,
                     (a96.shape[0] - n_jpg) if a96 is not None else "n/a"))
        # 前两帧数值，判断坐标系（归一化 vs 像素）
        print("  frame0[:8] = %s" % np.round(a48[0][:8], 3).tolist())

    # 全量统计（只统计形状不读内容，快）
    for name, root in (("lm48", LM48), ("lm96", LM96)):
        files = list(root.glob("*.npy"))
        print("\n%s: %d 个文件，示例 %s" % (name, len(files),
                                          [f.name for f in files[:3]]))
    # 预提取特征目录
    for p in sorted((REPO / "artifacts").glob("part3_features*")):
        sub = [x.name for x in p.iterdir()][:8]
        print("feature dir %s -> %s" % (p.name, sub))

    # _labels length（决定 48 步 aux CTC 是否够长）
    print("\n标签长度分布（按 gloss 的 '/' 分段，近似）")
    for split in ("train", "dev"):
        path = REPO / ("external/TFNet/data/CE-CSL/%s.csv" % split)
        lens = []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if not row or not row[0]:
                    continue
                if not row[0].startswith(split + "-"):
                    continue
                lens.append(len([w for w in row[3].split("/") if w]))
        lens = np.asarray(lens)
        print("  %-5s n=%d  mean=%.1f  p50=%d  p90=%d  p99=%d  max=%d  "
              ">40 的样本=%d" % (split, len(lens), lens.mean(),
                                 np.percentile(lens, 50), np.percentile(lens, 90),
                                 np.percentile(lens, 99), lens.max(),
                                 int((lens > 40).sum())))


if __name__ == "__main__":
    sys.exit(main())
