# -*- coding: utf-8 -*-
"""P40-A · 全量 RGB 窗口特征提取（r3d_18，Kinetics 预训练，冻结）

## 为什么要单独一个脚本

P40 的目标是「把 RGB 当主干接进 CTC」，需要 `(T, 512)` 的**时序**特征
（A2 的 `.a2_cache/r3d_18/` 是全局池化单向量 `(512,)`，架构上不能接 CTC）。
提取耗时约 2.5 小时，必须支持断点续跑，否则中断就白花。

## 关键设计

1. **断点续跑**：`ensure_rgb_cache` 按「文件存在且窗口数匹配」判定，
   中断后重跑会跳过已完成的。
2. **窗口数 20**：CTC 硬约束 `T >= 2L-1`。
   参考句长 mean 5.69 / max 16 → 需 T >= 10.4；20 窗覆盖 98.7% 样本。
3. **8 帧/窗**：实测 8 帧与 16 帧单 clip 耗时相同（32.6ms），8 帧更划算。
4. **每窗独立提特征**（不跨窗池化），保留时间维。
5. **dev 视频路径特殊**：`data/raw/CE-CSL/video/validation/` 是空目录，
   实际在 `/mnt/c/Users/su127/Desktop/csl视频/dev/`（P32b 实测）。

## 与 A2 的区别（不矛盾）

| | A2 | 本脚本 |
|---|---|---|
| 特征形状 | `(512,)` 全局单向量 | `(20, 512)` 窗口序列 |
| 能否接 CTC | **不能** | 能 |
| 测的是 | 视频级语义判别力 | 时序可对齐性 |

所以 A2 的「r3d_18 无效」与 P40 的结果不冲突。

## 用法

    python p40a_extract_rgb_windows.py            # 全量
    python p40a_extract_rgb_windows.py --limit 100 # 先试
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

TRAIN_VIDEO = REPO / "data/raw/CE-CSL/video/train"
DEV_VIDEO = Path("/mnt/c/Users/su127/Desktop/csl视频/dev")
CACHE = REPO / ".a2_cache/r3d_18_windows"

CLIP_MEAN = [0.43216, 0.394666, 0.37645]
CLIP_STD = [0.22803, 0.22145, 0.216989]


def read_csv_ids(p):
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["Number"]] = r["Gloss"]
    return out


def build_net(device):
    import torchvision
    net = getattr(torchvision.models.video, "r3d_18")(weights="DEFAULT")
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def find_video(sid):
    for root in (TRAIN_VIDEO, DEV_VIDEO):
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            p = d / (sid + ".mp4")
            if p.exists():
                return p
    return None


def extract_one(net, path, n_windows, frames, size, device):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    buf = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        buf.append(cv2.resize(fr, (size, size)))
    cap.release()
    if len(buf) < 3:
        return None
    edges = np.linspace(0, len(buf), n_windows + 1).astype(int)
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1, 1)
    out = []
    with torch.no_grad():
        for w in range(n_windows):
            seg = buf[edges[w]:edges[w + 1]]
            if len(seg) < 2:
                # 窗口内帧不足：复用最后一帧，保证时间步长度恒为 n_windows
                seg = [seg[-1]] if seg else [buf[-1]] * 2
            if len(seg) >= frames:
                idx = [round(i * (len(seg) - 1) / (frames - 1)) for i in range(frames)]
            else:
                idx = [min(len(seg) - 1, i) for i in range(frames)]
            arr = np.stack([seg[i] for i in idx])            # (T,H,W,3) BGR
            x = torch.from_numpy(arr).permute(3, 0, 1, 2).float().to(device) / 255.0
            x = (x.unsqueeze(0) - mean) / std
            out.append(net(x).squeeze(0).cpu().numpy())
    return np.stack(out).astype(np.float32)                   # (n_windows, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=20)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--limit", type=int, default=0, help="0=全量")
    ap.add_argument("--batch", type=int, default=8,
                    help="一次前向的窗口数（显存不足时调小）")
    ap.add_argument("--report-every", type=int, default=200)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CACHE.mkdir(parents=True, exist_ok=True)
    print("device={}  窗口={} x {} 帧  缓存={}".format(
        device, a.n_windows, a.frames, CACHE))

    tr = list(read_csv_ids(REPO / "data/raw/CE-CSL/label/train.csv").keys())
    dv = list(read_csv_ids(REPO / "data/raw/CE-CSL/label/dev.csv").keys())
    ids = tr + dv
    if a.limit:
        ids = ids[:a.limit]
    print("目标 {} 个视频（train {} + dev {}）".format(len(ids), len(tr), len(dv)))

    def has(sid):
        p = CACHE / (sid + ".npy")
        if not p.exists():
            return False
        try:
            return np.load(p, mmap_mode="r").shape[0] == a.n_windows
        except Exception:                                        # noqa: BLE001
            return False

    todo = [s for s in ids if not has(s)]
    print("待提取 {} 个（已完成 {}/{}）".format(
        len(todo), len(ids) - len(todo), len(ids)))
    if not todo:
        print("全部已完成，退出。")
        return

    net = build_net(device)
    t0 = time.time()
    ok = 0
    fail = []
    for k, sid in enumerate(todo, 1):
        vp = find_video(sid)
        if vp is None:
            fail.append((sid, "video not found"))
            continue
        try:
            arr = extract_one(net, vp, a.n_windows, a.frames, a.size, device)
        except Exception as e:                                  # noqa: BLE001
            fail.append((sid, "{}: {}".format(type(e).__name__, e)))
            continue
        if arr is None:
            fail.append((sid, "no frames"))
            continue
        np.save(CACHE / (sid + ".npy"), arr)
        ok += 1
        if k % a.report_every == 0 or k == len(todo):
            el = time.time() - t0
            rate = el / k
            eta = rate * (len(todo) - k) / 60
            print("  {}/{}  ok={}  {:.0f}s  {:.2f}s/个  ETA {:.0f} 分钟".format(
                k, len(todo), ok, el, rate, eta), flush=True)

    el = time.time() - t0
    print()
    print("完成：成功 {} / 失败 {}，耗时 {:.1f} 分钟".format(
        ok, len(fail), el / 60))
    if fail:
        print("失败清单（前 20 条）：")
        for sid, why in fail[:20]:
            print("  {}  {}".format(sid, why))
        bad = REPO / "artifacts/metrics/blank-gov/p40a-rgb-failed.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        import json
        bad.write_text(json.dumps(fail, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print("失败清单 -> {}".format(bad))
    tot = len(list(CACHE.glob("*.npy")))
    print("缓存现有 {} 个文件，体积 {:.1f} MB".format(
        tot, sum(f.stat().st_size for f in CACHE.glob("*.npy")) / 1e6))


if __name__ == "__main__":
    main()
