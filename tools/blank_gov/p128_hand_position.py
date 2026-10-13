"""P128：运动能量空间分布 —— CenterCrop(160) 到底裁掉了多少判别信息？

背景：源帧 256×256（已实测）。官方 RandomCrop(224) 保留 87.5%，
      我们 CenterCrop(160) 只保留中间 **62.5%**（x,y ∈ [48,208]），
      裁掉四周 48 px 边框（占面积 **61.0%**）。

为什么不用 MediaPipe 检手：
  - mediapipe 0.10.35 已移除 `solutions` API，新版 tasks 需额外模型文件
  - 更重要的：手语识别的判别信息本质是**运动**，帧间差分是更贴切的代理，
    且零依赖、零模型误差

方法（帧间运动能量图）：
  1. 每个视频抽 K 对相邻帧（间隔 dt），算灰度差 |I(t+dt) − I(t)|
  2. 所有差图按视频内平均后，再跨视频平均 ⇒ 256×256 的"运动能量图" E
  3. 比较两块区域：
     - 保留区（中心 160×160，占面积 39.0%）
     - 裁掉区（外圈 48 px，占面积 61.0%）
  4. 关键量：
     - loss_pct = 裁掉区能量 / 总能量  —— 裁掉 61% 面积实际损失多少信息
     - density_ratio = 裁掉区每像素能量 / 保留区每像素能量
       ≈ 1.0 ⇒ 边缘信息密度与中心相当 ⇒ 裁掉是**真损失**
       ≪ 1.0 ⇒ 信息集中在中心 ⇒ 160 合理

判据（预先写死）：
  loss_pct ≥ 45%  ⇒ 边缘承载了与其面积不相称的大量运动 ⇒ **支持 224**
  loss_pct 25~45% ⇒ 中性，以 P127 配对结果为准
  loss_pct < 25%  ⇒ 运动高度集中在中心 ⇒ **160 不是瓶颈**，P127 可止损
  （面积基线：裁掉区占 61.0%；若信息均匀铺满，loss_pct 应 ≈ 61%）
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
RGB = REPO / "artifacts" / "official_rgb" / "dev"

N_VIDEOS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
PAIRS_PER_VIDEO = int(sys.argv[2]) if len(sys.argv) > 2 else 8
DT = 3                 # 相邻帧间隔（太小则几乎无运动，太大则跨动作）
FULL, CROP = 256, 160
LO = (FULL - CROP) // 2            # 48
HI = LO + CROP                     # 208


def log(*a):
    print("[P128 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def main():
    vids = sorted([d for d in RGB.rglob("*") if d.is_dir()
                   and any(f.suffix == ".jpg" for f in d.iterdir())])
    random.Random(42).shuffle(vids)
    vids = vids[:N_VIDEOS]
    log("抽样 %d 个 dev 视频，每个抽 %d 对相邻帧（dt=%d）"
        % (len(vids), PAIRS_PER_VIDEO, DT))

    E = np.zeros((FULL, FULL), dtype=np.float64)
    n_used, n_pair = 0, 0
    t0 = time.time()

    for vi, vd in enumerate(vids):
        files = sorted(f for f in vd.iterdir() if f.suffix == ".jpg")
        if len(files) < DT + 2:
            continue
        # 均匀抽 PAIRS_PER_VIDEO 个起点
        starts = np.linspace(0, len(files) - DT - 1, PAIRS_PER_VIDEO).astype(int)
        acc = np.zeros((FULL, FULL), dtype=np.float64)
        cnt = 0
        for s in starts:
            a = cv2.imread(str(files[s]), cv2.IMREAD_GRAYSCALE)
            b = cv2.imread(str(files[s + DT]), cv2.IMREAD_GRAYSCALE)
            if a is None or b is None:
                continue
            if a.shape != (FULL, FULL) or b.shape != (FULL, FULL):
                a = cv2.resize(a, (FULL, FULL))
                b = cv2.resize(b, (FULL, FULL))
            acc += cv2.absdiff(a, b).astype(np.float64)
            cnt += 1
        if cnt == 0:
            continue
        E += acc / cnt          # 每个视频贡献等权（避免长视频主导）
        n_used += 1
        n_pair += cnt
        if (vi + 1) % 50 == 0:
            log("  %d/%d 视频，%.0fs" % (vi + 1, len(vids), time.time() - t0))

    log("有效视频 %d，帧对 %d，用时 %.0fs" % (n_used, n_pair, time.time() - t0))

    keep_m = np.zeros((FULL, FULL), dtype=bool)
    keep_m[LO:HI, LO:HI] = True
    E_keep = float(E[keep_m].sum())
    E_drop = float(E[~keep_m].sum())
    E_tot = E_keep + E_drop
    loss_pct = 100.0 * E_drop / max(E_tot, 1e-9)

    a_keep = keep_m.sum()
    a_drop = (~keep_m).sum()
    d_keep = E_keep / a_keep
    d_drop = E_drop / a_drop
    density_ratio = d_drop / max(d_keep, 1e-9)

    # 参照：若整帧能量均匀，loss_pct 应等于裁掉区面积占比
    area_baseline = 100.0 * a_drop / (a_keep + a_drop)

    if loss_pct >= 45.0:
        verdict = ("边缘承载大量运动（损失 %.1f%% 能量，面积基线 %.1f%%）"
                   "⇒ **支持 224**，P127 值得跑满" % (loss_pct, area_baseline))
    elif loss_pct >= 25.0:
        verdict = ("中性（损失 %.1f%%，面积基线 %.1f%%）⇒ 以 P127 配对结果为准"
                   % (loss_pct, area_baseline))
    else:
        verdict = ("运动高度集中在中心（损失仅 %.1f%%，面积基线 %.1f%%）"
                   "⇒ **160 不是瓶颈**，P127 若无配对收益即可止损"
                   % (loss_pct, area_baseline))

    print()
    log("──── 结果 ────")
    log("  保留区（中心 160²，面积 %.1f%%）：能量占比 %.1f%%"
        % (100.0 * a_keep / (a_keep + a_drop), 100.0 * E_keep / max(E_tot, 1e-9)))
    log("  裁掉区（外圈 48px，面积 %.1f%%）：能量占比 **%.2f%%**"
        % (area_baseline, loss_pct))
    log("  每像素能量密度比（裁掉区 / 保留区）= **%.3f**" % density_ratio)
    log("  判定：%s" % verdict)

    # 顺带：把能量图按 8×8 分块打印，直观看信息分布
    blk = E.reshape(8, FULL // 8, 8, FULL // 8).sum(axis=(1, 3))
    blk = blk / blk.max()
    log("  能量图（8×8，相对值；中间 5×5 是保留区）：")
    for r in range(8):
        log("    " + " ".join("%4.2f" % blk[r, c] for c in range(8)))

    out = REPO / "artifacts/metrics/blank-gov/p128-motion-distribution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "P128 运动能量空间分布：CenterCrop(160) 裁掉多少判别信息",
        "method": "帧间差分能量图（dt=%d），跨视频等权平均" % DT,
        "n_videos": n_used, "n_pairs": n_pair,
        "full_size": FULL, "crop_size": CROP, "keep_range": [LO, HI],
        "area_pct_keep": round(100.0 * a_keep / (a_keep + a_drop), 2),
        "area_pct_dropped": round(area_baseline, 2),
        "energy_pct_dropped": round(loss_pct, 2),
        "density_ratio_dropped_over_keep": round(density_ratio, 3),
        "energy_map_8x8": [[round(float(blk[r, c]), 3) for c in range(8)]
                           for r in range(8)],
        "verdict": verdict,
        "caveat": "帧间差分是运动的代理，不等价于手部解剖位置；"
                  "包含躯干/头部的运动。用于趋势判断，不用于精确结论。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)


if __name__ == "__main__":
    main()
