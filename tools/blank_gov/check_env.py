#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""环境自检：跑 D1/D3/D4 之前先执行这个，缺什么会直接告诉你。

用法
  python3 tools/blank_gov/check_env.py
"""
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OK, WARN, MISS = "✅", "⚠️ ", "❌"


def main() -> int:
    print("=" * 56)
    print(" CE-CSL 实验环境自检（仓库：%s）" % REPO)
    print("=" * 56)

    # 1. Python / torch
    print("\n[1] Python 与 torch")
    print("    python: %s" % sys.version.split()[0])
    try:
        import torch
        print("    %s torch %s  CUDA available=%s" % (OK, torch.__version__,
                                                      torch.cuda.is_available()))
        if torch.cuda.is_available():
            print("    %s GPU: %s  显存 %.1f GB" % (
                OK, torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9))
            cc = torch.cuda.get_device_capability(0)
            bf16 = cc[0] >= 8
            print("    %s Compute Capability %d.%d ⇒ bf16 %s（训练脚本用 --amp-bf16）"
                  % (OK if bf16 else MISS, cc[0], cc[1],
                     "支持" if bf16 else "不支持，需去掉 --amp-bf16"))
    except ImportError:
        print("    %s torch 未安装：pip install torch torchvision" % MISS)
        return 1

    # 2. 其他依赖
    print("\n[2] 依赖")
    for mod, pip in (("cv2", "opencv-python"), ("numpy", "numpy"),
                     ("tqdm", "tqdm")):
        try:
            __import__(mod)
            print("    %s %s" % (OK, mod))
        except ImportError:
            print("    %s %s 未装：pip install %s" % (MISS, mod, pip))

    # 3. 数据
    print("\n[3] 数据（解压到 artifacts/official_rgb/ 下）")
    root = REPO / "artifacts" / "official_rgb"
    for split, need in (("dev", 95891), ("train", 930841)):
        d = root / split
        if not d.exists():
            print("    %s %s/ 不存在 ⇒ 需要放数据" % (MISS, split))
            continue
        n = sum(1 for _ in d.rglob("*.jpg"))
        flag = OK if n >= need * 0.99 else (WARN if n > 0 else MISS)
        print("    %s %s/  帧数 %d（期望 %d）" % (flag, split, n, need))

    # 4. CSV 词表
    print("\n[4] 数据集 CSV")
    csvd = REPO / "data" / "CE-CSL"
    if not csvd.exists():
        # 常见备选位置
        cands = [REPO / "external" / "TFNet" / "data", REPO / "data"]
        csvd = next((c for c in cands if list(c.glob("*.csv"))), csvd)
    csvs = sorted(csvd.glob("*.csv")) if csvd.exists() else []
    if csvs:
        print("    %s 找到 %d 个 csv：%s" % (OK, len(csvs),
                                        [c.name for c in csvs[:6]]))
    else:
        print("    %s 没找到 csv（train/dev/test）⇒ 检查 data/ 目录" % MISS)

    # 5. 官方权重（只有 D1 需要）
    print("\n[5] 官方权重（仅 D1 需要）")
    found = []
    for base in (REPO.parent, Path("/mnt/c/Users"), REPO):
        for p in base.rglob("TFNet-CE-CSL-CSLDaily*.pth") if base.exists() else []:
            found.append(p)
            break
    if found:
        p = found[0]
        print("    %s %s（%.0f MB）" % (OK, p.name, p.stat().st_size / 1e6))
    else:
        print("    %s 未找到 TFNet-CE-CSL-CSLDaily-32.46.pth（只有 D1 需要，D3/D4 可忽略）" % WARN)

    # 6. 磁盘
    print("\n[6] 磁盘空间")
    total, used, free = shutil.disk_usage(str(REPO))
    print("    %s 剩余 %.0f GB" % (OK if free / 1e9 > 30 else WARN, free / 1e9))

    print("\n" + "=" * 56)
    print(" 下一步：先跑冒烟，再上 55 轮")
    print("   D3: bash tools/blank_gov/run_d3_vac224.sh smoke")
    print("   D4: bash tools/blank_gov/run_d4_tfnet224.sh smoke")
    print("   D1: 已由我们跑完，你们不用跑（且需要 880MB 官方权重，不需传）")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
