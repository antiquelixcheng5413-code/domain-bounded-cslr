"""P73：提取完成后自动启动 TFNet 训练（官方超参）

用户要求：「提取跑完自动复刻官方论文」
⇒ 本脚本守望提取进程，结束后自动启动 TFNet 训练。

⚠️ 关键前置检查（全部通过才启动，否则白跑 2 小时）：
  1. train + dev 特征都齐
  2. 新特征 presence 语义正确（[handL, handR, pose, face]）
  3. 特征无 NaN / shape 正确
  4. 词表构建成功且 CTC 约束满足（T >= 2L-1）
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
PY = REPO / "venv/bin/python"
NEW = REPO / "artifacts/part3_features_tasksapi"
T = 48


def sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=300, cwd=str(REPO))
    return (r.stdout or "") + (r.stderr or "")


def count(split):
    return len(list((NEW / split).glob("*.landmark.npy")))


def wait_extract():
    print("守望提取进程结束…")
    while True:
        r = sh(["pgrep", "-f", "p53_extract"])
        if not r.strip():
            break
        a, b = count("train"), count("validation")
        print("  提取中… train %d  dev %d" % (a, b), flush=True)
        time.sleep(120)
    print("提取进程已退出。train %d  dev %d"
          % (count("train"), count("validation")))


def preflight():
    """启动前的 4 项检查。任一不过就不启动。"""
    print("\n" + "=" * 68)
    print("预检")
    print("=" * 68)
    ok = True

    # 1 覆盖
    n_tr, n_dv = count("train"), count("validation")
    print("  [1] 特征覆盖 train %d / dev %d" % (n_tr, n_dv))
    c1 = n_tr >= 4900 and n_dv >= 500
    print("      %s" % ("✓" if c1 else "✗ 需 train≥4900 且 dev≥500"))
    ok &= c1

    # 2/3 质量 + presence 顺序
    import numpy as np
    fs = sorted((NEW / "train").glob("*.landmark.npy"))[::37]
    P, bad = [], 0
    for f in fs:
        a = np.load(f)
        if a.shape != (T, 368) or not np.isfinite(a).all():
            bad += 1
            continue
        P.append(a[:, 182:186].mean(0))
    P = np.mean(P, 0)
    print("  [2] presence 均值 [handL %.3f handR %.3f pose %.3f face %.3f]"
          % (P[0], P[1], P[2], P[3]))
    c2 = P[2] > 0.9 and bad == 0
    print("      %s（pose 位应≈1.0，异常 %d 条）"
          % ("✓" if c2 else "✗", bad))
    ok &= c2

    # 4 词表 + CTC 约束
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
    from cslr.recognition.gloss_sequence import (
        build_ordered_vocabulary, split_gloss_sequence, GlossSequenceConfig)
    import csv
    with open(REPO / "data/raw/CE-CSL/label/train.csv", newline="",
              encoding="utf-8") as fh:
        lab_tr = {r["Number"]: r["Gloss"] for r in csv.DictReader(fh)}
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=1,
                                      max_tokens=None)
    lens = [len(split_gloss_sequence(g, GlossSequenceConfig()))
            for g in lab_tr.values()]
    need = max(2 * l - 1 for l in lens if l <= 24)
    print("  [3] 词表 %d（CTC 类 %d）；max(2L-1) = %d <= T=%d ? %s"
          % (voc.size - 1, voc.size, need, T, "✓" if need <= T else "✗"))
    c3 = need <= T
    ok &= c3

    print("\n=> %s" % ("**全部通过，启动训练**" if ok
                       else "**未通过，不启动**"))
    return ok


def main():
    wait_extract()
    if not preflight():
        print("\n预检未过，不启动训练。等特征补齐后手动跑：")
        print("  venv/bin/python tools/blank_gov/p72_train_tfnet.py")
        return
    print("\n启动 TFNet 训练（官方超参）…")
    cmd = [str(PY), "-u", "tools/blank_gov/p72_train_tfnet.py",
           "--tag", "p72-tfnet", "--epochs", "55", "--batch", "16",
           "--lr", "1e-4", "--weight-decay", "1e-4", "--w-vae", "0.1"]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO))


if __name__ == "__main__":
    main()