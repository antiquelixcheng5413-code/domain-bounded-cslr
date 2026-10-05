# -*- coding: utf-8 -*-
"""P45b · dev 的 signer 是否也在 train 里？（泛化难度的关键）

## 为什么查
CE-CSL 每个 signer 录同一套句子，A~L 共 12 个 signer。
若 dev 的 signer 也在 train，则模型见过「这个人做这个动作」，
泛化难度远低于跨 signer；若不重叠，则 P7 测到的 signer 间 CV=0.018
就是当前 WER 上不去的合理解释。

## 做法
从视频目录名（A~L）反查每个 sample_id 属于哪个 signer。
ID 到 signer 的映射用文件实际位置，不靠猜。
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

TRAIN_V = Path("/mnt/c/Users/su127/Desktop/csl视频/train")
DEV_V = Path("/mnt/c/Users/su127/Desktop/csl视频/dev")

# 特征目录里实际存在的文件 = 真正可用的样本
FEAT = Path("/home/su127/FYP/domain-bounded-cslr/artifacts/part3_features")


def scan(video_root: Path, feat_split: str) -> dict:
    """id -> signer 字母（只统计特征文件存在的）"""
    out = {}
    if not video_root.exists():
        return out
    for d in sorted(video_root.iterdir()):
        if not d.is_dir():
            continue
        signer = d.name
        for v in sorted(d.glob("*.mp4")):
            sid = v.stem
            if (FEAT / feat_split / (sid + ".landmark.npy")).exists():
                out[sid] = signer
    return out


def main() -> None:
    tr = scan(TRAIN_V, "train")
    dv = scan(DEV_V, "validation")
    print("train 可用样本 = %d，signer 数 = %d"
          % (len(tr), len(set(tr.values()))))
    print("dev   可用样本 = %d，signer 数 = %d"
          % (len(dv), len(set(dv.values()))))

    ctr = collections.Counter(tr.values())
    cdv = collections.Counter(dv.values())
    print("\n%-4s %10s %10s" % ("signer", "train", "dev"))
    for s in sorted(set(ctr) | set(cdv)):
        print("%-4s %10d %10d" % (s, ctr.get(s, 0), cdv.get(s, 0)))

    str_ = set(ctr)
    sdv = set(cdv)
    ov = str_ & sdv
    print("\n=== 关键 ===")
    print("train signer: %s" % sorted(str_))
    print("dev   signer: %s" % sorted(sdv))
    print("交集: %s" % sorted(ov))
    print("dev signer 也出现在 train 的比例: %d/%d = %.1f%%"
          % (len(ov), len(sdv), 100 * len(ov) / max(len(sdv), 1)))
    n_dv_ov = sum(cdv.get(s, 0) for s in ov)
    print("落在重叠 signer 上的 dev 样本: %d/%d = %.1f%%"
          % (n_dv_ov, len(dv), 100 * n_dv_ov / max(len(dv), 1)))

    # 特征文件的 ID 与视频 ID 是否一致
    print("\n=== 特征文件覆盖 ===")
    print("train 特征数 = %d" % len(list((FEAT / "train").glob("*.landmark.npy"))))
    print("valid 特征数 = %d" % len(list((FEAT / "validation").glob("*.landmark.npy"))))

    receipt = {
        "experiment": "P45b", "date": "2026-10-05",
        "question": "dev 的 signer 是否也在 train 里",
        "train_samples": len(tr), "dev_samples": len(dv),
        "train_signers": sorted(str_), "dev_signers": sorted(sdv),
        "overlap_signers": sorted(ov),
        "dev_signers_seen_in_train_ratio": round(len(ov) / max(len(sdv), 1), 4),
        "dev_samples_on_overlapping_signers": n_dv_ov,
        "dev_samples_ratio_on_overlap": round(n_dv_ov / max(len(dv), 1), 4),
        "per_signer": {s: {"train": ctr.get(s, 0), "dev": cdv.get(s, 0)}
                       for s in sorted(set(ctr) | set(cdv))},
        "implication": (
            "若交集为 0（跨 signer 泛化），则 P7 测到的 signer 间 CV=0.018 "
            "就是 WER 上不去的主因之一，且与「特征判别力不足」是同一件事的两面"
            if not ov else
            "dev 与 train 共享 signer，泛化难度低于跨 signer 场景"),
    }
    p = Path("/home/su127/FYP/domain-bounded-cslr/artifacts/metrics/"
             "blank-gov/p45b-signer-overlap.json")
    p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
