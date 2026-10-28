#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P171 初始化转换器：把官方发布 TFNet 权重打包成 p78 可直接 --resume 的初始化 ckpt。

为什么不直接 --resume 发布权重：发布权重顶层是 moduleNet_state_dict（无 optimizer_state），
而 p78 的 --resume 要求 optimizer_state（否则 raise SystemExit）。本脚本：
  1. 用与 p78 完全相同的 Net.moduleNet(\"TFNet\") 构建模型；
  2. 载入发布 moduleNet_state_dict（strict=True，p156 已验证逐键吻合）；
  3. 新建 optimizer（Adam 1e-4 / wd 1e-4）和 word_classifier（随机，p78 的自加多任务头）；
  4. 存成 p78 兼容 ckpt：model_state / optimizer_state / word_classifier_state / epoch=0。

用法：
  python tools/blank_gov/make_p171_init.py --word-head 0.3 --out artifacts/checkpoints/p171-init-tfnet160.pt
"""
import argparse
import sys
import torch
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external" / "TFNet"))
sys.path.insert(0, str(REPO / "src"))

import p78_train_official_tfnet as P  # noqa: E402

DEFAULT_RELEASED = "/mnt/c/Users/su127/Desktop/中文手语识别/官方权重/TFNet-CE-CSL-CSLDaily-32.46.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", default=DEFAULT_RELEASED)
    ap.add_argument("--word-head", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--out", default=str(REPO / "artifacts/checkpoints/p171-init-tfnet160.pt"))
    ap.add_argument("--total-epochs", type=int, default=30)
    a = ap.parse_args()

    print("[make_p171] 载入词表…")
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    print("[make_p171] wordSetNum =", wordSetNum)

    print("[make_p171] 构建 Net.moduleNet(TFNet)…")
    model = P.Net.moduleNet(a.hidden, wordSetNum + 1, "TFNet",
                            torch.device("cpu"), "CE-CSL", True, dropout=0.0)

    print("[make_p171] 载入发布 moduleNet_state_dict (strict=True)…")
    sd = torch.load(a.released, map_location="cpu", weights_only=False,
                    mmap=True)["moduleNet_state_dict"]
    miss, unexp = model.load_state_dict(sd, strict=True)
    print("[make_p171] load strict=True OK | missing=%d unexpected=%d" % (len(miss), len(unexp)))

    params = list(model.parameters())
    word_classifier = None
    if a.word_head > 0:
        word_classifier = torch.nn.Linear(wordSetNum + 1, wordSetNum + 1)
        params += list(word_classifier.parameters())
        print("[make_p171] word_classifier 已建（随机初始化）")

    opt = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-4)
    # 新建 sched 仅用于存 sched_state（p78 续训时会覆盖/恢复，这里给个初值即可）
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(round(35*a.total_epochs/55)), int(round(45*a.total_epochs/55))],
        gamma=0.2)

    ck = {
        "epoch": 0,
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "sched_state": sched.state_dict(),
        "word_classifier_state": word_classifier.state_dict() if word_classifier is not None else None,
        "wer_official": float("inf"),
        "history": [],
        "total_epochs_planned": a.total_epochs,
        "config": {
            "module": "TFNet", "hidden": a.hidden, "img_size": 160,
            "word_head": a.word_head, "dropout": 0.0,
            "lr": 1e-4, "weight_decay": 1e-4,
            "note": "from released TFNet-CE-CSL-CSLDaily-32.46.pth (resnet34MAM), fine-tune @160+RGB",
        },
    }
    torch.save(ck, a.out)
    print("[make_p171] 已保存初始化 ckpt →", a.out)
    print("[make_p171] size MB = %.1f" % (Path(a.out).stat().st_size/1e6))


if __name__ == "__main__":
    main()
