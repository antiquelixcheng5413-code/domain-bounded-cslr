#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""技术探测（非实验）：测 repeat×2 最长样本 928 帧前向的显存峰值，不解码、不算 WER。

只回答「fp32 / bf16 下 8G 能否跑得下」，用于写死预注册里的显存失败处理规则。
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (REPO / "tools/blank_gov", REPO / "external/TFNet", REPO / "src"):
    sys.path.insert(0, str(p))

import p78_train_official_tfnet as P78  # noqa: E402
from p197_beam_lm import build_model  # noqa: E402

P78.OFFICIAL_PAD = True
P78.RGB_FIX = True
P78.BLANK_BIAS = 0.0

dev = "cuda"
_, n, _ = P78.DPM.Word2Id(
    str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
    str(P78.CSV_DIR / "test.csv"), "CE-CSL")
mdl = build_model(str(REPO / "artifacts/checkpoints/p184-tfnet-freeze-best.pt"),
                  1024, n, dev).eval()

for T, tag in ((464, "step=1 最长原始"), (928, "repeat×2 最长")):
    for amp in ("fp32", "bf16"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            vid = torch.zeros(1, T, 3, 160, 160, device=dev)
            dl = torch.LongTensor([[T]])
            with torch.no_grad():
                if amp == "bf16":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = mdl(vid, dl, False)
                else:
                    out = mdl(vid, dl, False)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            print("%-22s %-5s OK  peak=%.0f MiB  out0=%s"
                  % (tag, amp, peak, tuple(out[0].shape)), flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print("%-22s %-5s OOM  %s" % (tag, amp, str(e)[:100]), flush=True)
        finally:
            del vid
            torch.cuda.empty_cache()
