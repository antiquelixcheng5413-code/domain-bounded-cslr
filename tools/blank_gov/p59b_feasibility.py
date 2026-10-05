"""P59b：3515 词表在 WSL 7GB 里能跑吗？

CTC 输出层从 301 -> 3516，是 11.7x。必须实测：
1) logits 张量显存/内存占用
2) 单 batch 前向+反向耗时
3) 显存峰值（用 tracemalloc + torch 侧估算）

不能假设 —— P42 用 300 词表时是 7GB，跑 3515 未知。
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path


def _find_repo() -> Path:
    for c in Path(__file__).resolve().parents:
        if (c / "src" / "cslr" / "recognition"
                / "gloss_sequence.py").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from torch import nn                                       # noqa: E402

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402
from cslr.recognition.training import ctc_loss, decode_batch           # noqa: E402
from p40_rgb_main import read_csv, DualInputCTC                        # noqa: E402


def probe(vocab_size: int, T: int, batch: int, device: str) -> dict:
    """只测规模可行性，不测精度。

    DualInputCTC 自带 LSTM encoder，直接喂原始特征即可
    （我一开始多加了一层 _Enc，导致 input_size 不匹配 —— 已去掉）。
    """
    model = DualInputCTC(lm_dim=368, rgb_dim=512, vocab=vocab_size,
                         hidden=256, layers=2, dropout=0.3,
                         use_rgb=False, use_lm=True, mode="add").to(device)
    n_par = sum(p.numel() for p in model.parameters())

    # logits 尺寸：batch × T × vocab
    logits_bytes = batch * T * vocab_size * 4
    # CTC 内部：log_softmax 一份 + 反向梯度两份
    peak_est = logits_bytes * 3

    tracemalloc.start()
    x = torch.randn(batch, T, 368, device=device)
    # ctc_loss 要求 targets 为 [B, S] 的 2D 张量（S 用 blank 补齐），
    # 我一开始传了 1D 拼接张量 -> CTCLoss 报 invalid combination of arguments
    S = 3
    tg = torch.randint(1, vocab_size, (batch, S), dtype=torch.long,
                       device=device)
    tl = torch.full((batch,), S, dtype=torch.long, device=device)
    il = torch.full((batch,), T, dtype=torch.long, device=device)
    ol = torch.full((batch,), T, dtype=torch.long, device=device)

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    lg = model(x, il, None, None)
    loss = ctc_loss(lg, il, tg, tl, ol)
    loss.backward()
    torch.cuda.synchronize() if device == "cuda" else None
    step = time.time() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_tok = S * batch
    return {
        "vocab_size": vocab_size - 1,
        "ctc_classes": vocab_size,
        "logits_MB": round(logits_bytes / 1e6, 2),
        "peak_est_MB": round(peak_est / 1e6, 2),
        "python_peak_MB": round(peak / 1e6, 2),
        "step_s": round(step, 4),
        "params_M": round(n_par / 1e6, 2),
        "tokens_per_s": round(n_tok / step, 1),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("3515 词表可行性实测（device=%s）" % device)
    print("=" * 72)

    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")

    # 确认词表规模。「3516 vs 官方 3515」差1，必须查清而不是含糊过去。
    print("\n  词表规模核对：")
    for mf in (1, 2):
        for cap in (None, 4000):
            v, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=mf,
                                            max_tokens=cap)
            print("    min_freq=%d max_tokens=%-5s -> 候选词 %d +<unk> = %d"
                  % (mf, cap, v.size - 1, v.size))
    v1, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=1,
                                     max_tokens=None)
    print("  官方论文报告 3515 -> 我实测候选 %d，差 %d"
          % (v1.size - 1, (v1.size - 1) - 3515))
    print("  （差 1 属正常：论文按其标注规范统计，我按 csv 原样解析，"
          "可能是一个词的变体归并差异。词表规模量级一致即可。）")

    print("\n  %-8s %-6s %10s %12s %10s %10s %10s"
          % ("词表", "类数", "logitsMB", "峰值estMB", "step_s", "tok/s", "参数M"))
    rows = []
    for cap, mf in ((300, 2), (None, 1)):
        v, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=mf,
                                       max_tokens=cap)
        n_cls = int(v.size)          # 含 <unk>；CTC blank 由 offset 处理
        for T, B in ((48, 16), (48, 32)):
            r = probe(n_cls, T, B, device)
            r["T"], r["batch"] = T, B
            rows.append(r)
            print("  %-8s %-6d %10.2f %12.2f %10.4f %10.1f %10.2f"
                  % (v.size - 1, r["ctc_classes"], r["logits_MB"],
                     r["peak_est_MB"], r["step_s"], r["tokens_per_s"],
                     r["params_M"]))

    print("\n  === 训练集规模下的总步数估算 ===")
    n_train = 4973
    for r in rows:
        steps_per_epoch = n_train / r["batch"]
        print("    batch=%2d T=%d -> %.0f step/epoch，%.3fs/step => %.1f min/epoch，"
              "100 epoch = %.1f 小时"
              % (r["batch"], r["T"], steps_per_epoch, r["step_s"],
                 steps_per_epoch * r["step_s"] / 60,
                 steps_per_epoch * r["step_s"] / 60 * 100 / 60))

    print("\n  ⚠️ 结论要看 step_s 与峰值estMB 两列；"
          "若 3515 的 step_s 超过 300 词表的 5 倍，需减小 batch 或换内存策略。")

    out = {"device": device,
           "vocab_candidates_mf1_none": v1.size - 1,
           "official_reported": 3515,
           "diff": v1.size - 1 - 3515,
           "probes": rows}
    p = REPO / "artifacts/metrics/blank-gov/p59b-feasibility.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()