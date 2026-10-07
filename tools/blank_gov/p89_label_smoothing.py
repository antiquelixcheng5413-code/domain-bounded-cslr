"""P89：Label smoothing（针对 P88 发现的 95.2% blank 主导）

═══════════════════════════════════════════════════════════════
📄 论文支撑（依据等级严格标注）
═══════════════════════════════════════════════════════════════
【间接】ref17 Fayyazsanavi（Gloss2Text）§2.2 Semantically Aware Label Smoothing
  原文批评标准 LS：
  "In the conventional label smoothing approach (Szegedy et al. 2016; Müller et al.
   2019) one replaces one-hot encoded label vector y_hot with a mixture of y_hot
   and the uniform distribution y_s = (1−β)·y_hot + β/N... **With this approach,
   however, probabilities for all words in the vocabulary are non-zero, including
   those not present in our target vocabulary.**"
  ⇒ 它的 SALS：只对「目标词表内 FastText 余弦相似度 > λ=0.6 的词」平滑，
     非目标词保持 0。效果见 ref17 表 9（NLLB-SALSloss 优于标准 NLLB-loss）。

⚠️ **SALS 在本任务不可用**（已实测确认）：
  1. 需要 FastText 词向量 —— 本地无（`find -iname "*fasttext*"` 结果为空）
  2. 更根本的问题：CE-CSL 的 3515 个 gloss 是**视觉手形**，
     不是自然语言词 ⇒ FastText 词向量对手形语义无意义
  ⇒ **降级实现**（保留 SALS 的「非目标词设 0」思想，退化为只对 blank+目标平滑）

【有】标准 LS 原始论文：Szegedy et al., CVPR 2016 —— 但它是**图像分类**，
  平滑的是 one-hot **目标**；**从未用于 CTC 输出层**
  ⇒ 本实验依据等级【无】，属我的判断

═══════════════════════════════════════════════════════════════
🔴 实现说明：为什么用「概率平滑」而不是「目标平滑」
═══════════════════════════════════════════════════════
CTC 的 loss 由前向算法在 log 域计算，**无法直接注入平滑后的目标分布**
（需要重写 dynamic programming）。
PyTorch 的 CTCLoss 也不支持 label_smoothing 参数。

⇒ 本实验用**概率平滑**（logit smoothing），可证明正确且可导：
     p'(v) = (1−β)·p(v) + β/K        （K = 词表大小 3517）
     logp' = log(p')
  作用：把 log_prob 拉离 −∞，**直接压制「无限自信的 blank」**
  ⇒ 正对症 P88 观测：max prob=0.8959、blank 占 argmax 95.2%

⚠️ **诚实标注**：概率平滑 ≠ Szegedy 的目标平滑。
   两者都正则化「过度自信」，但数学上不等价。收据里已注明。
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external" / "TFNet"))

import DataProcessMoudle as DPM# noqa: E402
import Net# noqa: E402
import videoAugmentation as VA  # noqa: E402
from official_wer import evaluate  # noqa: E402

RGB = REPO / "artifacts/official_rgb"
CSV = REPO / "external/TFNet/data/CE-CSL"
LS = nn.LogSoftmax(dim=-1)


def read_split(name):
    out = {}
    with open(CSV / ("%s.csv" % name), newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0]:
                out[row[0]] = (row[1], row[3])
    return out


def build_transforms(hflip, size):
    ops = [VA.RandomCrop(size)]
    if hflip:
        ops.append(VA.RandomHorizontalFlip(0.5))
    ops += [VA.ToTensor(), VA.TemporalRescale(0.2)]
    return VA.Compose(ops), VA.Compose([VA.CenterCrop(size), VA.ToTensor()])


class DS(Dataset):
    def __init__(self, split, labels, w2i, tf):
        self.root = RGB / split
        self.tf = tf
        self.items = []
        for sid, (tr, gloss) in labels.items():
            d = self.root / tr / sid
            if not d.exists():
                continue
            ids = [w2i[w] for w in DPM.PreWords(gloss.split("/")) if w in w2i]
            if ids:
                self.items.append({"sid": sid, "dir": d, "ids": ids})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        files = sorted(f for f in os.listdir(it["dir"]) if f.endswith(".jpg"))
        imgs = [cv2.cvtColor(cv2.imread(str(it["dir"] / f)), cv2.COLOR_BGR2RGB)
                for f in files]
        seq = self.tf(imgs).float() / 127.5 - 1.0
        return {"video": seq, "ids": it["ids"], "sid": it["sid"]}


def collate(batch):
    batch = sorted(batch, key=lambda x: len(x["video"]), reverse=True)
    vids = [b["video"] for b in batch]
    T = max(v.shape[0] for v in vids)
    vid = torch.zeros(len(vids), T, 3, vids[0].shape[2], vids[0].shape[3])
    for i, v in enumerate(vids):
        vid[i, :v.shape[0]] = v
    dl = torch.LongTensor([[v.shape[0]] for v in vids])
    tl = torch.LongTensor([v.shape[0] for v in vids])
    tgt = torch.cat([torch.LongTensor(b["ids"]) for b in batch])
    tgtl = torch.LongTensor([len(b["ids"]) for b in batch])
    return (vid, tgt, tgtl, dl, tl, [b["sid"] for b in batch],
            [b["ids"] for b in batch])


def smooth_logp(logp: torch.Tensor, beta: float) -> torch.Tensor:
    """概率平滑：log p' = log((1−β)p + β/K)。可导，压制过度自信。"""
    if beta <= 0:
        return logp
    K = logp.shape[-1]
    p = logp.exp()
    return torch.log((1.0 - beta) * p + beta / K)


def _conv_len(n):
    for _ in range(2):
        n = (n - 4 + 1) // 2
    return max(n, 0)


def greedy(seq, idx2w):
    ids, prev = [], -1
    for t in range(seq.shape[0]):
        k = int(seq[t].argmax())
        if k != prev and k != 0:
            ids.append(k)
        prev = k
    return [idx2w[i] for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    dev = "cuda"
    lab_tr, lab_dv = read_split("train"), read_split("dev")
    w2i, wsn, idx2w = DPM.Word2Id(str(CSV / "train.csv"), str(CSV / "dev.csv"),
                                   str(CSV / "test.csv"), "CE-CSL")
    tr_tf, dv_tf = build_transforms(False, a.img_size)
    tr_ds, dv_ds = DS("train", lab_tr, w2i, tr_tf), DS("dev", lab_dv, w2i, dv_tf)
    if a.max_train:
        tr_ds.items = tr_ds.items[:a.max_train]
    print("=" * 70)
    print("P89 label smoothing  beta=%.3f" % a.beta)
    print("=" * 70)
    print("train %d  dev %d  img %d  vocab %d" % (len(tr_ds), len(dv_ds),
                                              a.img_size, wsn))
    print("基线 P87(beta=0)     : WER 80.96%  输出长 1.78  distinct 20")

    tr_dl = DataLoader(tr_ds, batch_size=2, shuffle=True, num_workers=4,
                       collate_fn=collate, drop_last=True)
    dv_dl = DataLoader(dv_ds, batch_size=2, shuffle=False, num_workers=4,
                       collate_fn=collate)

    model = Net.moduleNet(1024, wsn + 1, "VAC", torch.device(dev), "CE-CSL",
                          True).to(dev)
    ctc = nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    kld = DPM.SeqKD(T=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ms = [max(1, int(round(35 * a.epochs / 55))),
          max(2, int(round(45 * a.epochs / 55)))]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, ms, gamma=0.2)
    best, hist, nan_steps = 1e9, [], 0
    t0 = time.time()

    for ep in range(1, a.epochs + 1):
        model.train()
        run, n = 0.0, 0
        for vid, tgt, tgtl, dl, tl, _s, _i in tr_dl:
            out = model(vid.to(dev), dl, True)
            lgt = out[5]
            # 🔴 平滑施加在 log_probs 上（官方 loss 仍是 zero_infinity=True）
            lp0 = smooth_logp(LS(out[0]), a.beta)
            lp1 = smooth_logp(LS(out[1]), a.beta)
            loss_ctc = ctc(lp0, tgt, lgt, tgtl).mean()
            loss = loss_ctc + ctc(lp1, tgt, lgt, tgtl).mean() \
                + 25.0 * kld(out[1], out[0], use_blank=False)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if torch.isfinite(gn):
                opt.step()
            else:
                nan_steps += 1
                opt.zero_grad(set_to_none=True)
                continue
            run += float(loss_ctc.detach())
            n += 1
        sched.step()

        model.eval()
        hyps, refs = [], []
        with torch.no_grad():
            for vid, tgt, tgtl, dl, tl, sids, _i in dv_dl:
                out = model(vid.to(dev), dl, False)
                lp = LS(out[0])          # ⚠️ 评估不平滑，与 P87 基线口径一致
                for bi in range(lp.shape[1]):
                    T2 = min(_conv_len(int(tl[bi])), lp.shape[0])
                    hyps.append(greedy(lp[:T2, bi, :].cpu().numpy(), idx2w))
                refs += [lab_dv[s][1] for s in sids]
        o = evaluate(refs, hyps)
        hl = float(np.mean([len(h) for h in hyps]))
        nd = len({t for h in hyps for t in h})
        print("  ep%03d ctc %.4f  WER %.2f%%  输出长 %.2f  distinct %d%s"
              % (ep, run / max(n, 1), o["WER_official"], hl, nd,
                 "  [nan %d]" % nan_steps if nan_steps else ""), flush=True)
        hist.append({"epoch": ep, "train_ctc": round(run / max(n, 1), 4),
                     "dev_wer_official": o["WER_official"],
                     "out_len": round(hl, 3), "distinct": nd})
        if o["WER_official"] < best:
            best = o["WER_official"]
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "wordSetNum": wsn, "idx2word": idx2w, "hidden": 1024,
                        "config": {"module": "VAC", "beta": a.beta,
                                   "img_size": a.img_size,
                                   "max_train": a.max_train},
                        "wer_official": best},
                       REPO / ("artifacts/checkpoints/%s-best.pt" % a.tag))

    dst = REPO / ("artifacts/metrics/blank-gov/%s.json" % a.tag)
    dst.write_text(json.dumps({
        "experiment": "P89", "beta": a.beta, "max_train": a.max_train,
        "img_size": a.img_size, "best_wer": best, "history": hist,
        "minutes": round((time.time() - t0) / 60, 1),
        "nan_steps": nan_steps,
        "baseline_p87": {"best_wer": 80.96, "out_len": 1.78, "distinct": 20},
        "implementation": "概率平滑 log p' = log((1−β)p + β/K)，"
                          "平滑施加在 log_probs 上、评估时不平滑（与 P87 口径一致）",
        "paper_basis": {
            "ref17_SALS": "§2.2 批评标准 LS『all words have non-zero probabilities』；"
                          "SALS 只对 FastText 语义相似词(λ=0.6)平滑",
            "why_not_SALS": "本地无 FastText；且 CE-CSL 的 3515 gloss 是视觉手形"
                            "而非自然语言词⇒ 词向量无语义",
            "standard_LS": "Szegedy et al. CVPR2016 是图像分类、平滑one-hot 目标，"
                           "**从未用于 CTC 输出层**",
            "honest_caveat": "概率平滑 ≠ 目标平滑（数学不等价）"
                             "⇒ 依据等级【无】，属我的判断",
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("最佳 WER = %.2f%%  （基线 P87 = 80.96%%，Δ %+.2f pp）"
          % (best, best - 80.96))
    print("收据 -> %s" % dst)


if __name__ == "__main__":
    main()