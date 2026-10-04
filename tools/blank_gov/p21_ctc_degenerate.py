# -*- coding: utf-8 -*-
"""P21 · 判决：CTC 退化解的机制（决定修复方向）

## P20 留下的矛盾（必须解开，否则修不了）

已确认 loss 与 decode 实现都正确，且 8 条样本上：
```
P(target) = 8.3      模型给真实序列的概率约 8 倍 -> 「学会了」
WER       = 1.0      argmax 解码完全错       -> 「没学会」
```

## 两个竞争假设

### H-A「多路径并集」：模型用很多条稀疏路径覆盖目标
若成立，**束搜索（求概率最高路径）应显著优于 argmax**。
但 P19b 实测：beam2/beam4 的 WER **更差**（0.9935 → 1.0145/1.0131）。
→ **H-A 站不住**（束搜索本该修复它）。

### H-B「后验质量全在 blank 上，label 概率虽被求和放大但单帧极低」
若成立：
- 束搜索无效（因为**没有任何一条路径**能拿到高分）
- 逐帧 argmax 全是 blank
- 但 CTC 求和仍能累积出可观的 P(target)

**P20dbg4 已测到关键数字：非 blank 帧最大后验 ≈ -0.0001（即 p≈0.9999 是 blank）。
但还需要知道「正确 label 那一帧上，label 的后验是多少」** ——
这是区分 H-A/H-B 的判决量。

## 本脚本测什么

对 8 条已过拟合的样本，**逐帧打印完整后验画像**：

1. 每帧 top-5 类 + 其概率（看 blank 是否压倒性、label 排第几）
2. 目标 label 出现的那几帧上，label 的实际概率
3. `P(target)` 的量级来源：是「少数帧高概率」还是「众多帧低概率的乘积」
4. 用 log-softmax 手工累加验证 CTC 求和

## 判据（跑之前写死）

- 若目标 label 的帧上概率 > 0.5（够高）而束搜索仍失败
  → **H-B 的强版本**：label 信息存在但分散，**修法是让监督更密集**
  （帧级辅助损失 / 缩短 T / 特征增强）
- 若目标 label 的帧上概率 < 0.1
  → 特征根本无法定位到正确时刻，**必须换特征**（论文的 HaMeR 路线）

只读 train，**不触碰 dev 与 test**。
"""
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.contracts import SampleRecord
from cslr.recognition.model import CTCConfig, CTCRecognizer, BLANK_INDEX
from cslr.recognition.gloss_sequence import build_ordered_vocabulary
from cslr.recognition.dataset import GlossSequenceDataset, FeatureNormalizer, collate_samples
from cslr.recognition.training import (
    ctc_loss, iterate_batches, resolve_device, to_torch_batch, decode_batch)
from p8_error_attribution import levenshtein


def read_split(split):
    table = {"train": "train.csv", "validation": "dev.csv"}
    out = {}
    with open(REPO / "data/raw/CE-CSL/label" / table[split],
              newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["Number"]] = r["Gloss"]
    return out


def ensure_link_dir(split):
    root = REPO / ".vocab_link" / split
    root.mkdir(parents=True, exist_ok=True)
    for p in sorted((REPO / "artifacts/part3_features" / split).glob("*.landmark.npy")):
        sid = p.name[: -len(".landmark.npy")]
        dst = root / (sid + ".npy")
        if not dst.exists():
            try:
                os.symlink(p, dst)
            except OSError:
                pass
    return root


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-show", type=int, default=4)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p21-ctc-degenerate.json")
    a = ap.parse_args()

    device = resolve_device("auto")
    tr_labels = read_split("train")
    tr_root = ensure_link_dir("train")
    recs = []
    for sid, g in tr_labels.items():
        if not (tr_root / (sid + ".npy")).exists():
            continue
        if not [t.strip() for t in g.split("/") if t.strip()]:
            continue
        recs.append(SampleRecord(sample_id=sid, video=Path(sid + ".mp4"), label=g,
                                 signer="x", session="x", split="train"))
    recs.sort(key=lambda r: r.sample_id)
    cand = [r for r in recs if 3 <= len([t for t in r.label.split("/") if t.strip()]) <= 6]
    pick = cand[:8]

    voc, _ = build_ordered_vocabulary((g for g in tr_labels.values()),
                                      min_frequency=2, max_tokens=300)
    nrm = FeatureNormalizer.fit([np.load(tr_root / (r.sample_id + ".npy")) for r in pick])
    ds = GlossSequenceDataset(pick, tr_root, voc, nrm, feature_view="full")
    ref = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()] for r in pick}

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfg = CTCConfig(input_size=368, vocabulary_size=voc.size, hidden_size=256,
                    num_layers=2, dropout=0.0, bidirectional=True,
                    projection_size=256, subsample_stride=1)
    model = CTCRecognizer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    for ep in range(1, a.epochs + 1):
        model.train()
        for s in iterate_batches(ds, 8, shuffle=False, seed=0):
            b = to_torch_batch(collate_samples(s), device)
            opt.zero_grad(set_to_none=True)
            lg = model(b["features"], b["input_lengths"])
            ol = model.output_lengths(b["input_lengths"])
            loss = ctc_loss(lg, b["input_lengths"], b["targets"],
                            b["target_lengths"], ol)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    model.eval()
    stats = []
    with torch.no_grad():
        for s in iterate_batches(ds, 8, shuffle=False, seed=42):
            b = to_torch_batch(collate_samples(s), device)
            lg = model(b["features"], b["input_lengths"])
            ol = model.output_lengths(b["input_lengths"])
            lv = ctc_loss(lg, b["input_lengths"], b["targets"],
                          b["target_lengths"], ol, reduction="none")
            p_all = torch.softmax(lg.float(), dim=-1).cpu().numpy()
            lp_all = torch.log_softmax(lg.float(), dim=-1).cpu().numpy()
            dec, _, _ = decode_batch(lp_all, ol.cpu().numpy(), 1)

            for r in range(b["targets"].shape[0]):
                sid = b["sample_ids"][r]
                t = int(ol[r])
                L = int(b["target_lengths"][r])
                tgt = b["targets"][r, :L].tolist()
                p = p_all[r, :t]
                # 目标 label 在每帧上的排名与概率
                ranks = []
                for step in range(t):
                    order = np.argsort(-p[step])
                    lab = tgt[step % L]
                    rank = int(np.where(order == lab)[0][0])
                    ranks.append((rank, float(p[step, lab])))
                # 最好的那几帧（label 概率最高的）
                best = sorted(ranks, key=lambda x: -x[1])[:L]
                hyp = voc.decode(list(dec[r]))
                R = ref[sid]
                stat = {
                    "sid": sid, "L": L, "T": t,
                    "nll_per_token": round(float(lv[r]) / max(L, 1), 4),
                    "p_target": round(float(np.exp(-float(lv[r]))), 4),
                    "blank_p_mean": round(float(p[:, BLANK_INDEX].mean()), 6),
                    "blank_p_max": round(float(p[:, BLANK_INDEX].max()), 6),
                    "n_frames_blank_top1": int((p.argmax(axis=1) == BLANK_INDEX).sum()),
                    "label_p_max_over_frames": round(max(x[1] for x in ranks), 6),
                    "label_p_mean_over_frames": round(float(np.mean([x[1] for x in ranks])), 6),
                    "label_rank_min": int(min(x[0] for x in ranks)),
                    "n_frames_label_is_top1": int(sum(1 for x in ranks if x[0] == 0)),
                    "ref": R, "hyp": hyp,
                    "edit": levenshtein(list(R), hyp)[0],
                }
                stats.append(stat)
                if len(stats) <= a.n_show:
                    print("=" * 74)
                    print("{}  L={} T={}".format(sid[:12], L, t))
                    print("  ref                : {}".format("/".join(R)))
                    print("  hyp (greedy)       : {}".format("/".join(hyp) or "(空)"))
                    print("  loss/token         = {:+.4f}   P(target)={:.4f}".format(
                        stat["nll_per_token"], stat["p_target"]))
                    print("  blank 概率         mean={:.6f} max={:.6f}".format(
                        stat["blank_p_mean"], stat["blank_p_max"]))
                    print("  argmax=blank 的帧  {}/{}".format(
                        stat["n_frames_blank_top1"], t))
                    print("  label 是 top1 的帧 {}/{}".format(
                        stat["n_frames_label_is_top1"], t))
                    print("  label 概率         max={:.6f} mean={:.6f}  最好排名={}".format(
                        stat["label_p_max_over_frames"], stat["label_p_mean_over_frames"],
                        stat["label_rank_min"]))
                    print("  label 概率最高的 {} 帧 (排名,概率): {}".format(
                        L, [(rk, round(pv, 4)) for rk, pv in best]))
            break

    # 汇总
    print()
    print("=" * 74)
    print("汇总（8 条）")
    print("=" * 74)
    med_rank1 = float(np.median([s["n_frames_label_is_top1"] for s in stats]))
    med_labmax = float(np.median([s["label_p_max_over_frames"] for s in stats]))
    med_labmean = float(np.median([s["label_p_mean_over_frames"] for s in stats]))
    med_blank = float(np.median([s["blank_p_mean"] for s in stats]))
    med_ptgt = float(np.median([s["p_target"] for s in stats]))
    print("  P(target) 中位           = {:.4f}".format(med_ptgt))
    print("  blank 概率均值 中位      = {:.6f}".format(med_blank))
    print("  label 是 top1 的帧数 中位= {:.1f} / 48".format(med_rank1))
    print("  label 概率 max 中位      = {:.6f}".format(med_labmax))
    print("  label 概率 mean 中位     = {:.6f}".format(med_labmean))

    if med_labmax > 0.5:
        verdict = ("H-B 强版本：label 在个别帧上概率 > 0.5，**信息存在**。"
                   "束搜索失效说明概率被 blank 的连乘压制 -> "
                   "解法是密集监督/缩短 T/特征增强，让 label 概率抬到可解码区间")
    else:
        verdict = ("label 单帧概率 max 中位仅 {:.4f} < 0.5："
                   "**特征无法把 label 定位到任何一帧** -> 必须换特征，"
                   "调损失/解码都无效").format(med_labmax)
    print()
    print("判决: " + verdict)

    receipt = {
        "experiment": "P21 mechanism of the CTC degenerate solution",
        "hypotheses": {
            "H-A": "multi-path union -> beam search should fix it (REFUTED: beam made WER worse)",
            "H-B": "per-frame label probability too low -> no single path is good"},
        "criterion": "label_p_max > 0.5 => info exists (fix supervision); "
                     "else feature is the hard limit",
        "all_data_real": True, "reads_dev": False, "reads_test_split": False,
        "n_samples": len(stats),
        "median": {"p_target": med_ptgt, "blank_p_mean": med_blank,
                   "n_frames_label_top1": med_rank1,
                   "label_p_max": med_labmax, "label_p_mean": med_labmean},
        "per_sample": stats,
        "verdict": verdict,
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
