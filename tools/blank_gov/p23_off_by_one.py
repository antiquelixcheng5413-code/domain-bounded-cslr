# -*- coding: utf-8 -*-
"""P23 · 判决性验证：CTC target 缺少 +1 导致训练/解码索引错位

## 已定位的 bug（train-00004 实测）

```
CSV ref        : 美/女/一/。
target_ids     : [249, 0, 45, 1]      <- 词表索引（0 起）
第 0 帧 argmax  : class 249
解码后 token    : 网                    <- voc.tokens[248]
期望 token      : 美                    <- voc.tokens[249]
```

`vocab.encode` 返回**词表索引**（`<unk>`=0），但 CTC 的类空间是
**blank=0，词表索引 i 对应类 i+1**（见 model.py 注释与
`classes_to_token_ids` 的 `index - 1`）。

`collate_samples`（dataset.py:218-221）把 `token_ids` 原样塞进 targets，
**没有 +1**。于是：

- 训练时 target「美」= 类 249，而模型输出类 249 时解码器还原成词表 248 = 「网」
- **训练目标与解码口径整体错位一格（off-by-one）**
- 更糟：词表索引 0（`<unk>`）被当成 **CTC blank**，
  于是 `<unk>` 在训练里**根本不是要输出的类，而是 blank**

## 这一个 bug 解释此前全部现象

| 现象 | 解释 |
|---|---|
| blank 率 97% | 词表索引 0 = `<unk>` 占训练目标 28.3%，被当成 blank 训练 |
| 28.3% `<unk>` 是坍缩燃料 | 它们根本没被当作要预测的类 |
| P(target) 极高但解码全错 | 训练完美拟合「错位一格」的目标，解码时必然全错 |
| 8 条样本也拟合不了 | 目标与解码不可能同时对 |
| 束搜索更差 | 声学分数高但映射到错的词 |
| P0~P19b 全部 WER | 全部在测一个 off-by-one 的模型 |

## 本脚本的判决性实验

**只改一件事**：target 构造时 +1，其余完全不变。
在同样 8 条样本、同样 400 轮、同样超参下：
- 若 WER 从 1.0 降到 0.0、8/8 完美解码 → **bug 确认，且这就是唯一根因**
- 否则还有其他问题

同时跑一个 40 类子集的小规模 CTC 训练做交叉验证。

## 判据（跑之前写死）

`WER < 0.05 且 完美解码 8/8` → off-by-one 是唯一根因。

只读 train，**不触碰 dev 与 test**。
"""
import argparse
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


def make_targets_collate(samples, plus_one: bool):
    """复刻 collate_samples，但可选择是否对 target +1。"""
    lengths = [s.features.shape[0] for s in samples]
    width = {s.features.shape[1] for s in samples}.pop()
    B = len(samples)
    features = np.zeros((B, max(lengths), width), dtype=np.float32)
    for i, s in enumerate(samples):
        features[i, :s.features.shape[0]] = s.features
    tl = [len(s.token_ids) for s in samples]
    flat = []
    for s in samples:
        ids = [i + 1 for i in s.token_ids] if plus_one else list(s.token_ids)
        flat.extend(ids)
    width_t = max(max(tl), 1)
    padded = torch.full((B, width_t), BLANK_INDEX, dtype=torch.long)
    pos = 0
    for i, L in enumerate(tl):
        padded[i, :L] = torch.tensor(flat[pos:pos + L], dtype=torch.long)
        pos += L
    return {
        "features": torch.from_numpy(features),
        "input_lengths": torch.tensor(lengths, dtype=torch.long),
        "targets": padded,
        "target_lengths": torch.tensor(tl, dtype=torch.long),
    }


def run(plus_one, pick, tr_root, voc, nrm, a, device):
    ds = GlossSequenceDataset(pick, tr_root, voc, nrm, feature_view="full")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfg = CTCConfig(input_size=368, vocabulary_size=voc.size, hidden_size=256,
                    num_layers=2, dropout=0.0, bidirectional=True,
                    projection_size=256, subsample_stride=1)
    model = CTCRecognizer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    tag = "target+1（修复）" if plus_one else "target 原样（现状 off-by-one）"
    print()
    print("=" * 74)
    print("配置：{}".format(tag))
    print("=" * 74)
    for ep in range(1, a.epochs + 1):
        model.train()
        for s in iterate_batches(ds, 8, shuffle=False, seed=0):
            b = make_targets_collate(s, plus_one)
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            lg = model(b["features"], b["input_lengths"])
            ol = model.output_lengths(b["input_lengths"])
            loss = ctc_loss(lg, b["input_lengths"], b["targets"],
                            b["target_lengths"], ol)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if ep % 100 == 0 or ep == a.epochs:
            model.eval()
            L = N = ok = 0
            with torch.no_grad():
                for s in iterate_batches(ds, 8, shuffle=False, seed=42):
                    b = make_targets_collate(s, plus_one)
                    b = {k: v.to(device) for k, v in b.items()}
                    lg = model(b["features"], b["input_lengths"])
                    ol = model.output_lengths(b["input_lengths"])
                    lp = torch.log_softmax(lg.float(), dim=-1).cpu().numpy()
                    dec, _, _ = decode_batch(lp, ol.cpu().numpy(), 1)
                    for r, ids in enumerate(dec):
                        hyp = voc.decode(list(ids))
                        R = [voc.tokens[i] for i in s[r].token_ids]
                        e = levenshtein(list(R), hyp)[0]
                        L += e; N += len(R); ok += (e == 0)
            print("  ep {:3d}  WER={:.4f}  完美 {}/{}".format(
                ep, L / max(N, 1), ok, len(pick)), flush=True)
    # 最终明细
    model.eval()
    with torch.no_grad():
        for s in iterate_batches(ds, 8, shuffle=False, seed=42):
            b = make_targets_collate(s, plus_one)
            b = {k: v.to(device) for k, v in b.items()}
            lg = model(b["features"], b["input_lengths"])
            ol = model.output_lengths(b["input_lengths"])
            lp = torch.log_softmax(lg.float(), dim=-1).cpu().numpy()
            dec, _, _ = decode_batch(lp, ol.cpu().numpy(), 1)
            for r, ids in enumerate(dec):
                hyp = voc.decode(list(ids))
                R = [voc.tokens[i] for i in s[r].token_ids]
                print("    ref={:s}".format("/".join(R)))
                print("    hyp={:s}".format("/".join(hyp) or "(空)"))
    return {"plus_one": plus_one, "tag": tag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p23-off-by-one.json")
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

    print("device = {}".format(device))
    print("voc.index_of('美')={}  该 token 索引".format(voc.index_of("美")))
    print("CTC 类空间：0=blank，词表索引 i -> 类 i+1")
    print("decoding 时 classes_to_token_ids 做 index-1")
    print("=> 训练 target 若不 +1，与解码整体错位一格")

    r_bad = run(False, pick, tr_root, voc, nrm, a, device)
    r_fix = run(True, pick, tr_root, voc, nrm, a, device)

    receipt = {
        "experiment": "P23 CTC target off-by-one verification",
        "bug": "collate_samples passes vocabulary indices as CTC class indices; "
               "class 0 is blank so vocabulary index i must be sent as class i+1",
        "criterion": "WER<0.05 and 8/8 perfect with +1",
        "all_data_real": True, "reads_dev": False, "reads_test_split": False,
        "runs": [r_bad, r_fix],
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
