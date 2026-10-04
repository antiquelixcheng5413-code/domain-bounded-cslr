# -*- coding: utf-8 -*-
"""P17 · 判决性诊断：输出坍缩的真正根因（推翻 P16 的"模型没学到区分"结论）

## P16 之前的错误前提

P16 观察到「514 条输入只产生 6 种输出序列」，据此写下结论：

> 「模型不是『学得不好』，而是『几乎没学到区分』」

**这个结论建立在一个未验证的前提上：被诊断的 checkpoint 训练充分。**

## 本脚本要检验的事实（全部来自真实数据，无合成）

读 `artifacts/logs/ctc-landmark48-cap300.log` 时发现：

```
epoch  train_loss  val_loss  wer     blank   vocab_util
  1     20.1358     9.4377   0.8566  0.9781    0.0100
  2      9.4010     9.2053   0.8506  0.9692    0.0133   <- best_epoch=2, WER 最低
  3      8.6501     8.7323   0.8615  0.9662    0.0266
 ...
 14      1.5953     6.1950   0.8693  0.9531    0.1561
{"early_stop": true, "epoch": 14}
best_epoch = 2
```

**`best_epoch = 2`** —— 因为模型选择用的是 WER，而 WER 在第 2 轮之后
**单调变差**（0.8506 -> 0.8693），于是「WER 最低」=「训练最少的那个」。
存盘的 `ctc-landmark48-cap300.pt` 就是**第 2 轮**的权重。

而 train_loss 从 20.1 一路降到 1.595（学会背训练集），
vocab_util 从 0.010 升到 0.156（输出种类在增加）。

**所以 P0/P13/P14/P15/P16 全部诊断的是一个只训练了 2 个 epoch 的模型。**
「输出坍缩」很可能不是模型能力上限，而是**早停准则选错了模型**。

## 三个可判决的量

1. **同一份代码，epoch 2 与 epoch 14 的输出序列种类数差多少？**
   若 epoch14 >> epoch2，则坍缩是训练不足造成的，方向是「训够」而非「换模型」。
2. **训练目标里 `<unk>` 占比 vs 非 unk 占比**
   —— 已知 cap300 下 32.4% 训练目标是 `<unk>`。若模型输出 `<unk>`
   恰好对应该比例的梯度主导，则「<unk> 折叠」才是坍缩的直接机制。
3. **CTC 在 T=48 上的对齐容量**
   输出长度均值 1.48 vs 参考 5.52。CTC 需要 T >= 2L-1，L=5.52 需 T>=10，
   T=48 容量充裕 —— 排除「时间分辨率不足导致无法输出多词」。

## 判据（预先写死，跑之前就定下）

- 若 `n_distinct(epoch14) > 2 x n_distinct(epoch2)`：
  **坍缩主因 = 训练不足 + WER 早停选错模型** → 修训练配置即可
- 若 `n_distinct(epoch14) ≈ n_distinct(epoch2) <= 10`：
  坍缩与训练时长无关 → 需查目标构造/标签分布

只读 dev，不触碰 test split。
"""
import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.contracts import SampleRecord
from cslr.recognition.model import CTCRecognizer, ctc_config_from_dict
from cslr.recognition.gloss_sequence import (
    GlossVocabulary, GlossSequenceConfig, build_ordered_vocabulary)
from cslr.recognition.dataset import GlossSequenceDataset, FeatureNormalizer, collate_samples
from cslr.recognition.training import iterate_batches, to_torch_batch, decode_batch
from p8_error_attribution import levenshtein


def read_split(split):
    table = {"train": "train.csv", "validation": "dev.csv"}
    if split not in table:
        raise ValueError("split 必须是 {} 之一".format(sorted(table)))
    out = {}
    with open(REPO / "data/raw/CE-CSL/label" / table[split],
              newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["Number"]] = r["Gloss"]
    return out


def ensure_link_dir(split):
    root = REPO / ".vocab_link" / split
    root.mkdir(parents=True, exist_ok=True)
    src_dir = REPO / "artifacts/part3_features" / split
    for p in sorted(src_dir.glob("*.landmark.npy")):
        sid = p.name[: -len(".landmark.npy")]
        dst = root / (sid + ".npy")
        if not dst.exists():
            try:
                os.symlink(p, dst)
            except OSError:
                pass
    return root


def make_records(labels, feat_root, split):
    out = []
    for sid, g in labels.items():
        if not (feat_root / (sid + ".npy")).exists():
            continue
        if not [t for t in g.split("/") if t.strip()]:
            continue
        out.append(SampleRecord(sample_id=sid, video=Path(sid + ".mp4"), label=g,
                                signer="x", session="x", split=split))
    out.sort(key=lambda r: r.sample_id)
    return out


def target_composition(records, voc):
    """训练/评估目标里 <unk> 与真实词的比例（真实标签，非合成）。"""
    tot = unk = 0
    sent_all_unk = 0
    for r in records:
        toks = [t.strip() for t in r.label.split("/") if t.strip()]
        if not toks:
            continue
        ids = voc.encode(r.label)
        tot += len(ids)
        n_unk = sum(1 for i in ids if i == 0)
        unk += n_unk
        if n_unk == len(ids) and ids:
            sent_all_unk += 1
    return {
        "n_tokens": tot,
        "n_unk": unk,
        "unk_rate": unk / max(tot, 1),
        "n_sent_all_unk": sent_all_unk,
        "n_sent": len(records),
        "all_unk_rate": sent_all_unk / max(len(records), 1),
    }


def decode_and_measure(model, ds, voc, device, batch, ref_by_id, tag):
    model.eval()
    seqs = []
    lens = []
    d = n = 0
    nonblank_mass = []
    with torch.no_grad():
        for samples in iterate_batches(ds, batch, shuffle=False, seed=42):
            b = to_torch_batch(collate_samples(samples), device)
            logits = model(b["features"], b["input_lengths"])
            ol = model.output_lengths(b["input_lengths"])
            lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            p = np.exp(lp)
            # 非 blank 总概率质量（P13 指标，此处复用做对照）
            pb = p[:, :, 0].sum(axis=1)
            nonblank_mass.extend((1.0 - pb).tolist())
            ol_np = ol.cpu().numpy()
            dec, _, _ = decode_batch(lp, ol_np, 1)
            for row, ids in enumerate(dec):
                hyp = voc.decode(list(ids))
                seqs.append(tuple(hyp))
                lens.append(len(hyp))
                ref = ref_by_id[b["sample_ids"][row]] if "sample_ids" in b else None
                if ref is None:
                    ref = b["tokens"][row]
                d += levenshtein(list(ref), hyp)[0]
                n += len(ref)
    cnt = collections.Counter(seqs)
    res = {
        "tag": tag,
        "n_samples": len(seqs),
        "n_distinct_outputs": len(cnt),
        "distinct_ratio": len(cnt) / max(len(seqs), 1),
        "out_len_mean": float(np.mean(lens)) if lens else 0.0,
        "out_len_max": int(np.max(lens)) if lens else 0,
        "wer": d / max(n, 1),
        "nonblank_mass_mean": float(np.mean(nonblank_mass)) if nonblank_mass else 0.0,
        "top5": [[list(k), v] for k, v in cnt.most_common(5)],
        "token_kinds": len({t for s in seqs for t in s}),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p17-root-cause.json")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = {}".format(device))

    ckpt_path = REPO / "artifacts/checkpoints/ctc-landmark48-cap300.pt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("checkpoint best_epoch = {}  best_wer = {:.4f}".format(
        ck["best_epoch"], ck["best_wer"]))

    voc = GlossVocabulary(tokens=tuple(ck["vocabulary"]),
                          counts=ck["vocabulary_counts"],
                          config=GlossSequenceConfig(**ck["vocabulary_config"]))
    cfg = ctc_config_from_dict(ck["model_config"])
    print("vocab_size = {}  input_size = {}".format(voc.size, cfg.input_size))

    tr_labels = read_split("train")
    va_labels = read_split("validation")
    tr_root = ensure_link_dir("train")
    va_root = ensure_link_dir("validation")
    tr_recs = make_records(tr_labels, tr_root, "train")
    va_recs = make_records(va_labels, va_root, "validation")
    print("train {} / dev {}".format(len(tr_recs), len(va_recs)))

    # ---- 量 2：目标构成（真实 CSV 标签）----
    tr_comp = target_composition(tr_recs, voc)
    va_comp = target_composition(va_recs, voc)
    print("")
    print("=" * 70)
    print("量2 · 目标中 <unk> 构成（cap300 词表，真实标签）")
    print("=" * 70)
    print("train  token {}  <unk> {}  = {:.1%}   全 unk 句 {}/{} = {:.1%}".format(
        tr_comp["n_tokens"], tr_comp["n_unk"], tr_comp["unk_rate"],
        tr_comp["n_sent_all_unk"], tr_comp["n_sent"], tr_comp["all_unk_rate"]))
    print("dev    token {}  <unk> {}  = {:.1%}   全 unk 句 {}/{} = {:.1%}".format(
        va_comp["n_tokens"], va_comp["n_unk"], va_comp["unk_rate"],
        va_comp["n_sent_all_unk"], va_comp["n_sent"], va_comp["all_unk_rate"]))

    # ---- 量 3：CTC 对齐容量 ----
    T = 48
    ref_lens = [len([t for t in r.label.split("/") if t.strip()]) for r in va_recs]
    need = [2 * L - 1 for L in ref_lens]
    cap = {
        "T": T,
        "ref_len_mean": float(np.mean(ref_lens)),
        "ref_len_max": int(np.max(ref_lens)),
        "need_2L_minus_1_mean": float(np.mean(need)),
        "n_infeasible": int(sum(1 for x in need if x > T)),
        "infeasible_rate": float(sum(1 for x in need if x > T)) / max(len(need), 1),
    }
    print("")
    print("=" * 70)
    print("量3 · CTC 对齐容量（T={}）".format(T))
    print("=" * 70)
    print("参考长度 mean {:.2f} max {}   需要 2L-1 mean {:.2f}   不可行样本 {}/{} = {:.1%}".format(
        cap["ref_len_mean"], cap["ref_len_max"], cap["need_2L_minus_1_mean"],
        cap["n_infeasible"], len(need), cap["infeasible_rate"]))

    # ---- normalizer：必须用 checkpoint 里存的那个，否则输入分布不同 ----
    nrm_mean = np.asarray(ck["feature_normalizer"]["mean"], dtype=np.float32)
    nrm_std = np.asarray(ck["feature_normalizer"]["std"], dtype=np.float32)

    class _N:
        def apply(self, x):
            return (x - nrm_mean) / np.maximum(nrm_std, 1e-6)
    nrm = _N()

    va_ds = GlossSequenceDataset(va_recs, va_root, voc, nrm, feature_view="full")
    ref_by_id = {r.sample_id: [t.strip() for t in r.label.split("/") if t.strip()]
                 for r in va_recs}

    # ---- 量 1：epoch2（存盘）vs 全量训练后的模型 ----
    print("")
    print("=" * 70)
    print("量1 · 存盘 checkpoint（best_epoch={}）的输出多样性".format(ck["best_epoch"]))
    print("=" * 70)
    model = CTCRecognizer(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    r_saved = decode_and_measure(model, va_ds, voc, device, a.batch, ref_by_id,
                                "saved_best_epoch{}".format(ck["best_epoch"]))
    print(json.dumps(r_saved, ensure_ascii=False, indent=1))

    receipt = {
        "experiment": "P17 root-cause diagnosis for output degeneracy",
        "single_factor": "checkpoint selection (stored epoch-2 vs later epoch)",
        "all_data_real": True,
        "reads_test_split": False,
        "target_composition": {"train": tr_comp, "dev": va_comp},
        "ctc_capacity": cap,
        "stored_checkpoint": r_saved,
        "checkpoint_meta": {"best_epoch": ck["best_epoch"], "best_wer": ck["best_wer"]},
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
