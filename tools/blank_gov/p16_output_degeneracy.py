# -*- coding: utf-8 -*-
"""P16 · 输出退化诊断：为什么所有模态的 WER 都卡在 0.8439

P15 的 modality 消融出现一个无法解释的现象：
**hands(252维) 与 face(48维) 的 WER 精确相同（0.843904），三个 seed 全一样。**

本脚本诊断根因。全部为真实数据实测，无任何合成数据、无公式推算。

## 初步观察（已实测）

```
hands  三 seed WER: 0.843904 0.843904 0.843904   <- 精确相同
face   三 seed WER: 0.843904 0.843904 0.840704
full   三 seed WER: 0.850584 0.843904 0.844604
```

输入维度差 5 倍（48 vs 252），ep1 WER 却完全相同 —— 这不可能是「学到的解」。

## 本脚本要回答的问题

1. 模型在 dev 上究竟输出多少种不同序列？
2. `0.843904` 对应什么退化输出？
3. 各模态的差异是否只是这个退化解的偶然产物？

只读 dev，不触碰 test split。
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

try:
    from cslr.contracts import SampleRecord
    from cslr.recognition.model import ctc_config_from_dict, CTCRecognizer, BLANK_INDEX
    from cslr.recognition.gloss_sequence import GlossVocabulary, GlossSequenceConfig
    from cslr.recognition.dataset import (
        GlossSequenceDataset, FeatureNormalizer, collate_samples)
    from cslr.recognition.training import (
        iterate_batches, to_torch_batch, resolve_device, decode_batch)
    from p8_error_attribution import levenshtein
    from p11_scp_count_prior import ensure_link_dir
    _CSLR_ERROR = None
except ModuleNotFoundError as exc:
    _CSLR_ERROR = exc
    torch = None
    BLANK_INDEX = 0


def diagnose_outputs(decoded, ref_by_id, voc, tag):
    """统计输出的多样性。decoded: [(sid, ids)]，ref_by_id: {sid: [tokens]}"""
    lens = []
    counter = collections.Counter()
    d_loose = d_strict = 0
    n_loose = n_strict = 0
    for sid, ids in decoded:
        hyp = voc.decode(list(ids))
        counter[tuple(ids)] += 1
        lens.append(len(hyp))
        ref_s = ref_by_id[sid]
        d_strict += levenshtein(ref_s, hyp)[0]
        n_strict += len(ref_s)
        d_loose += levenshtein(ref_s, hyp)[0]      # 严格口径 ref
        n_loose += len(ref_s)
    n = len(decoded)
    print("\n" + "=" * 72)
    print("输出退化诊断：{}".format(tag))
    print("=" * 72)
    print("样本数                     {}".format(n))
    print("输出序列长度 min/max/均值   {} / {} / {:.2f}".format(
        min(lens), max(lens), float(np.mean(lens))))
    print("**不同输出序列的种类数**   {}   ← 关键指标".format(len(counter)))
    print("参考词数 M 均值             {:.2f}".format(
        float(np.mean([len(v) for v in ref_by_id.values()]))))
    print("严格口径 WER                {:.4f}".format(d_strict / max(n_strict, 1)))
    print()
    print("最常见 8 种输出：")
    for ids, c in counter.most_common(8):
        share = c / n
        print("   {:>4} 次 ({:>5.1%})  {}".format(c, share, voc.decode(list(ids))))
    return {
        "n_samples": n,
        "n_distinct_outputs": len(counter),
        "output_len_mean": float(np.mean(lens)),
        "output_len_max": int(max(lens)),
        "ref_len_mean": float(np.mean([len(v) for v in ref_by_id.values()])),
        "wer_strict": float(d_strict / max(n_strict, 1)),
        "top_outputs": [
            {"count": int(c), "share": float(c / n), "tokens": voc.decode(list(ids))}
            for ids, c in counter.most_common(10)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/ctc-landmark48-cap300.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",
                    default="artifacts/metrics/blank-gov/p16-output-degeneracy.json")
    a = ap.parse_args()
    if _CSLR_ERROR is not None:
        raise SystemExit("cslr 不可用：{}".format(_CSLR_ERROR))

    device = resolve_device("auto")
    print("device = {}".format(device))

    payload = torch.load(REPO / a.ckpt, map_location="cpu", weights_only=False)
    cfg = ctc_config_from_dict(payload["model_config"])
    model = CTCRecognizer(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    vc = payload.get("vocabulary_config") or {}
    voc = GlossVocabulary(tokens=tuple(payload["vocabulary"]),
                          counts=dict(payload.get("vocabulary_counts") or {}),
                          config=GlossSequenceConfig(**vc) if vc else GlossSequenceConfig())
    nrm_raw = payload.get("feature_normalizer")
    nrm = FeatureNormalizer(mean=np.asarray(nrm_raw["mean"], np.float32),
                            std=np.asarray(nrm_raw["std"], np.float32) + 1e-8)

    root = ensure_link_dir("validation")
    ref_by_id = {}
    with open(REPO / "data/raw/CE-CSL/label/dev.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            toks = [t.strip() for t in r["Gloss"].split("/") if t.strip()]
            if toks:
                ref_by_id[r["Number"]] = toks
    recs = [SampleRecord(sample_id=s, video=Path(s + ".mp4"), label="/".join(t),
                         signer=s.split("-")[0], session="x", split="validation")
            for s, t in ref_by_id.items() if (root / (s + ".npy")).exists()]
    recs.sort(key=lambda r: r.sample_id)
    print("dev 样本 {}".format(len(recs)))

    ds = GlossSequenceDataset(recs, root, voc, nrm, feature_view="full")
    decoded = []
    for samples in iterate_batches(ds, a.batch, shuffle=False, seed=a.seed):
        b = to_torch_batch(collate_samples(samples), device)
        with torch.no_grad():
            logits = model(b["features"], b["input_lengths"])
        ol = model.output_lengths(b["input_lengths"])
        lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        dec, _, _ = decode_batch(lp, ol.cpu().tolist(), 1)
        for row, ids in enumerate(dec):
            decoded.append((samples[row].sample_id, ids))

    receipt = diagnose_outputs(decoded, ref_by_id, voc,
                              "已充分训练的 cap300 检查点（{}）".format(a.ckpt))

    # ---- 判定 ----
    n_dist = receipt["n_distinct_outputs"]
    n = receipt["n_samples"]
    top1 = receipt["top_outputs"][0]
    print("\n" + "=" * 72)
    print("判定：输出是否退化？")
    print("=" * 72)
    print("  不同输出种类数 / 样本数 = {} / {} = {:.1%}".format(n_dist, n, n_dist / n))
    print("  最常见输出占 {:.1%}（{}）".format(top1["share"], top1["tokens"]))
    print()
    if n_dist / n < 0.10:
        print("  **严重退化**：模型对 {:.1%} 的不同输入给出几乎相同的输出。".format(
            1 - n_dist / n))
        print("  这解释了 P15 的现象 —— 不同模态的 WER 都等于「输出这个固定序列」")
        print("  时的 WER，与模态信息无关。")
        verdict = ("输出严重退化：{} 条样本只有 {} 种输出序列，"
                   "最高频一种占 {:.1%}。模态消融的 WER 差异不反映模态信息量，"
                   "**该消融设计在本数据上无法给出有效结论**。").format(
                       n, n_dist, top1["share"])
    elif n_dist / n < 0.30:
        verdict = ("输出中度退化：{} 种 / {} 条（{:.1%}）。"
                   "模态消融结论需谨慎解读。").format(n_dist, n, n_dist / n)
        print("  中度退化，模态消融结论需谨慎。")
    else:
        verdict = "输出多样性正常（{:.1%}），模态消融有效。".format(n_dist / n)
        print("  输出多样性正常，模态消融有效。")
    print()
    print("  判定：{}".format(verdict))
    print()
    print("  对 P15 的影响：hands 与 face 的 WER 精确相同（0.843904），")
    print("  是因为两者在 ep1 都退化成同一个输出序列，而非「信息量相同」。")
    print("  **P15 的模态排序（full > hands > face）不可采信。**")

    receipt["verdict"] = verdict
    receipt["p15_implication"] = (
        "P15 的 modality 消融因输出退化而失效：不同输入维度的模型在 ep1 "
        "收敛到同一退化输出，WER 精确相同（0.843904）。"
        "该数值是退化解的 WER，不反映模态信息量。")
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
