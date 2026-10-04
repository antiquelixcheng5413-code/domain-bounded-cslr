# -*- coding: utf-8 -*-
"""P0 诊断：判定 97% blank 的成因是 peaky 病理还是词间过渡。

只读 checkpoint + 缓存特征 + dev 标签。不训练、不写任何 model artifact。
test split 不触碰。

用法:
  ./venv/bin/python tools/blank_gov/diag_peaky.py \
      --ckpt artifacts/checkpoints/ctc-landmark48-cap300.pt \
      --limit 200
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.recognition.model import CTCRecognizer, ctc_config_from_dict, BLANK_INDEX
from cslr.recognition.gloss_sequence import GlossVocabulary, GlossSequenceConfig
import diagnose as D


def load_ckpt(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ctc_config_from_dict(payload["model_config"])
    model = CTCRecognizer(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    vc = payload.get("vocabulary_config") or {}
    gconf = GlossSequenceConfig(**vc) if vc else GlossSequenceConfig()
    voc = GlossVocabulary(
        tokens=tuple(payload["vocabulary"]),
        counts=dict(payload.get("vocabulary_counts") or {}),
        config=gconf,
    )
    return model, voc, cfg, payload.get("feature_normalizer")


def read_labels(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        idcol = next(
            (c for c in cols if c.lower() in ("number", "id", "name")),
            cols[0],
        )
        gcol = next(
            (c for c in cols if "gloss" in c.lower() or "label" in c.lower()),
            cols[-1],
        )
        for r in rd:
            rows.append((r[idcol], r[gcol]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/ctc-landmark48-cap300.pt")
    ap.add_argument("--feat-root", default="artifacts/part3_features/validation")
    ap.add_argument("--labels", default="data/raw/CE-CSL/label/dev.csv")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p0-diagnosis.json")
    a = ap.parse_args()

    model, voc, cfg, nrm = load_ckpt(REPO / a.ckpt)
    print(
        "model input_size={} vocab={} blank={}".format(
            cfg.input_size, voc.size, BLANK_INDEX
        ),
        flush=True,
    )

    nrm_mean = nrm_std = None
    if nrm:
        nrm_mean = np.asarray(nrm["mean"], dtype=np.float32)
        nrm_std = np.asarray(nrm["std"], dtype=np.float32) + 1e-8
        print("normalizer applied width={}".format(len(nrm_mean)), flush=True)

    rows = read_labels(REPO / a.labels)
    ref_by_id = {rid: g for rid, g in rows}

    feat_dir = REPO / a.feat_root
    feat_map = {}
    for p in sorted(feat_dir.glob("*.npy")):
        if ".receipt." in p.name:
            continue
        try:
            shp = np.load(p).shape
        except Exception:
            continue
        if shp and shp[-1] == cfg.input_size:
            stem = p.stem.split(".landmark")[0].split(".clip")[0]
            feat_map[stem] = p
    print(
        "features matching input_size={}: {}".format(cfg.input_size, len(feat_map)),
        flush=True,
    )

    keys = [k for k in feat_map if k in ref_by_id]
    if not keys:
        keys = list(feat_map.keys())
    keys.sort()
    rng = np.random.RandomState(0)
    rng.shuffle(keys)
    keys = keys[: a.limit]
    print("evaluating {} samples".format(len(keys)), flush=True)

    logits_batch = []
    n_gloss_batch = []
    per_detail = []

    with torch.no_grad():
        for i, k in enumerate(keys):
            arr = np.load(feat_map[k])
            if nrm_mean is not None:
                arr = ((arr - nrm_mean) / nrm_std).astype(np.float32)
            feats = torch.from_numpy(arr).float().unsqueeze(0)
            if feats.shape[2] != cfg.input_size:
                if feats.shape[1] == cfg.input_size:
                    feats = feats.transpose(1, 2)
            logits = model(feats, torch.tensor([feats.shape[1]], dtype=torch.long))
            lp = torch.log_softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
            logits_batch.append(lp)

            ref = ref_by_id.get(k, "")
            ref_toks = voc.encode(ref) if ref else []
            n_gloss_batch.append(len(ref_toks))

            best = lp.argmax(axis=1)
            segs = D.peak_segments(lp, BLANK_INDEX)
            per_detail.append(
                {
                    "id": k,
                    "T": int(lp.shape[0]),
                    "n_ref_gloss": len(ref_toks),
                    "blank_ratio": float((best == BLANK_INDEX).mean()),
                    "n_peak": len(segs),
                    "peak_lens": [e - s for s, e, _ in segs],
                }
            )
            if (i + 1) % 50 == 0:
                print("  {}/{}".format(i + 1, len(keys)), flush=True)

    rep = D.peak_alignment_report(logits_batch, n_gloss_batch, BLANK_INDEX)
    rep["checkpoint"] = a.ckpt
    rep["split"] = "dev(validation)"
    rep["n_samples"] = len(keys)
    rep["vocab_size"] = int(voc.size)
    rep["input_size"] = int(cfg.input_size)
    rep["samples"] = per_detail[:50]

    print()
    print("=" * 62)
    print("P0 blank 诊断  ({} 条 dev)".format(len(keys)))
    print("=" * 62)
    print("blank_ratio argmax     {:.4f}".format(rep["blank_ratio_argmax_mean"]))
    print("blank_ratio thresh     {:.4f}".format(rep["blank_ratio_thresh_mean"]))
    print("peak_frames_per_token  {:.2f}   <- 关键判据".format(rep["peak_frames_per_token"]))
    print("peak_count_ratio       {:.3f}".format(rep["peak_count_ratio_global"]))
    print(
        "peak_len median/p90    {:.0f} / {:.0f}".format(
            rep["peak_len_median"], rep["peak_len_p90"]
        )
    )
    print(
        "blank_run median/p90   {:.0f} / {:.0f}".format(
            rep["blank_run_median"], rep["blank_run_p90"]
        )
    )
    print()
    dg = rep["diagnosis"]
    print(
        "判定: {}  (peaky={}, severe={})".format(
            dg["verdict"], dg["peaky"], dg["severe"]
        )
    )
    print("建议路线:")
    for r in dg["route"]:
        print("  - {}".format(r))

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("")
    print("收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
