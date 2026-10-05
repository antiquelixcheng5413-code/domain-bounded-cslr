# -*- coding: utf-8 -*-
"""Pure diagnostic: quantify where CTC blank / collapse comes from.

Reads only cached features + labels (test split is NOT touched). Loads a given
checkpoint and, over the *validation* split, reports:
  - blank_ratio breakdown by sample / by frame region
  - per-token predicted distribution (vocab utilization)
  - hypothesis length vs reference length
  - how many samples collapse to empty / single token / "?" etc.
No training, no writes to any artifact.
"""
import sys, json, collections
from pathlib import Path
import numpy as np
import torch

sys.setrecursionlimit(10000)

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO))

from cslr.recognition.model import CTCRecognizer, ctc_config_from_dict, BLANK_INDEX
from cslr.recognition.gloss_sequence import GlossVocabulary, GlossSequenceConfig

def load_ckpt(path):
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ctc_config_from_dict(payload["model_config"])
    model = CTCRecognizer(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    # rebuild vocabulary: checkpoint stores to_json()-style payload in 'vocabulary'
    # plus a plain config dict in 'vocabulary_config'
    vc = payload.get("vocabulary_config") or {}
    gconf = GlossSequenceConfig(**vc) if vc else GlossSequenceConfig()
    voc = GlossVocabulary(
        tokens=tuple(payload["vocabulary"]),
        counts=dict(payload.get("vocabulary_counts") or {}),
        config=gconf,
    )
    nrm = payload.get("feature_normalizer")
    return model, voc, cfg, payload, nrm

def main():
    ckpt = sys.argv[1]
    feat_root = Path(sys.argv[2])  # e.g. .../artifacts/part3_features/validation
    label_csv = Path(sys.argv[3])
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    model, voc, cfg, payload, nrm = load_ckpt(ckpt)
    print("device cpu, model input_size", cfg.input_size, "vocab", voc.size,
          "blank_index", BLANK_INDEX, flush=True)
    # standardize using train-split statistics stored in the checkpoint
    nrm_mean = None; nrm_std = None
    if nrm:
        nrm_mean = np.asarray(nrm["mean"], dtype=np.float32)
        nrm_std = np.asarray(nrm["std"], dtype=np.float32) + 1e-8
        print("normalizer applied (width=%d)" % len(nrm_mean), flush=True)

    # parse label csv
    import csv
    rows = []
    with open(label_csv, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        idcol = next((c for c in cols if "id" in c.lower() or "name" in c.lower()), cols[0])
        gloss_col = next((c for c in cols if "gloss" in c.lower() or "label" in c.lower()), None)
        for r in rd:
            rows.append((r[idcol], r[gloss_col] if gloss_col else r[cols[-1]]))

    # find clip id -> feature file whose last dim matches cfg.input_size
    feat_ids = {}
    seq = None
    for p in sorted(feat_root.glob("*.npy")):
        if ".receipt." in p.name:
            continue
        try:
            arr_shape = np.load(p).shape
        except Exception:
            continue
        if arr_shape and arr_shape[-1] == cfg.input_size:
            stem = p.stem.split(".landmark")[0].split(".motion")[0].split(".vl48")[0].split(".clip")[0]
            feat_ids[stem] = p
            seq = arr_shape
    print("features matching input_size=%d: %d  (seq %s)" % (cfg.input_size, len(feat_ids), seq))

    rng = np.random.RandomState(0)
    keys = [k for k in feat_ids if any(k == rid or k.startswith(rid[:12]) for rid, g in rows)]
    if not keys:
        keys = list(feat_ids.keys())
    rng.shuffle(keys)
    keys = keys[:limit]

    agg = collections.Counter()
    frame_blank = collections.Counter()  # region -> blank count
    frame_total = collections.Counter()
    hyp_len_hist = collections.Counter()
    ref_len_hist = collections.Counter()
    tok_emit = collections.Counter()
    examples = []
    n_empty = n_single = n_qmark = 0
    blank_step = blank_total = 0

    with torch.no_grad():
        for k in keys:
            arr = np.load(feat_ids[k])
            if nrm_mean is not None:
                arr = ((arr - nrm_mean) / nrm_std).astype(np.float32)
            feats = torch.from_numpy(arr).float().unsqueeze(0)  # [1,T,D]
            if feats.shape[2] != cfg.input_size:
                # try transpose if cached as [D,T]
                if feats.shape[1] == cfg.input_size:
                    feats = feats.transpose(1, 2)
            logits = model(feats, torch.tensor([feats.shape[1]], dtype=torch.long))
            T = logits.shape[1]
            lp = torch.log_softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
            best = lp.argmax(axis=1)
            # blank stats
            n_blk = int((best == BLANK_INDEX).sum())
            blank_step += n_blk; blank_total += T
            # region buckets
            for t in range(T):
                region = "early" if t < T*0.2 else ("late" if t >= T*0.8 else "mid")
                frame_total[region] += 1
                if best[t] == BLANK_INDEX:
                    frame_blank[region] += 1
            # decode
            collapsed = []
            prev = None
            for t in range(T):
                c = int(best[t])
                if c != prev and c != BLANK_INDEX:
                    collapsed.append(c - 1)  # vocab index
                prev = c
            for cid in collapsed:
                tok_emit[cid] += 1
            hyp = voc.decode(collapsed)
            hyp_len_hist[len(collapsed)] += 1
            # reference
            ref = next((g for rid, g in rows if k == rid or k.startswith(rid[:12])), "")
            ref_toks = voc.encode(ref) if ref else []
            ref_len_hist[len(ref_toks)] += 1
            if len(collapsed) == 0:
                n_empty += 1
            if len(collapsed) == 1:
                n_single += 1
            if hyp == "？" or hyp == "?":
                n_qmark += 1
            if len(examples) < 8:
                examples.append({"id": k, "hyp": hyp, "ref": ref, "lenT": T,
                                 "n_blank": n_blk, "n_tokens": len(collapsed)})

    total = len(keys)
    print("\n=== SUMMARY (validation, %d samples) ===" % total)
    print("blank_ratio  = %.4f  (%d/%d frames)" % (
        blank_step/max(blank_total,1), blank_step, blank_total))
    print("region blank  =", {r: "%.3f" % (frame_blank[r]/max(frame_total[r],1))
                              for r in ["early","mid","late"]})
    print("empty_hyp    = %d (%.3f)" % (n_empty, n_empty/max(total,1)))
    print("single_tok   = %d (%.3f)" % (n_single, n_single/max(total,1)))
    print("qmark_hyp    = %d" % n_qmark)
    top = tok_emit.most_common(12)
    print("top emitted vocab ids (with gloss):")
    for cid, cnt in top:
        print("   %4d  %-12s x%d" % (cid, voc.decode([cid]) if cid < voc.size else "<?>", cnt))
    total_emit = sum(tok_emit.values()) or 1
    print("vocab utilization = %d distinct / %d (%.4f)" % (
        len(tok_emit), voc.size, len(tok_emit)/voc.size))
    print("hyp_len hist:", dict(sorted(hyp_len_hist.items())))
    print("ref_len hist(top):", dict(sorted(ref_len_hist.items())[:10]))
    print("\nexamples:")
    for e in examples:
        print("  ", e)

if __name__ == "__main__":
    main()