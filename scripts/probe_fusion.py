"""Fusion discriminative-power probe: does concatenating all cached families beat the best single family?

Sections 18/22 concluded the real bottleneck is the *input feature material*: raw VL48 pooled
representations cap at macro-AUC 0.646 (landmark 0.682, rgb 0.626, motion 0.535), and
contrastive pretraining does not add information beyond that ceiling. The SpaMo hypothesis is
exactly that the families carry *complementary* gloss information. This probe tests that
directly: concatenate lens the same mean-pooled cached features across landmark + rgb + motion
+ VL48 (3440 dims) and ask whether linear macro-AUC rises above the best single family. If it
does, fusion is a genuine lever; if it stays at the max of the parts, the families do not add
complementary signal and fusion has nothing to exploit.

Same protocol as scripts/discrim_probe.py: fit OneVsRest logreg on train (4973), ROC-AUC on
dev (515), frozen test never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = "/home/su127/FYP/domain-bounded-cslr"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
    sys.path.insert(0, str(Path(REPO) / "src"))

from scripts.discrim_probe import (
    build_labels,
    encode_rows,
    evaluate,
    load_features,
    load_gloss_tokens,
)
from cslr.data.manifest import read_manifest
from cslr.recognition.io import write_json

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/home/su127/FYP/domain-bounded-cslr/data/manifests/ce-csl.csv"))
    parser.add_argument("--label-dir", type=Path, default=Path("/home/su127/FYP/domain-bounded-cslr/data/raw/CE-CSL/label"))
    parser.add_argument("--landmark-root", type=Path, default=Path("/home/su127/FYP/domain-bounded-cslr/artifacts/part3_features"))
    parser.add_argument("--rgb-root", type=Path, default=Path("/home/su127/FYP/domain-bounded-cslr/artifacts/part3_features"))
    parser.add_argument("--motion-root", type=Path, default=Path("/home/su127/FYP/domain-bounded-cslr/artifacts/part3_features"))
    parser.add_argument("--vl48-root", type=Path, default=Path("/home/su127/FYP/domain-bounded-cslr/data/processed/ce-csl-qwenvl48"))
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--max-labels", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", default=True)
    parser.add_argument("--out", type=Path, default=Path("/home/su127/FYP/domain-bounded-cslr/artifacts/metrics/part4-probe-fusion.json"))
    return parser


def run_fusion(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    train_ids = [r.sample_id for r in records if r.split == "train"]
    eval_ids = [r.sample_id for r in records if r.split == "validation"]

    train_tokens = load_gloss_tokens(args.label_dir, "train.csv")
    eval_tokens = load_gloss_tokens(args.label_dir, "dev.csv")
    labels, index = build_labels(train_tokens, args.min_freq, args.max_labels)
    train_rows = encode_rows(train_tokens, labels, index)
    eval_rows = encode_rows(eval_tokens, labels, index)
    train_ids_present = [sid for sid in train_ids if sid in train_rows]
    y_train = np.stack([train_rows[sid] for sid in train_ids_present])
    y_eval_full = np.stack([eval_rows[sid] for sid in eval_ids])

    families = [
        ("landmark", args.landmark_root, "train", "validation", ".landmark"),
        ("rgb", args.rgb_root, "train", "validation", ".rgb"),
        ("motion", args.motion_root, "train", "validation", ".motion"),
        ("vl48", args.vl48_root, None, None, ""),
    ]

    # per-family matrices; train is complete (4973), eval drops any absent sample id
    train_mats: dict[str, np.ndarray] = {}
    eval_mats: dict[str, np.ndarray] = {}
    eval_present: dict[str, list[str]] = {}
    for name, root, td, ed, suffix in families:
        x_train, _ = load_features(root, train_ids_present, td, suffix, skip_missing=True)
        x_eval, missing = load_features(root, eval_ids, ed, suffix, skip_missing=True)
        train_mats[name] = x_train
        eval_mats[name] = x_eval
        present = [sid for sid in eval_ids if sid not in set(missing)]
        eval_present[name] = present
        print(f"[{name}] train {x_train.shape} eval {x_eval.shape} present {len(present)}", flush=True)

    # use the intersection of dev samples present in every family for a fair comparison
    present_all = set(eval_present["landmark"]) & set(eval_present["rgb"]) & set(eval_present["motion"]) & set(eval_present["vl48"])
    kept = [position for position, sid in enumerate(eval_ids) if sid in present_all]
    eval_kept_ids = [eval_ids[position] for position in kept]
    y_eval = y_eval_full[kept]

    def align_to(mat: np.ndarray, present: list[str], keep: list[str]) -> np.ndarray:
        if list(present) == keep:
            return mat
        lookup = {sid: i for i, sid in enumerate(present)}
        return mat[[lookup[sid] for sid in keep]]

    eval_aligned = {name: align_to(eval_mats[name], eval_present[name], eval_kept_ids) for name, *_ in families}
    y_eval = np.stack([eval_rows[sid] for sid in eval_kept_ids])

    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler() if args.normalize else None
    all_name = "+".join(n for n, *_ in families)
    results: dict = {}
    rows = [(n, train_mats[n], eval_aligned[n]) for n in eval_aligned] + [
        (all_name, np.concatenate([train_mats[n] for n, *_ in families], axis=1),
         np.concatenate([eval_aligned[n] for n, *_ in families], axis=1))
    ]
    for name, mat_train, mat_eval in rows:
        x_train, x_eval = mat_train, mat_eval
        if scaler is not None:
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_eval = scaler.transform(x_eval)
        clf = OneVsRestClassifier(LogisticRegression(max_iter=args.max_iter))
        clf.fit(x_train, y_train)
        summary = evaluate(clf, x_eval, y_eval, labels)
        summary.update({
            "feature": name,
            "feature_dim": int(x_train.shape[1]),
            "n_train": len(x_train),
            "n_eval": len(x_eval),
            "n_labels": len(labels),
            "baseline_macro_auc": 0.5,
            "normalized": args.normalize,
            "eval_kept": len(eval_kept_ids),
            "test_split_read": False,
        })
        results[name] = summary
        print(f"[{name}] macro_auc={summary['macro_auc']}", flush=True)

    receipt = {
        "probe": "discrim-power-fusion",
        "protocol": "mean-pool + linear OneVsRest on train, ROC-AUC on dev (intersection 514)",
        "fusion": {
            "concat_after_pool": True,
            "families": list(dict.fromkeys(n for n, *_ in families)),
            "per_family_pooled_dims": {n: int(train_mats[n].shape[1]) for n, *_ in families},
            "total_dims": int(sum(train_mats[n].shape[1] for n, *_ in families)),
        },
        "single_families": {k: results[k] for k, *_ in families},
        "fusion_row": results.get(all_name),
    }
    write_json(args.out, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))

    best_single = max((results[k]["macro_auc"] for k, *_ in families), default=0.0)
    fusion_auc = results[all_name]["macro_auc"]
    print(f"\nVERDICT: best-single={best_single:.4f}  fusion={fusion_auc:.4f}  "
          f"gain={fusion_auc - best_single:+.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_fusion(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())