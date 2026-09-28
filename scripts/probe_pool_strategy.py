"""Discriminative-power probe across pooling/temporal strategies (single family).

Previous probes (sec 20/22/23) scored the *mean-pooled* cached vector per clip.
This probe asks whether that ceiling (~0.69 macro-AUC) is an encoding-layer limit
or an artifact of mean-pooling wiping out temporal discriminative signal. It
measure the same per-gloss ROC-AUC OneVsRest protocol but on landmark frame
features (48x368, fixed length) under several aggregations: mean, max, mean+std,
and sparse equispaced frame-concat that preserves a low-rate temporal signal.

If a temporal-preserving strategy clears the mean-pooled ceiling, the free lever
is "temporal aggregation" (revises sec 22/23 toward pooling, no new encoder). If
all strategies sit at ~0.69, the ceiling lives in the encoded features themselves
(old/revisionary conclusion stands, spend on a stronger encoder is required).

test split stays frozen: only train/dev are read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = "/home/su127/FYP/domain-bounded-cslr"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
    sys.path.insert(0, str(Path(REPO) / "src"))

from cslr.data.manifest import read_manifest  # noqa: E402
from scripts.discrim_probe import (  # noqa: E402
    build_labels,
    encode_rows,
    evaluate,
    load_gloss_tokens,
)


def _frame_layout(root: Path, split_dir: str, sample_ids: list[str], suffix: str) -> tuple[list[str], list[str]]:
    """Return (sample_ids whose file exists, missing)."""

    kept: list[str] = []
    missing: list[str] = []
    for sample_id in sample_ids:
        path = root / split_dir / f"{sample_id}{suffix}.npy"
        if path.exists():
            kept.append(sample_id)
        else:
            missing.append(sample_id)
    return kept, missing


def load_frames(root: Path, split_dir: str, sample_ids: list[str], suffix: str) -> list[np.ndarray]:
    return [np.load(root / split_dir / f"{sid}{suffix}.npy") for sid in sample_ids]


def aggregate(frames: list[np.ndarray], strategy: str) -> np.ndarray:
    """Stack one vector per sample under the given aggregation. Frame dim assumed last-dim = feature."""

    out = []
    for mat in frames:  # (T, D)
        if strategy == "mean":
            out.append(mat.mean(axis=0))
        elif strategy == "max":
            out.append(mat.max(axis=0))
        elif strategy == "meanstd":
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)
            out.append(np.concatenate([mean, std]))
        elif strategy.startswith("sample"):
            k = int(strategy.replace("sample", ""))
            idx = np.linspace(0, mat.shape[0] - 1, k).round().astype(int)
            selected = mat[idx]
            out.append(selected.reshape(-1))
        else:
            raise ValueError(f"unknown strategy: {strategy}")
    return np.stack(out)


def run(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    train_all = [r.sample_id for r in records if r.split == "train"]
    eval_all = [r.sample_id for r in records if r.split == "validation"]
    if not train_all or not eval_all:
        raise SystemExit("manifest has no train/validation records")

    train_tokens = load_gloss_tokens(args.label_dir, "train.csv")
    eval_tokens = load_gloss_tokens(args.label_dir, "dev.csv")
    labels, index = build_labels(train_tokens, args.min_freq, args.max_labels)
    train_rows = encode_rows(train_tokens, labels, index)
    eval_rows = encode_rows(eval_tokens, labels, index)

    train_present = [sid for sid in train_all if sid in train_rows]
    train_kept, _miss_train = _frame_layout(args.root, "train", train_present, args.suffix)
    eval_kept, miss_eval = _frame_layout(args.root, "validation", eval_all, args.suffix)

    y_train = np.stack([train_rows[sid] for sid in train_kept])
    y_eval_full = np.stack([eval_rows[sid] for sid in eval_all])
    eval_pos = [i for i, sid in enumerate(eval_all) if sid in set(eval_kept)]
    y_eval = y_eval_full[eval_pos]

    frames_train = load_frames(args.root, "train", train_kept, args.suffix)
    frames_eval = load_frames(args.root, "validation", eval_kept, args.suffix)

    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    results = {}
    for strategy in args.strategies:
        x_train = aggregate(frames_train, strategy)
        x_eval = aggregate(frames_eval, strategy)
        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=args.max_iter, solver="liblinear")
        )
        clf.fit(x_train, y_train)
        summary = evaluate(clf, x_eval, y_eval, labels)
        summary.update(
            {
                "feature_family": args.name,
                "strategy": strategy,
                "feature_dim": int(x_train.shape[1]),
                "n_train": len(x_train),
                "n_eval": len(x_eval),
                "n_labels": len(labels),
                "min_freq": args.min_freq,
                "baseline_macro_auc": 0.5,
                "test_split_read": False,
            }
        )
        results[strategy] = summary
        print(f"[{strategy}] macro_auc={summary['macro_auc']} dim={int(x_train.shape[1])}", flush=True)

    receipt = {
        "probe": "pool-strategy",
        "protocol": "per-gloss ROC-AUC OneVsRest, train->dev, pooling/temporal aggregation only",
        "results": results,
    }
    from cslr.recognition.io import write_json

    write_json(args.out, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.probe_pool_strategy")
    parser.add_argument("--manifest", type=Path, default=Path(REPO) / "data/manifests/ce-csl.csv")
    parser.add_argument("--label-dir", type=Path, default=Path(REPO) / "data/raw/CE-CSL/label")
    parser.add_argument("--root", type=Path, default=Path(REPO) / "artifacts/part3_features")
    parser.add_argument("--name", default="landmark")
    parser.add_argument("--suffix", default=".landmark")
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--max-labels", type=int, default=60)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["mean", "max", "meanstd", "sample4", "sample8"],
    )
    parser.add_argument("--out", type=Path, default=Path(REPO) / "artifacts/metrics/part4-probe-pool.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())