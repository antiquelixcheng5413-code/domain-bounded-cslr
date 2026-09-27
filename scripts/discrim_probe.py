"""Discriminative-power probe: does a cached feature family carry gloss information?

The probe answers one question: can a *linear* classifier tell which gloss tokens a clip
contains, from its mean-pooled cached features? Prior work reported "0.864 accuracy"
for landmark features, but plain accuracy is dominated by the majority class (most glosses
are absent). This probe uses per-gloss ROC-AUC and macro-F1 over high-frequency glosses,
which stay informative under class imbalance. Four families are compared: landmark, CLIP
rgb, CLIP motion, and frozen Qwen2.5-VL vision-tower (VL48).

Protocol: fit on train (4973), evaluate on dev (515). The frozen ``test`` split is never
read. Receipts are written to artifacts/metrics/part4-discrim-<feature>.json.
"""

from __future__ import annotations

import argparse
import csv
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
from cslr.recognition.io import write_json  # noqa: E402


def load_gloss_tokens(label_dir: Path, split_file: str) -> dict[str, list[str]]:
    """Read {sample_id: gloss token list} from an official CE-CSL label CSV."""

    path = label_dir / split_file
    if not path.exists():
        raise FileNotFoundError(f"label file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            sample_id = (row["Number"] or "").strip()
            gloss = (row["Gloss"] or "").strip()
            tokens = [token for token in gloss.split("/") if token]
            rows[sample_id] = tokens
    return rows


def build_labels(tokens_by_id: dict[str, list[str]], min_freq: int, max_labels: int) -> tuple[list[str], dict[str, int]]:
    """Return (label tokens, {token: position}) over frequent glosses."""

    counts: Counter[str] = Counter()
    for tokens in tokens_by_id.values():
        counts.update(set(tokens))
    labels = sorted(
        (token for token, count in counts.items() if count >= min_freq),
        key=lambda token: (-counts[token], token),
    )[:max_labels]
    return labels, {token: position for position, token in enumerate(labels)}


def encode_rows(tokens_by_id: dict[str, list[str]], labels: list[str], index: dict[str, int]) -> dict[str, np.ndarray]:
    """Encode each sample's gloss set as a one-hot row over the shared label set."""

    rows = {}
    for sample_id, tokens in tokens_by_id.items():
        vector = np.zeros(len(labels), dtype=np.float32)
        for token in set(tokens):
            position = index.get(token)
            if position is not None:
                vector[position] = 1.0
        rows[sample_id] = vector
    return rows


def load_features(
    feature_root: Path, sample_ids: list[str], split_dir: str | None, suffix: str, skip_missing: bool
) -> tuple[np.ndarray, list[str]]:
    """Mean-pool each clip's cached sequence into one feature vector per sample.

    ``split_dir=None`` selects the flat layout ({root}/{id}{suffix}.npy, used by the VL48
    cache); otherwise the layout is {root}/{split_dir}/{id}{suffix}.npy (used by
    artifacts/part3_features/{train,validation}). With ``skip_missing`` the callers whose
    feature file is absent are dropped and reported instead of raising (the legacy cache
    misses one dev sample).
    """

    vectors = []
    missing: list[str] = []
    for sample_id in sample_ids:
        parts = [feature_root]
        if split_dir is not None:
            parts.append(split_dir)
        path = Path(*parts) / f"{sample_id}{suffix}.npy"
        if not path.exists():
            if skip_missing:
                missing.append(sample_id)
                continue
            raise FileNotFoundError(f"feature file is missing: {path}")
        vectors.append(np.load(path).mean(axis=0))
    return np.stack(vectors), missing


def evaluate(clf, x_dev: np.ndarray, y_dev: np.ndarray, labels: list[str]) -> dict[str, object]:
    from sklearn.metrics import roc_auc_score

    proba = clf.predict_proba(x_dev) if hasattr(clf, "predict_proba") else None
    aucs: dict[str, float] = {}
    for position, token in enumerate(labels):
        y_true = y_dev[:, position].astype(int)
        if proba is None:
            y_score = y_true  # not reached for OneVsRestClassifier
        else:
            y_score = proba[:, position]
        if y_true.sum() == 0 or (y_true == 0).sum() == 0:
            continue
        aucs[token] = float(roc_auc_score(y_true, y_score))

    macro_auc = sum(aucs.values()) / len(aucs) if aucs else 0.0
    ranked = sorted(aucs.items(), key=lambda item: item[1])
    return {
        "macro_auc": round(macro_auc, 4),
        "n_labels_with_positive_in_eval": len(aucs),
        "top_auc": [{"gloss": token, "auc": round(value, 4)} for token, value in ranked[-5:][::-1]],
        "bottom_auc": [{"gloss": token, "auc": round(value, 4)} for token, value in ranked[:5]],
    }


def run_probe(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    train_ids = [r.sample_id for r in records if r.split == "train"]
    eval_ids = [r.sample_id for r in records if r.split == "validation"]
    if not train_ids or not eval_ids:
        raise SystemExit("manifest has no train/validation records")

    train_tokens = load_gloss_tokens(args.label_dir, "train.csv")
    eval_tokens = load_gloss_tokens(args.label_dir, "dev.csv")
    labels, index = build_labels(train_tokens, args.min_freq, args.max_labels)
    train_rows = encode_rows(train_tokens, labels, index)
    eval_rows = encode_rows(eval_tokens, labels, index)

    y_train = np.stack([train_rows[sample_id] for sample_id in train_ids if sample_id in train_rows])
    y_eval_full = np.stack([eval_rows[sample_id] for sample_id in eval_ids])
    train_ids_present = [sample_id for sample_id in train_ids if sample_id in train_rows]

    roots = [
        ("landmark", args.landmark_root, "train", "validation", ".landmark"),
        ("rgb", args.rgb_root, "train", "validation", ".rgb"),
        ("motion", args.motion_root, "train", "validation", ".motion"),
        ("vl48", args.vl48_root, None, None, ""),
    ]
    results: dict[str, object] = {}
    for name, root, train_dir, eval_dir, suffix in roots:
        if root is None:
            continue
        print(f"[{name}] loading features ...", flush=True)
        x_train, missing_train = load_features(root, train_ids_present, train_dir, suffix, skip_missing=True)
        x_eval, missing_eval = load_features(root, eval_ids, eval_dir, suffix, skip_missing=True)

        # load_features already dropped eval samples whose features are absent
        # (legacy cache misses dev-00403); align y to the same kept positions
        keep_positions = [position for position, sample_id in enumerate(eval_ids) if sample_id not in missing_eval]
        y_eval = y_eval_full[keep_positions]

        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        clf = OneVsRestClassifier(LogisticRegression(max_iter=args.max_iter))
        clf.fit(x_train, y_train)
        summary = evaluate(clf, x_eval, y_eval, labels)
        summary.update(
            {
                "feature": name,
                "feature_dim": int(x_train.shape[1]),
                "n_train": len(x_train),
                "n_eval": len(x_eval),
                "n_labels": len(labels),
                "min_freq": args.min_freq,
                "baseline_macro_auc": 0.5,
                "skipped_missing_train": len(missing_train),
                "skipped_missing_eval": len(missing_eval),
                "test_split_read": False,
            }
        )
        results[name] = summary
        print(f"[{name}] macro_auc={summary['macro_auc']} n_labels={len(labels)}", flush=True)

    receipt = {
        "probe": "discrim-power",
        "protocol": "mean-pool + linear OneVsRest on train, ROC-AUC on dev",
        "results": results,
    }
    write_json(args.out, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.discrim_probe")
    parser.add_argument("--manifest", type=Path, default=Path(REPO) / "data/manifests/ce-csl.csv")
    parser.add_argument("--label-dir", type=Path, default=Path(REPO) / "data/raw/CE-CSL/label")
    parser.add_argument("--landmark-root", type=Path)
    parser.add_argument("--rgb-root", type=Path)
    parser.add_argument("--motion-root", type=Path)
    parser.add_argument("--vl48-root", type=Path)
    parser.add_argument("--min-freq", type=int, default=5, help="keep glosses seen >= N times in train")
    parser.add_argument("--max-labels", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path(REPO) / "artifacts/metrics/part4-discrim-probe.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
