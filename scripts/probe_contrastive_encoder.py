"""Discriminative-power probe *of the contrastive-pretrained encoder*.

The warm-started CTC (section 22) reproduced the collapse (WER 0.854 vs 0.853,
blank 0.964 vs 0.965), which is ambiguous: the pretraining may either have failed to
make clip representations more discriminative, or its pooled-alignment signal may
just not transfer to frame-wise CTC. This probe resolves that ambiguity by running
the SAME per-gloss ROC-AUC machinery as scripts/discrim_probe.py on the encoder's
own pooled representations. If AUC clearly exceeds the raw VL48 0.646 (and approaches
the landmark 0.682), the pretraining built discriminative clip embeddings and the
bottleneck is transfer; if AUC is flat, the pretraining promise itself is weak.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = "/home/su127/FYP/domain-bounded-cslr"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
    sys.path.insert(0, str(Path(REPO) / "src"))

from cslr.recognition.contrastive_pretrain import ContrastiveEncoder
from cslr.recognition.dataset import (
    FeatureNormalizer,
    GlossSequenceDataset,
    collate_samples,
    load_records,
)
from cslr.recognition.gloss_sequence import GlossVocabulary
from cslr.recognition.io import write_json
from cslr.recognition.training import iterate_batches, resolve_device, to_torch_batch

from scripts.discrim_probe import build_labels, encode_rows, evaluate, load_gloss_tokens


def pooled_embeddings(
    checkpoint: Path, feature_root: Path, records, device: torch.device
) -> tuple[np.ndarray, list[str]]:
    """Feed each clip through the pretrained encoder, return L2-normalised pooled vectors."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["pretrain_config"]
    encoder = ContrastiveEncoder(
        int(config["input_size"]), int(config["projection_size"])
    ).to(device)
    encoder.load_state_dict(payload["state_dict"])
    encoder.eval()
    normalizer_payload = payload.get("feature_normalizer")
    normalizer = (
        FeatureNormalizer.from_dict(normalizer_payload) if normalizer_payload else None
    )
    vocabulary = GlossVocabulary(
        tokens=tuple(str(token) for token in payload.get("vocabulary", ["<unk>"]))
    )
    dataset = GlossSequenceDataset(
        records, feature_root, vocabulary, normalizer, feature_view="full"
    )
    vectors: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for samples in iterate_batches(dataset, 32, shuffle=False, seed=0):
            batch = to_torch_batch(collate_samples(samples), device)
            anchors = encoder(batch["features"], batch["input_lengths"])
            vectors.extend(anchors.float().cpu().numpy())
            sample_ids.extend(batch["sample_ids"])
    return np.stack(vectors), sample_ids


def main(args: argparse.Namespace) -> int:
    records = load_records(args.manifest)
    train_ids = [r.sample_id for r in records if r.split == "train"]
    eval_ids = [r.sample_id for r in records if r.split == "validation"]

    train_tokens = load_gloss_tokens(args.label_dir, "train.csv")
    eval_tokens = load_gloss_tokens(args.label_dir, "dev.csv")
    labels, index = build_labels(train_tokens, args.min_freq, args.max_labels)
    train_rows = encode_rows(train_tokens, labels, index)
    eval_rows = encode_rows(eval_tokens, labels, index)

    y_train = np.stack([train_rows[rid] for rid in train_ids if rid in train_rows])
    y_eval_full = np.stack([eval_rows[rid] for rid in eval_ids])

    device = resolve_device(args.device)
    train_records = [
        r for r in records if r.split == "train" and r.sample_id in train_rows
    ]
    eval_records = [r for r in records if r.split == "validation"]

    x_train, train_sample_ids = pooled_embeddings(
        args.checkpoint, args.features, train_records, device
    )
    x_eval, eval_sample_ids = pooled_embeddings(
        args.checkpoint, args.features, eval_records, device
    )

    # align y to the (possibly smaller) set of eval ids the encoder produced
    index_of_eval = {sid: pos for pos, sid in enumerate(eval_ids)}
    keep_positions = [index_of_eval[sid] for sid in eval_sample_ids]
    y_eval = y_eval_full[keep_positions]
    assert len(x_eval) == len(y_eval), (x_eval.shape, y_eval.shape)

    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    clf = OneVsRestClassifier(LogisticRegression(max_iter=args.max_iter))
    clf.fit(x_train, y_train)
    summary = evaluate(clf, x_eval, y_eval, labels)
    summary.update(
        {
            "feature": "contrastive-encoder",
            "checkpoint": str(args.checkpoint),
            "feature_dim": int(x_train.shape[1]),
            "n_train": len(x_train),
            "n_eval": len(x_eval),
            "n_labels": len(labels),
            "baseline_macro_auc": 0.5,
            "test_split_read": False,
        }
    )
    receipt = {
        "probe": "discrim-power-of-contrastive-encoder",
        "results": {"contrastive_encoder": summary},
    }
    write_json(args.out, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(REPO) / "artifacts/checkpoints/contrastive-vl48.pt",
    )
    parser.add_argument(
        "--features", type=Path, default=Path(REPO) / "data/processed/ce-csl-qwenvl48"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path(REPO) / "data/manifests/ce-csl.csv"
    )
    parser.add_argument(
        "--label-dir", type=Path, default=Path(REPO) / "data/raw/CE-CSL/label"
    )
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--max-labels", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(REPO) / "artifacts/metrics/part4-probe-contrastive-encoder.json",
    )
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(main(args))