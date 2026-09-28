"""Pretrain the contrastive alignment head (video <-> gloss) and save a warm-start checkpoint.

Produces ``artifacts/checkpoints/contrastive-vl48-*.pt`` whose ``normalize.*`` /
``projection.*`` weights can be loaded by the CTC trainer via ``--init-frontend``.
Only the train split is read; the test split stays frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cslr.recognition.contrastive_pretrain import ContrastiveConfig, pretrain_contrastive
from cslr.recognition.dataset import build_vocabulary_from_records, load_records
from cslr.recognition.gloss_sequence import GlossSequenceConfig
from cslr.recognition.vocab_cli import probe_feature_width

REPO = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "data/manifests/ce-csl.csv")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=REPO / "artifacts/checkpoints/contrastive-vl48.pt"
    )
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--input-size", type=int, default=None, help="defaults to the cached feature width")
    parser.add_argument("--projection-size", type=int, default=256)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    parser.add_argument("--present-only", action="store_true")
    parser.add_argument("--limit-train", type=int, help="smoke / ablation: cap train samples")
    parser.add_argument("--normalizer-samples", type=int, default=600)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_records(args.manifest)
    vocabulary, _ = build_vocabulary_from_records(
        records,
        min_frequency=args.min_frequency,
        max_tokens=args.max_tokens,
        config=GlossSequenceConfig(),
    )
    input_size = args.input_size if args.input_size is not None else probe_feature_width(
        args.manifest, args.features
    )
    config = ContrastiveConfig(
        input_size=input_size,
        projection_size=args.projection_size,
        embedding_size=args.embedding_size,
        temperature=args.temperature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
    )
    result = pretrain_contrastive(
        manifest_path=args.manifest,
        feature_root=args.features,
        vocabulary=vocabulary,
        config=config,
        output_path=args.output,
        present_only=args.present_only,
        normalizer_sample_limit=args.normalizer_samples,
        limit_train=args.limit_train,
    )
    payload = result.as_dict()
    payload["test_split_read"] = False
    payload["vocabulary_config"] = vocabulary.config.as_dict()
    payload["input_size"] = args.input_size
    receipt = args.output.with_suffix(".receipt.json")
    receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(receipt)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
