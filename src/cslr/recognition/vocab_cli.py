"""CLI for the ordered-gloss CTC recognition package.

Commands::

    python -m cslr.recognition build-vocab --manifest data/manifests/ce-csl.csv
    python -m cslr.recognition train --manifest data/manifests/ce-csl.csv \
        --features data/processed/ce-csl-96 --output artifacts/checkpoints/ctc-landmark96.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cslr.data.manifest import read_manifest, validate_manifest
from cslr.recognition.dataset import (
    build_vocabulary_from_records,
    describe_split,
    feature_view_indices,
    filter_present,
    split_records,
)
from cslr.recognition.gloss_sequence import (
    GlossSequenceConfig,
    build_ordered_vocabulary,
    coverage_report,
    split_gloss_sequence,
)
from cslr.recognition.io import write_json
from cslr.recognition.model import CTCConfig
from cslr.recognition.training import TrainingConfig, train_ctc


def probe_feature_width(manifest: Path, feature_root: Path) -> int:
    """Width of the first cached feature file that exists (train split first)."""

    from cslr.recognition.dataset import load_feature

    records = read_manifest(manifest)
    for split in ("train", "validation"):
        for record in records:
            if record.split != split:
                continue
            path = feature_root / f"{record.sample_id}.npy"
            if path.exists():
                return int(load_feature(path).shape[1])
    raise ValueError(f"no cached feature file found under {feature_root}")


def _config_from_args(args: argparse.Namespace) -> GlossSequenceConfig:    return GlossSequenceConfig(
        keep_punctuation=not args.drop_punctuation,
        keep_numeric_tokens=not args.drop_numeric_tokens,
        strip_variant_numbering=not args.keep_variant_numbering,
        strip_annotations=not args.keep_annotations,
        target_unit=args.target_unit,
    )


def build_vocab_command(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    validate_manifest(records)

    def glosses_for(split: str) -> list[str]:
        return [record.label for record in records if record.split == split]

    config = _config_from_args(args)
    train_glosses = glosses_for("train")
    if not train_glosses:
        raise ValueError("manifest contains no train records")

    vocabulary, counts = build_ordered_vocabulary(
        train_glosses,
        min_frequency=args.min_frequency,
        max_tokens=args.max_tokens,
        config=config,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "token", "frequency"])
        writer.writeheader()
        for row in vocabulary.as_rows():
            writer.writerow(row)

    report: dict[str, object] = {
        "manifest": str(args.manifest),
        "min_frequency": args.min_frequency,
        "max_tokens": args.max_tokens,
        "config": config.as_dict(),
        "vocabulary_size": vocabulary.size,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        split_glosses = glosses_for(split)
        if not split_glosses:
            report["splits"][split] = {"samples": 0, "read": False}
            continue
        coverage = coverage_report(split_glosses, vocabulary, counts)
        coverage["read"] = True
        report["splits"][split] = coverage

    variants: dict[str, object] = {}
    if args.compare_variants:
        for name, candidate in (
            ("strip_variant", GlossSequenceConfig(
                keep_punctuation=config.keep_punctuation,
                keep_numeric_tokens=config.keep_numeric_tokens,
                strip_variant_numbering=True,
                strip_annotations=config.strip_annotations,
            )),
            ("keep_variant", GlossSequenceConfig(
                keep_punctuation=config.keep_punctuation,
                keep_numeric_tokens=config.keep_numeric_tokens,
                strip_variant_numbering=False,
                strip_annotations=config.strip_annotations,
            )),
        ):
            candidate_vocabulary, _ = build_ordered_vocabulary(
                train_glosses, min_frequency=args.min_frequency, config=candidate
            )
            coverage = coverage_report(train_glosses, candidate_vocabulary)
            variants[name] = {
                "config": candidate.as_dict(),
                "vocabulary_size": candidate_vocabulary.size,
                "oov_rate": coverage["oov_rate"],
                "token_occurrences": coverage["token_occurrences"],
            }
        report["variant_comparison"] = variants

    report["raw_token_stats"] = _raw_stats(config, train_glosses)
    report["feature_integrity"] = {
        split: describe_split(split_records(records, split), args.features_root)
        if args.features_root is not None
        else {"checked": False}
        for split in ("train", "validation")
    }

    write_json(args.report, report)
    print(
        json.dumps(
            {
                "status": "ok",
                "vocabulary_size": vocabulary.size,
                "min_frequency": args.min_frequency,
                "config": config.as_dict(),
                "output": str(args.output),
                "report": str(args.report) if args.report else None,
                "test_split_read": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _raw_stats(config: GlossSequenceConfig, glosses: list[str]) -> dict[str, object]:
    """Before/after token statistics that do not require a vocabulary."""

    raw_tokens = 0
    clean_tokens = 0
    changed = 0
    for gloss in glosses:
        raw_parts = [part.strip() for part in gloss.replace("\u3000", " ").split("/") if part.strip()]
        cleaned = split_gloss_sequence(gloss, config)
        raw_tokens += len(raw_parts)
        clean_tokens += len(cleaned)
        if cleaned != raw_parts:
            changed += 1
    return {
        "samples": len(glosses),
        "raw_tokens": raw_tokens,
        "clean_tokens": clean_tokens,
        "samples_changed": changed,
        "mean_raw_length": (raw_tokens / len(glosses)) if glosses else 0.0,
        "mean_clean_length": (clean_tokens / len(glosses)) if glosses else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cslr.recognition")
    commands = parser.add_subparsers(dest="command", required=True)

    vocab = commands.add_parser("build-vocab", help="build an ordered gloss vocabulary")
    vocab.add_argument("--manifest", type=Path, default=Path("data/manifests/ce-csl.csv"))
    vocab.add_argument(
        "--output", type=Path, default=Path("data/manifests/ce-csl-gloss-seq-vocab.csv")
    )
    vocab.add_argument(
        "--report", type=Path, default=Path("artifacts/metrics/part4-data-profile.json")
    )
    vocab.add_argument("--min-frequency", type=int, default=2)
    vocab.add_argument("--max-tokens", type=int)
    vocab.add_argument(
        "--target-unit",
        choices=["token", "char"],
        default="token",
        help="predict whole gloss tokens (default) or individual characters",
    )
    vocab.add_argument("--drop-punctuation", action="store_true")
    vocab.add_argument("--drop-numeric-tokens", action="store_true")
    vocab.add_argument("--keep-variant-numbering", action="store_true")
    vocab.add_argument("--keep-annotations", action="store_true")
    vocab.add_argument("--compare-variants", action="store_true", default=True)
    vocab.add_argument("--no-compare-variants", dest="compare_variants", action="store_false")
    vocab.add_argument(
        "--features-root",
        type=Path,
        help="optional cached feature root; when given, feature integrity is reported",
    )

    train = commands.add_parser("train", help="train the CTC gloss recognizer")
    train.add_argument("--manifest", type=Path, default=Path("data/manifests/ce-csl.csv"))
    train.add_argument("--features", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/ctc-landmark96.pt"))
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--max-tokens", type=int)
    train.add_argument(
        "--target-unit",
        choices=["token", "char"],
        default="token",
        help="predict whole gloss tokens (default) or individual characters",
    )
    train.add_argument("--keep-variant-numbering", action="store_true")
    train.add_argument("--drop-punctuation", action="store_true")
    train.add_argument("--drop-numeric-tokens", action="store_true")
    train.add_argument("--keep-annotations", action="store_true")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--hidden-size", type=int, default=256)
    train.add_argument("--num-layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.3)
    train.add_argument("--projection-size", type=int, default=256)
    train.add_argument("--subsample-stride", type=int, default=1)
    train.add_argument("--input-size", type=int, default=None, help="defaults to the feature view width")
    train.add_argument(
        "--sequence-length",
        type=int,
        default=96,
        help="frames per clip the cached features were extracted with (recorded in the checkpoint)",
    )
    train.add_argument(
        "--feature-view",
        default="full",
        help="named block subset of the 368-dim vector, e.g. hands+hand_deltas",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    train.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--beam-width", type=int, default=1)
    train.add_argument("--limit-train", type=int)
    train.add_argument("--limit-validation", type=int)
    train.add_argument(
        "--present-only",
        action="store_true",
        help="train only on records whose feature file already exists (useful mid-extraction)",
    )
    train.add_argument(
        "--init-frontend",
        type=Path,
        help="contrastive-pretrain checkpoint whose normalize/projection warm-start the frontend",
    )
    train.add_argument(
        "--vocab-present-only",
        action="store_true",
        help="build the vocabulary from extracted train records only (keeps it honest mid-extraction)",
    )
    train.add_argument(
        "--metrics",
        type=Path,
        help="optional path for the machine-readable validation receipt",
    )
    evaluate = commands.add_parser(
        "evaluate", help="load a checkpoint and decode a split, writing metrics and predictions"
    )
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--manifest", type=Path, default=Path("data/manifests/ce-csl.csv"))
    evaluate.add_argument("--features", type=Path, required=True)
    evaluate.add_argument("--split", default="dev", choices=["train", "dev", "validation", "test"])
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/metrics/part4-eval.json"))
    evaluate.add_argument("--beam-width", type=int, default=1)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument(
        "--allow-missing-features",
        dest="present_only",
        action="store_false",
        default=True,
        help="fail instead of skipping records whose feature file is missing",
    )

    predict = commands.add_parser(
        "predict", help="run a video file through a trained checkpoint and print gloss tokens"
    )
    predict.add_argument("checkpoint", type=Path)
    predict.add_argument("video", type=Path)
    predict.add_argument("--beam-width", type=int, default=1)
    predict.add_argument("--device", default="auto")
    predict.add_argument("--sequence-length", type=int)
    predict.add_argument("--output", type=Path, help="optional JSON output path")
    return parser


def train_command(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    validate_manifest(records)
    vocab_records = records
    if args.vocab_present_only:
        vocab_records = filter_present(split_records(records, "train"), args.features)[0]
        if not vocab_records:
            raise ValueError("--vocab-present-only found no extracted train features")
        print(
            json.dumps(
                {
                    "vocab_source": "extracted train records only",
                    "vocab_train_records": len(vocab_records),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    vocabulary, _ = build_vocabulary_from_records(
        vocab_records,
        min_frequency=args.min_frequency,
        max_tokens=args.max_tokens,
        config=_config_from_args(args),
    )
    columns = feature_view_indices(args.feature_view)
    if columns is not None:
        inferred_input_size = len(columns)
    else:
        # "full" means whatever width the cached features actually have, so probe a real file
        # instead of assuming the 368-dim MediaPipe layout (a CLIP/SigLIP cache is much wider).
        inferred_input_size = probe_feature_width(args.manifest, args.features)
    input_size = args.input_size if args.input_size is not None else inferred_input_size
    if args.input_size is not None and input_size != inferred_input_size:
        raise ValueError(
            f"--input-size {input_size} does not match feature view {args.feature_view!r} "
            f"({inferred_input_size} dimensions)"
        )
    model_config = CTCConfig(
        input_size=input_size,
        vocabulary_size=vocabulary.size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        projection_size=args.projection_size,
        subsample_stride=args.subsample_stride,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        early_stopping_patience=args.patience,
        beam_width=args.beam_width,
        feature_view=args.feature_view,
        sequence_length=args.sequence_length,
    )
    result = train_ctc(
        manifest_path=args.manifest,
        feature_root=args.features,
        vocabulary=vocabulary,
        model_config=model_config,
        training_config=training_config,
        output_path=args.output,
        limit_train=args.limit_train,
        limit_validation=args.limit_validation,
        present_only=args.present_only,
        init_frontend=args.init_frontend,
    )
    payload = result.as_dict()
    payload["test_split_read"] = False
    payload["vocabulary_config"] = vocabulary.config.as_dict()
    if args.metrics:
        write_json(args.metrics, payload)
    else:
        default_metrics = args.output.with_suffix(".metrics.json")
        write_json(default_metrics, payload)
        payload["metrics_path"] = str(default_metrics)
    print(json.dumps(payload["validation_metrics"], ensure_ascii=False))
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    from cslr.recognition.inference import evaluate_checkpoint

    payload = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        feature_root=args.features,
        split=args.split,
        output_path=args.output,
        beam_width=args.beam_width,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        present_only=args.present_only,
    )
    summary = {
        "checkpoint": payload["checkpoint"],
        "split": payload["split"],
        "samples": payload["samples"],
        "latency_ms_per_sample": payload["latency_ms_per_sample"],
        "test_split_read": payload["test_split_read"],
        "output": str(args.output),
        "metrics": payload["metrics"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def predict_command(args: argparse.Namespace) -> int:
    from cslr.recognition.inference import predict_video

    payload = predict_video(
        checkpoint_path=args.checkpoint,
        video_path=args.video,
        beam_width=args.beam_width,
        device=args.device,
        sequence_length=args.sequence_length,
    )
    if args.output:
        write_json(args.output, payload)
        payload["output"] = str(args.output)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-vocab":
        return build_vocab_command(args)
    if args.command == "train":
        return train_command(args)
    if args.command == "evaluate":
        return evaluate_command(args)
    if args.command == "predict":
        return predict_command(args)
    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
