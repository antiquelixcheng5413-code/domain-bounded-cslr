from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cslr.data.adapters import available_adapters, get_adapter
from cslr.data.gloss import build_gloss_vocabulary
from cslr.data.manifest import read_manifest, validate_manifest, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cslr")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list-adapters", help="list registered dataset adapters")

    manifest = commands.add_parser("validate-manifest", help="validate a dataset manifest")
    manifest.add_argument("path", type=Path)

    build_manifest = commands.add_parser(
        "build-manifest", help="build a shared manifest from a dataset config"
    )
    build_manifest.add_argument("config", type=Path)
    build_manifest.add_argument("--output", type=Path, required=True)
    build_manifest.add_argument("--data-root", type=Path)

    extract = commands.add_parser(
        "extract", help="extract MediaPipe landmarks from one video or image sequence"
    )
    extract.add_argument("source", type=Path)
    extract.add_argument("--output", type=Path, required=True)

    extract_manifest = commands.add_parser(
        "extract-manifest", help="extract landmarks for records in a manifest"
    )
    extract_manifest.add_argument("manifest", type=Path)
    extract_manifest.add_argument("--data-root", type=Path, required=True)
    extract_manifest.add_argument("--output", type=Path, required=True)
    extract_manifest.add_argument("--limit", type=int)
    extract_manifest.add_argument("--overwrite", action="store_true")
    extract_manifest.add_argument("--continue-on-error", action="store_true")
    extract_manifest.add_argument("--report", type=Path)

    gloss_vocab = commands.add_parser(
        "build-gloss-vocab", help="build a CE-CSL gloss token vocabulary from a manifest"
    )
    gloss_vocab.add_argument("manifest", type=Path)
    gloss_vocab.add_argument("--output", type=Path, required=True)
    gloss_vocab.add_argument("--split", default="train")
    gloss_vocab.add_argument("--min-frequency", type=int, default=2)
    gloss_vocab.add_argument("--max-tokens", type=int)

    verify_features = commands.add_parser(
        "verify-features", help="validate a local landmark bundle against a manifest"
    )
    verify_features.add_argument("manifest", type=Path)
    verify_features.add_argument("--features", type=Path, required=True)
    verify_features.add_argument("--sha256", action="store_true")
    verify_features.add_argument("--receipt", type=Path)

    train = commands.add_parser("train", help="train a temporal model from extracted features")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--features", type=Path, required=True)
    train.add_argument("--model-config", type=Path, default=Path("configs/models/lstm.yaml"))
    train.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    train.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/lstm.pt"))

    export = commands.add_parser("export", help="export a checkpoint to ONNX")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--sequence-length", type=int, default=48)

    train_ctc = commands.add_parser("train-ctc", help="train a CTC sequence recognizer")
    train_ctc.add_argument("--manifest", type=Path, required=True)
    train_ctc.add_argument("--features", type=Path, required=True)
    train_ctc.add_argument("--vocab", type=Path, required=True)
    train_ctc.add_argument(
        "--model-config", type=Path, default=Path("configs/models/ctc_lstm.yaml")
    )
    train_ctc.add_argument(
        "--training-config", type=Path, default=Path("configs/training_ctc.yaml")
    )
    train_ctc.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/ctc_v2.pt"))
    train_ctc.add_argument("--feature-receipt", type=Path)
    train_ctc.add_argument("--resume", type=Path)

    evaluate_ctc = commands.add_parser(
        "evaluate-ctc", help="evaluate a CTC checkpoint or ONNX model"
    )
    evaluate_ctc.add_argument("--manifest", type=Path, required=True)
    evaluate_ctc.add_argument("--features", type=Path, required=True)
    evaluate_ctc.add_argument("--model", type=Path, required=True)
    evaluate_ctc.add_argument("--model-kind", choices=["legacy_ctc", "ctc_v2"], required=True)
    evaluate_ctc.add_argument("--vocab", type=Path)
    evaluate_ctc.add_argument("--split", choices=["validation", "test"], required=True)
    evaluate_ctc.add_argument("--output", type=Path, required=True)

    export_ctc = commands.add_parser("export-ctc", help="export a ctc_v2 checkpoint to ONNX")
    export_ctc.add_argument("checkpoint", type=Path)
    export_ctc.add_argument("output", type=Path)

    commands.add_parser("gpu-preflight", help="verify CUDA access from the current container")

    benchmark_ctc = commands.add_parser(
        "benchmark-ctc", help="benchmark CTC v2 extraction, ONNX inference, and total latency"
    )
    benchmark_ctc.add_argument("--manifest", type=Path, required=True)
    benchmark_ctc.add_argument("--data-root", type=Path, required=True)
    benchmark_ctc.add_argument("--model", type=Path, required=True)
    benchmark_ctc.add_argument("--split", choices=["validation", "test"], default="validation")
    benchmark_ctc.add_argument("--limit", type=int, default=50)
    benchmark_ctc.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-adapters":
        print("\n".join(available_adapters()))
        return 0
    if args.command == "validate-manifest":
        records = read_manifest(args.path)
        validate_manifest(records)
        print(json.dumps({"status": "ok", "records": len(records)}))
        return 0
    if args.command == "build-manifest":
        import yaml

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        adapter_name = config.get("adapter")
        adapter = get_adapter(adapter_name).from_config(config, data_root_override=args.data_root)
        records = list(adapter.records())
        validate_manifest(records)
        count = write_manifest(args.output, records)
        print(json.dumps({"status": "ok", "records": count, "output": str(args.output)}))
        return 0
    if args.command == "extract":
        import numpy as np

        from cslr.features.extractor import MediaPipeHolisticExtractor

        result = MediaPipeHolisticExtractor().extract(args.source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, result.features)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "shape": list(result.features.shape),
                    "quality": result.quality.__dict__,
                },
                ensure_ascii=False,
            )
        )
        return 0 if result.quality.accepted else 2
    if args.command == "extract-manifest":
        from cslr.features.batch import extract_manifest_features, write_extraction_report
        from cslr.features.extractor import MediaPipeHolisticExtractor

        records = read_manifest(args.manifest)
        validate_manifest(records)
        summary = extract_manifest_features(
            records=records,
            data_root=args.data_root,
            output_root=args.output,
            extractor=MediaPipeHolisticExtractor(),
            limit=args.limit,
            overwrite=args.overwrite,
            continue_on_error=args.continue_on_error,
        )
        payload = summary.as_dict()
        if args.report:
            payload["report_records"] = write_extraction_report(args.report, summary.items)
            payload["report"] = str(args.report)
        print(json.dumps(payload, ensure_ascii=False))
        return 1 if summary.failed else 0
    if args.command == "build-gloss-vocab":
        records = read_manifest(args.manifest)
        validate_manifest(records)
        split_records = [record for record in records if record.split == args.split]
        if not split_records:
            raise ValueError(f"manifest contains no records for split: {args.split}")
        tokens, counts = build_gloss_vocabulary(
            split_records,
            min_frequency=args.min_frequency,
            max_tokens=args.max_tokens,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["token", "frequency"])
            writer.writeheader()
            for token in tokens:
                writer.writerow({"token": token, "frequency": counts[token]})
        print(
            json.dumps(
                {
                    "status": "ok",
                    "split": args.split,
                    "tokens": len(tokens),
                    "output": str(args.output),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "verify-features":
        from cslr.data.feature_bundle import verify_feature_bundle

        records = read_manifest(args.manifest)
        validate_manifest(records)
        result = verify_feature_bundle(records, args.features, include_sha256=args.sha256)
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            result["receipt"] = str(args.receipt)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "ok" else 1
    if args.command == "train":
        from cslr.training.runner import train_model

        result = train_model(
            manifest_path=args.manifest,
            feature_root=args.features,
            model_config_path=args.model_config,
            training_config_path=args.training_config,
            output_path=args.output,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "export":
        from cslr.training.export import export_checkpoint_to_onnx

        result = export_checkpoint_to_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            sequence_length=args.sequence_length,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "train-ctc":
        from cslr.training.ctc import train_ctc_model

        result = train_ctc_model(
            manifest_path=args.manifest,
            feature_root=args.features,
            vocabulary_path=args.vocab,
            model_config_path=args.model_config,
            training_config_path=args.training_config,
            output_path=args.output,
            feature_receipt_path=args.feature_receipt,
            resume_path=args.resume,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "evaluate-ctc":
        from cslr.training.ctc import (
            evaluate_ctc_checkpoint,
            evaluate_ctc_onnx,
            evaluate_legacy_ctc_checkpoint,
            write_ctc_evaluation,
        )

        if args.model.suffix.lower() == ".onnx":
            metrics, records = evaluate_ctc_onnx(
                model_path=args.model,
                vocabulary_path=args.vocab,
                manifest_path=args.manifest,
                feature_root=args.features,
                split=args.split,
                model_kind=args.model_kind,
            )
        elif args.model_kind == "legacy_ctc":
            if args.vocab is None:
                raise ValueError("legacy_ctc checkpoint evaluation requires --vocab")
            metrics, records = evaluate_legacy_ctc_checkpoint(
                checkpoint_path=args.model,
                vocabulary_path=args.vocab,
                manifest_path=args.manifest,
                feature_root=args.features,
                split=args.split,
            )
        else:
            metrics, records = evaluate_ctc_checkpoint(
                checkpoint_path=args.model,
                manifest_path=args.manifest,
                feature_root=args.features,
                split=args.split,
            )
        summary_path = write_ctc_evaluation(args.output, metrics, records)
        print(
            json.dumps(
                {
                    "metrics": metrics,
                    "output": str(args.output),
                    "summary": str(summary_path),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "export-ctc":
        from cslr.training.ctc import export_ctc_checkpoint_to_onnx

        result = export_ctc_checkpoint_to_onnx(args.checkpoint, args.output)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "gpu-preflight":
        from cslr.training.gpu import gpu_preflight

        result = gpu_preflight()
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "ok" else 2
    if args.command == "benchmark-ctc":
        from cslr.training.ctc import benchmark_ctc_end_to_end, write_ctc_benchmark

        metrics, rows = benchmark_ctc_end_to_end(
            model_path=args.model,
            manifest_path=args.manifest,
            data_root=args.data_root,
            split=args.split,
            limit=args.limit,
        )
        summary_path = write_ctc_benchmark(args.output, metrics, rows)
        print(
            json.dumps(
                {"metrics": metrics, "output": str(args.output), "summary": str(summary_path)},
                ensure_ascii=False,
            )
        )
        return 0
    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
