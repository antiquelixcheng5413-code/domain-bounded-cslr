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
    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
