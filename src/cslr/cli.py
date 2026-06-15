from __future__ import annotations

import argparse
import json
from pathlib import Path

from cslr.data.adapters import available_adapters
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

        from cslr.data.adapters import NationalCSLDPImageSequenceAdapter

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        adapter_name = config.get("adapter")
        if adapter_name != "nationalcsl_dp_image_sequence":
            raise ValueError(f"unsupported build-manifest adapter: {adapter_name}")
        adapter = NationalCSLDPImageSequenceAdapter.from_config(
            config, data_root_override=args.data_root
        )
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
