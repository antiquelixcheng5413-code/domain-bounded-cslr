"""CLI entrypoint for the isolated Part 3 pipeline.

``python -m cslr.translation`` — independent of the CTC CLI. Does not import or
modify any CTC training module, and does not touch the Web inference path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cslr.translation.config import Part3ConfigError, load_config
from cslr.translation.service import run_p2_chain, run_real_train, run_smoke


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cslr.translation", description="Part 3 SpaMo-style SLT pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="run the isolated smoke test")
    smoke.add_argument("--config", default="configs/translation_part3_smoke.yaml")
    smoke.add_argument("--split", default=None, help="override split: train|dev")
    smoke.add_argument("--limit", type=int, default=None)
    smoke.add_argument("--train-steps", type=int, default=None)
    smoke.add_argument("--dev-limit", type=int, default=None)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--synthetic-features", action="store_true", help="use synthetic visual features")

    extract = sub.add_parser(
        "extract",
        help="extract and cache real RGB/motion/landmark features (Phase 2)",
    )
    extract.add_argument("--manifest", default="data/manifests/ce-csl.csv")
    extract.add_argument("--video-root", default="data/raw/CE-CSL")
    extract.add_argument("--landmark-root", default="data/processed/ce_csl")
    extract.add_argument("--cache-root", default="artifacts/part3_features")
    extract.add_argument("--split", required=True, choices=["train", "validation", "dev", "test"])
    extract.add_argument("--start", type=int, default=0)
    extract.add_argument("--limit", type=int, default=None)
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--encoder", default="ViT-B-32")

    chain = sub.add_parser(
        "p2-chain",
        help="run the real-feature -> fusion -> decoder -> metrics chain check",
    )
    chain.add_argument("--config", default="configs/translation_part3_smoke.yaml")
    chain.add_argument("--feature-root", default="artifacts/part3_features")
    chain.add_argument("--split", default="validation", choices=["train", "validation", "dev"])
    chain.add_argument("--limit", type=int, default=2)
    chain.add_argument("--device", default=None)

    train = sub.add_parser(
        "train-real",
        help="teacher-forcing baseline on cached real features (frozen eval split)",
    )
    train.add_argument("--config", default="configs/translation_part3_baseline.yaml")
    train.add_argument("--feature-root", default="artifacts/part3_features")
    train.add_argument("--train-split", default="train")
    train.add_argument("--eval-split", default="validation")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--eval-limit", type=int, default=20)
    train.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "extract":
        if args.split == "test":
            print("Part3DataError: split 'test' is frozen for Part 3 and cannot be extracted")
            return 2
        return _run_extract(args)

    try:
        cfg = load_config(args.config)
    except Part3ConfigError as exc:
        print(f"Part3ConfigError: {exc}")
        return 2

    if args.command == "p2-chain":
        split = "validation" if args.split in ("dev", "validation") else args.split
        device = args.device or cfg.device
        summary = run_p2_chain(
            cfg,
            root=cfg.source_config_path.parent if cfg.source_config_path else Path.cwd(),
            feature_root=Path(args.feature_root),
            split=split,
            limit=args.limit,
            device=device,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "train-real":
        # eval split 'test' is authoritative-rejected here too
        if args.eval_split == "test":
            print("Part3DataError: test split is frozen for Part 3")
            return 2
        if args.train_split == "test":
            print("Part3DataError: test split is frozen for Part 3")
            return 2
        train_split = "validation" if args.train_split in ("dev", "validation") else args.train_split
        eval_split = "validation" if args.eval_split in ("dev", "validation") else args.eval_split
        device = args.device or cfg.device
        summary = run_real_train(
            cfg,
            root=cfg.source_config_path.parent if cfg.source_config_path else Path.cwd(),
            feature_root=Path(args.feature_root),
            train_split=train_split,
            eval_split=eval_split,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            eval_limit=args.eval_limit,
            device=device,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command != "smoke":
        parser.error(f"unknown command {args.command!r}")

    summary = run_smoke(
        cfg,
        train_steps=args.train_steps if args.train_steps is not None else cfg.smoke.train_steps,
        dev_limit=args.dev_limit if args.dev_limit is not None else cfg.smoke.dev_limit,
        synthetic=args.synthetic_features or cfg.smoke.synthetic_features,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _run_extract(args: argparse.Namespace) -> int:
    from cslr.data.manifest import read_manifest
    from cslr.translation.extract import RealFeatureExtractor

    root = Path(".")
    manifest_path = root / args.manifest
    video_root = root / args.video_root
    landmark_root = root / args.landmark_root
    cache_root = root / args.cache_root

    norm_split = "validation" if args.split in ("dev", "validation") else args.split
    records = [r for r in read_manifest(manifest_path) if r.split == norm_split]
    if args.start > 0:
        records = records[args.start :]
    if args.limit is not None and args.limit > 0:
        records = records[: args.limit]
    if not records:
        print(f"no records for split={norm_split} in {manifest_path}")
        return 0

    extractor = RealFeatureExtractor(
        landmark_root=landmark_root,
        cache_root=cache_root,
        split=norm_split,
        encoder_name=args.encoder,
        device=args.device,
    )
    done = {"rgb": 0, "motion": 0, "landmark": 0}
    skipped = 0
    split_cache = extractor.cache_root
    modalities = ("rgb", "motion", "landmark")
    for record in records:
        sample_paths = {m: split_cache / f"{record.sample_id}.{m}.npy" for m in modalities}
        if all(p.exists() for p in sample_paths.values()):
            skipped += 1
            continue
        video_path = video_root / record.video
        try:
            saved = extractor.cache(record.sample_id, video_path)
            for mod in done:
                done[mod] += 1
        except Exception as exc:  # surface per-sample instead of killing the run
            print(f"[{record.sample_id}] FAILED: {exc}", file=sys.stderr)
            continue

    summary = {
        "status": "ok" if (done["rgb"] or skipped) else "partial",
        "split": norm_split,
        "requested": len(records),
        "cached": done,
        "skipped_cached": skipped,
        "cache_root": str(split_cache),
        "encoder": args.encoder,
        "device": args.device,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())