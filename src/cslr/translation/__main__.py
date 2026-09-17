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
from cslr.translation.service import run_smoke


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "smoke":
        parser.error(f"unknown command {args.command!r}")

    try:
        cfg = load_config(args.config)
    except Part3ConfigError as exc:
        print(f"Part3ConfigError: {exc}")
        return 2

    summary = run_smoke(
        cfg,
        train_steps=args.train_steps if args.train_steps is not None else cfg.smoke.train_steps,
        dev_limit=args.dev_limit if args.dev_limit is not None else cfg.smoke.dev_limit,
        synthetic=args.synthetic_features or cfg.smoke.synthetic_features,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())