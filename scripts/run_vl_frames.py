"""Run the frozen Qwen2.5-VL vision-tower feature extraction + CTC training + dev eval
for a requested frame count. This is the §9.1 "swap the representation, keep the recogniser"
recommendation, now testing 48/96 sampled frames (the 16-frame cache held too little
information for long glosses).

Usage (from the repo root, PYTHONPATH=src is set internally):

    python scripts/run_vl_frames.py --frames 48
    python scripts/run_vl_frames.py --frames 48 96 --vocab-cap 300 \
        --model /mnt/d/cslr-tools/models/Qwen2.5-VL-3B-Instruct

Pipeline per frame count (all dev/train only; the frozen test split is never read):

1. extract Qwen2.5-VL vision-tower features to ``data/processed/ce-csl-qwenvl{frames}``
2. train the CTC recognizer (same head as the landmark baseline)
3. evaluate that checkpoint on the full dev split and print the score table.

Every receipt sets ``test_split_read: false`` and ``formal_result: false``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / "venv" / "bin" / "python3"
VENV_BIN = REPO / "venv" / "bin"
MODEL_DEFAULT = "/mnt/d/cslr-tools/models/Qwen2.5-VL-3B-Instruct"


def _env() -> dict[str, str]:
    env = {
        "PYTHONPATH": str(REPO / "src"),
        "HF_HOME": "/home/su127/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
    }
    return env


def run(args: list[str], what: str, strict: bool = True) -> None:
    cmd = [str(VENV_BIN / "python3"), *args]
    print(f"\n=== {what} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=_env(), cwd=REPO)
    if proc.returncode != 0 and strict:
        raise SystemExit(f"{what} failed with exit code {proc.returncode}")
    if proc.returncode != 0:
        print(f"  (warning: {what} exited {proc.returncode}; per-sample failures are expected)")


def extract(data_root: Path, split: str, frames: int, out: Path, model: str, device: str) -> None:
    run(
        [
            "-m",
            "cslr.recognition.qwen_vl_features",
            "--data-root",
            str(data_root),
            "--output",
            str(out),
            "--split",
            split,
            "--frames",
            str(frames),
            "--model",
            model,
            "--device",
            device,
            "--continue-on-error",
            "--log-every",
            "50",
        ],
        f"extract VL features ({split}, T={frames})",
        strict=False,
    )


def train(features: Path, frames: int, vocab_cap: int) -> None:
    run(
        [
            "-m",
            "cslr.recognition",
            "train",
            "--manifest",
            str(REPO / "data/manifests/ce-csl.csv"),
            "--features",
            str(features),
            "--output",
            str(REPO / f"artifacts/checkpoints/ctc-vl{frames}-cap{vocab_cap}.pt"),
            "--epochs",
            "150",
            "--batch-size",
            "32",
            "--learning-rate",
            "0.0012",
            "--patience",
            "40",
            "--device",
            "auto",
            "--present-only",
            "--vocab-present-only",
            "--max-tokens",
            str(vocab_cap),
            "--metrics",
            str(REPO / f"artifacts/metrics/part4-vl{frames}-cap{vocab_cap}-train.json"),
        ],
        f"train CTC (T={frames}, vocab cap {vocab_cap})",
    )


def evaluate(features: Path, frames: int, vocab_cap: int) -> None:
    run(
        [
            "-m",
            "cslr.recognition",
            "evaluate",
            str(REPO / f"artifacts/checkpoints/ctc-vl{frames}-cap{vocab_cap}.pt"),
            "--features",
            str(features),
            "--split",
            "dev",
            "--output",
            str(REPO / f"artifacts/metrics/part4-eval-vl{frames}-cap{vocab_cap}.json"),
        ],
        f"evaluate on dev (T={frames}, vocab cap {vocab_cap})",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", default=[48])
    parser.add_argument("--vocab-cap", type=int, default=300)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=str(REPO / "data/raw/CE-CSL"))
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_extract:
        for frames in args.frames:
            out = REPO / f"data/processed/ce-csl-qwenvl{F_out(frames)}"
            extract(Path(args.data_root), "train", frames, out, args.model, args.device)
            extract(Path(args.data_root), "dev", frames, out, args.model, args.device)

    if not args.skip_train:
        for frames in args.frames:
            features = REPO / f"data/processed/ce-csl-qwenvl{F_out(frames)}"
            train(features, frames, args.vocab_cap)
            evaluate(features, frames, args.vocab_cap)

    print("\nVL_FRAMES_SWEEP_DONE")
    return 0


def F_out(value: int) -> str:
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())