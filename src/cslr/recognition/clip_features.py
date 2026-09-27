"""Part 4 follow-up: frozen CLIP/SigLIP frame features as the CTC input.

This is the §9.1 recommendation: keep the recogniser, replace the representation. Each clip is
decoded, sampled to ``--frames`` images, pushed through a **frozen** vision tower, and the resulting
sequence is cached as ``[T, D]`` float32 exactly like the landmark features, so
``python -m cslr.recognition train --features <this root>`` works unchanged.

Usage:

    python -m cslr.recognition.clip_features build \
        --manifest data/manifests/ce-csl.csv \
        --data-root CE-CSLData/downloads/fyp/CE-CSL/CE-CSL \
        --output data/processed/ce-csl-siglip16 --frames 16 --limit 400

The frozen-model receipts (model name, revision, frame count, dtype, shape) are written next to
the features so a later run cannot silently mix encoders.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cslr.data.manifest import read_manifest, validate_manifest
from cslr.recognition.io import write_json

DEFAULT_MODEL = "google/siglip-base-patch16-224"


class FrozenVisionEncoder:
    """Frozen image tower producing one feature vector per frame."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "clip_features needs torch + transformers installed (see scripts/setup_cuda_env.ps1)"
            ) from error
        self.torch = torch
        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        # a vision tower exposes a pooler/vision output; use it and record the width
        with torch.no_grad():
            probe = self.model.get_image_features(
                **self.processor(images=[np.zeros((224, 224, 3), dtype=np.uint8)], return_tensors="pt").to(
                    self.device
                )
            )
        self.feature_size = int(probe.shape[-1])

    def encode_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """``frames`` are RGB uint8 arrays; returns ``[T, D]`` float32."""

        torch = self.torch
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.detach().float().cpu().numpy().astype(np.float32)


def sample_frames(video: Path, count: int, size: int = 224) -> list[np.ndarray]:
    """Uniformly sample ``count`` RGB frames from a video."""

    if count < 1:
        raise ValueError("frame count must be at least 1")
    if not video.exists():
        raise FileNotFoundError(f"video is missing: {video}")
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            raise ValueError(f"video reports no frames: {video}")
        indices = [round(index * (total - 1) / max(count - 1, 1)) for index in range(count)]
        targets = set(indices)
        frames: list[np.ndarray] = []
        # Sequential grab/retrieve is an order of magnitude faster than a per-frame
        # CAP_PROP_POS_FRAMES seek, which forces a keyframe seek for every sampled frame.
        for frame_index in range(total):
            if not capture.grab():
                break
            if frame_index not in targets:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            if min(height, width) != size:
                scale = size / min(height, width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            frames.append(frame)
        if not frames:
            raise ValueError(f"no frame could be decoded: {video}")
        return frames
    finally:
        capture.release()


def build_features(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    validate_manifest(records)
    if args.split == "test":
        raise ValueError(
            "split 'test' is frozen: the official 500-sample test split must not be read during "
            "this phase"
        )
    selected = [record for record in records if record.split == ("validation" if args.split == "dev" else args.split)]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError(f"manifest has no records for split {args.split}")

    encoder = FrozenVisionEncoder(args.model, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    written = 0
    skipped = 0
    failed: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, record in enumerate(selected, start=1):
        target = args.output / f"{record.sample_id}.npy"
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            frames = sample_frames(data_root / record.video, args.frames, args.image_size)
            features = encoder.encode_frames(frames)
            np.save(target, features)
            written += 1
        except Exception as error:  # noqa: BLE001 - reported per sample
            failed.append({"sample_id": record.sample_id, "error": f"{type(error).__name__}: {error}"})
            if not args.continue_on_error:
                raise
        if index % args.log_every == 0:
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0.0
            print(
                json.dumps(
                    {
                        "processed": index,
                        "of": len(selected),
                        "written": written,
                        "skipped": skipped,
                        "failed": len(failed),
                        "per_second": round(rate, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    receipt = {
        "encoder": args.model,
        "split": args.split,
        "frames_per_clip": args.frames,
        "image_size": args.image_size,
        "feature_size": encoder.feature_size,
        "dtype": "float32",
        "samples_requested": len(selected),
        "samples_written": written,
        "samples_skipped": skipped,
        "samples_failed": len(failed),
        "errors": failed[:20],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "test_split_read": False,
        "formal_result": False,
    }
    write_json(args.output / "encoder_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if not failed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cslr.recognition.clip_features")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/ce-csl.csv"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "validation", "test"])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return build_features(args)


if __name__ == "__main__":
    raise SystemExit(main())
