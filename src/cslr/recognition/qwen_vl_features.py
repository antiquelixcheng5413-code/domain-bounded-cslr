"""Part 4 follow-up: frozen Qwen2.5-VL vision tower as the CTC input representation.

This is the §9.1 recommendation ("swap the encoder, keep the head") implemented with the model
that is already on disk, so no new download is needed. Each clip is decoded, sampled to
``--frames`` images, pushed through the **frozen** Qwen2.5-VL vision tower, and the mean-pooled
patch embedding per frame is cached as ``[T, D]`` float32 — the same contract the landmark
features use, so ``python -m cslr.recognition train --features <root>`` runs unchanged.

Usage:

    python -m cslr.recognition.qwen_vl_features build \
        --data-root CE-CSLData/downloads/fyp/CE-CSL/CE-CSL \
        --output data/processed/ce-csl-qwenvl --split dev --frames 16 --limit 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cslr.data.manifest import read_manifest, validate_manifest
from cslr.recognition.clip_features import sample_frames
from cslr.recognition.io import write_json

DEFAULT_MODEL = r"G:\cslr-tools\models\Qwen2.5-VL-3B-Instruct"


class FrozenQwenVisionEncoder:
    """Frozen Qwen2.5-VL vision tower returning one vector per sampled frame."""

    def __init__(self, model_path: str = DEFAULT_MODEL, device: str | None = None) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "qwen_vl_features needs torch + transformers (see scripts/setup_cuda_env.ps1)"
            ) from error
        self.torch = torch
        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoProcessor.from_pretrained(model_path)
        # load in bfloat16 and drop the language model: only the vision tower is used
        full = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        # this transformers build exposes the tower as ``model.visual`` (there is no top-level
        # ``visual`` attribute), verified by inspecting the module tree
        self.visual = full.model.visual.to(self.device).eval()
        del full
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.feature_size = self._probe_feature_size()

    def _probe_feature_size(self) -> int:
        torch = self.torch
        probe = np.zeros((224, 224, 3), dtype=np.uint8)
        with torch.no_grad():
            pixel_values, grid = self._pixel_values([probe])
            out = self.visual(pixel_values, grid_thw=grid)
        return int(self._pool_per_frame(out, 1).shape[-1])

    def _pixel_values(self, frames: list[np.ndarray]):
        processed = self.processor.image_processor(images=frames, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device)
        grid = processed["image_grid_thw"].to(self.device)
        return pixel_values, grid

    def _pool_per_frame(self, out, frame_count: int) -> "np.ndarray":
        """Merge the per-frame visual tokens into one vector per frame.

        The tower returns ``pooler_output`` of shape ``[frame_count * tokens_per_frame, D]`` for a
        batch of independent images (verified: 4 images of 224x224 give 256 rows at D=2048, i.e. 64
        tokens per frame). Averaging inside each frame keeps the sequence length equal to the
        number of sampled frames, which is exactly what the CTC head expects.
        """

        pooled = out.pooler_output
        if pooled is None:
            raise RuntimeError("vision tower did not return pooled tokens; cannot pool per frame")
        rows = pooled.shape[0]
        if frame_count < 1 or rows % frame_count:
            raise RuntimeError(
                f"vision tower returned {rows} pooled tokens, not divisible by {frame_count} frames"
            )
        per_frame = rows // frame_count
        return pooled.reshape(frame_count, per_frame, pooled.shape[-1]).mean(dim=1)

    def encode_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """Returns ``[T, D]`` float32 (visual tokens mean-pooled per frame)."""

        torch = self.torch
        with torch.no_grad():
            pixel_values, grid = self._pixel_values(frames)
            out = self.visual(pixel_values, grid_thw=grid)
            features = self._pool_per_frame(out, len(frames))
        return features.detach().float().cpu().numpy().astype(np.float32)


def build_features(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    validate_manifest(records)
    if args.split == "test":
        raise ValueError(
            "split 'test' is frozen: the official 500-sample test split must not be read during "
            "this phase"
        )
    split_name = "validation" if args.split == "dev" else args.split
    selected = [record for record in records if record.split == split_name]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError(f"manifest has no records for split {args.split}")

    encoder = FrozenQwenVisionEncoder(args.model, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    written = skipped = 0
    failed: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, record in enumerate(selected, start=1):
        target = args.output / f"{record.sample_id}.npy"
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            frames = sample_frames(data_root / record.video, args.frames, args.image_size)
            np.save(target, encoder.encode_frames(frames))
            written += 1
        except Exception as error:  # noqa: BLE001 - reported per sample
            failed.append({"sample_id": record.sample_id, "error": f"{type(error).__name__}: {error}"})
            if not args.continue_on_error:
                raise
        if index % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "processed": index,
                        "of": len(selected),
                        "written": written,
                        "skipped": skipped,
                        "failed": len(failed),
                        "per_second": round(index / elapsed, 3) if elapsed else 0.0,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    receipt = {
        "encoder": "Qwen2.5-VL-3B vision tower (frozen)",
        "model_path": args.model,
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
    parser = argparse.ArgumentParser(prog="python -m cslr.recognition.qwen_vl_features")
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
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return build_features(args)


if __name__ == "__main__":
    raise SystemExit(main())
