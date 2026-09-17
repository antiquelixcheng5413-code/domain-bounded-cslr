"""Phase-2 real feature extraction and caching for Part 3.

RGB / motion features are computed frame-wise from the source video with a
pretrained vision encoder (CLIP ViT-B/32 by default); motion is the adjacent
frame difference of the RGB tokens (P3-04 first version). Landmark features are
read-only reuse of the existing Part1/2 ``(48, 368)`` MediaPipe cache.

All outputs are saved through :mod:`cslr.translation.cache` with a SHA-256
receipt and keyed by ``sample_id`` per modality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cslr.translation.cache import load_feature, save_feature

# Official CLIP ViT-B/32 normalization matches ImageNet-style preprocessing.
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_DEFAULT_ENCODER = "ViT-B-32"
_DEFAULT_RGB_DIM = 512
_FRAME_BATCH = 16


@dataclass
class ExtractedSample:
    rgb: np.ndarray
    motion: np.ndarray
    landmark: np.ndarray
    frames: int


class FeatureExtractionError(RuntimeError):
    pass


class ClipFrameEncoder:
    """Frame-wise vision encoder returning one CLS token per frame."""

    def __init__(self, model_name: str = _DEFAULT_ENCODER, device: str = "cpu") -> None:
        import open_clip
        import torch as _t

        self.device = _t.device(device)
        self.encoder_name = model_name
        model, _preprocess, _ = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        self.model = model.visual.eval().to(self.device)
        self._t = _t
        self.mean = _t.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = _t.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self.embed_dim = model.visual.output_dim

    @staticmethod
    def _load_frame_rgb_bytes(frame: np.ndarray) -> np.ndarray:
        import cv2

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def encode_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """Return ``[N, D]`` frame-level CLS tokens with no media pipeline."""

        if not frames:
            raise FeatureExtractionError("no frames supplied to encoder")
        t = self._t
        tokens: list[np.ndarray] = []
        for start in range(0, len(frames), _FRAME_BATCH):
            chunk = frames[start : start + _FRAME_BATCH]
            batch = t.from_numpy(np.stack([self._load_frame_rgb_bytes(f) for f in chunk]))
            batch = batch.permute(0, 3, 1, 2).to(self.device)
            batch = (batch - self.mean) / self.std
            with t.inference_mode():
                out = self.model(batch)  # [B, D] CLIP space
            tokens.append(out.cpu().numpy())
        return np.concatenate(tokens, axis=0).astype(np.float32)


def compute_motion(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[0] < 2:
        raise FeatureExtractionError("motion requires at least 2 rgb frames")
    return (rgb[1:] - rgb[:-1]).astype(np.float32)


class RealFeatureExtractor:
    """Video -> RGB/motion via CLIP, landmark reused from the existing cache."""

    def __init__(
        self,
        *,
        landmark_root: Path,
        cache_root: Path,
        split: str,
        encoder_name: str = _DEFAULT_ENCODER,
        rgb_dim: int = _DEFAULT_RGB_DIM,
        device: str = "cpu",
        allow_landmark_overwrite: bool = False,
    ) -> None:
        self.landmark_root = Path(landmark_root)
        self.cache_root = Path(cache_root) / split
        self.split = split
        self.encoder_name = encoder_name
        self.rgb_dim = rgb_dim
        self.allow_landmark_overwrite = allow_landmark_overwrite
        self.encoder = ClipFrameEncoder(encoder_name, device)
        self.device = device

    def landmark_path(self, sample_id: str) -> Path:
        return self.landmark_root / f"{sample_id}.npy"

    def load_landmark(self, sample_id: str) -> np.ndarray:
        path = self.landmark_path(sample_id)
        if not path.exists():
            raise FeatureExtractionError(f"landmark feature not found: {path}")
        array = np.load(path).astype(np.float32)
        if not np.isfinite(array).all():
            raise FeatureExtractionError(f"non-finite landmark feature: {path}")
        return array

    def read_video_frames(self, video_path: Path) -> list[np.ndarray]:
        import cv2

        if not video_path.exists():
            raise FeatureExtractionError(f"video not found: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FeatureExtractionError(f"unable to open video: {video_path}")
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        if not frames:
            raise FeatureExtractionError(f"no decodable frames in video: {video_path}")
        return frames

    def extract(self, sample_id: str, video_path: Path) -> ExtractedSample:
        frames = self.read_video_frames(video_path)
        rgb = self.encoder.encode_frames(frames)
        motion = compute_motion(rgb)
        landmark = self.load_landmark(sample_id)
        return ExtractedSample(rgb=rgb, motion=motion, landmark=landmark, frames=len(frames))

    def cache(self, sample_id: str, video_path: Path) -> dict[str, Path]:
        sample = self.extract(sample_id, video_path)
        save_feature(
            self.cache_root, sample_id, "rgb", sample.rgb,
            model_name=self.encoder_name, version="part3-p2-clip",
            frame_sampling="frame-wise",
        )
        save_feature(
            self.cache_root, sample_id, "motion", sample.motion,
            model_name=self.encoder_name, version="part3-p2-clip",
            frame_sampling="adjacent-frame-diff",
        )
        # Landmark is a read-only reuse of the Part1/2 cache; copy into the
        # Part 3 cache keyed by modality without touching the source file.
        save_feature(
            self.cache_root, sample_id, "landmark", sample.landmark,
            model_name="mediapipe-holistic", version="part1/2-cache",
            frame_sampling="resample-48",
            allow_overwrite=self.allow_landmark_overwrite,
        )
        return {
            "rgb": self.cache_root / f"{sample_id}.rgb.npy",
            "motion": self.cache_root / f"{sample_id}.motion.npy",
            "landmark": self.cache_root / f"{sample_id}.landmark.npy",
        }


def load_cached_features(cache_root: Path, sample_id: str) -> dict[str, np.ndarray]:
    return {
        mod: load_feature(cache_root, sample_id, mod)
        for mod in ("rgb", "motion", "landmark")
    }