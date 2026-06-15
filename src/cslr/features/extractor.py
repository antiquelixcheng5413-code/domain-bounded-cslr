from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cslr.contracts import QualityReport
from cslr.features.quality import assess_frame_quality

POSE_INDICES = (11, 12, 13, 14, 15, 16, 23, 24)
FACE_INDICES = (10, 33, 61, 133, 152, 263, 291, 362)
BASE_FEATURE_SIZE = 182
MASK_SIZE = 4
OUTPUT_FEATURE_SIZE = BASE_FEATURE_SIZE + MASK_SIZE + BASE_FEATURE_SIZE
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class ExtractionResult:
    features: np.ndarray
    quality: QualityReport
    source_frames: int


class MediaPipeHolisticExtractor:
    """Extract a compact holistic sequence from a video.

    MediaPipe is imported lazily so manifest and evaluation utilities remain usable without
    the ML dependency set.
    """

    def __init__(self, sequence_length: int = 48, minimum_valid_ratio: float = 0.80) -> None:
        self.sequence_length = sequence_length
        self.minimum_valid_ratio = minimum_valid_ratio

    def extract(self, source_path: Path) -> ExtractionResult:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe extraction dependencies are unavailable. Install requirements.txt."
            ) from exc

        frames: list[np.ndarray] = []
        valid_flags: list[bool] = []
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            for frame in self._iter_frames(source_path, cv2):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)
                vector, valid = self._frame_vector(result)
                frames.append(vector)
                valid_flags.append(valid)
        finally:
            holistic.close()

        quality = assess_frame_quality(valid_flags, self.minimum_valid_ratio)
        if not frames:
            return ExtractionResult(
                features=np.empty((0, OUTPUT_FEATURE_SIZE), dtype=np.float32),
                quality=quality,
                source_frames=0,
            )

        sequence = np.stack(frames).astype(np.float32)
        sequence = self._append_motion(sequence)
        sequence = self._resample(sequence, self.sequence_length)
        return ExtractionResult(
            features=sequence,
            quality=quality,
            source_frames=len(frames),
        )

    @classmethod
    def _image_paths(cls, directory: Path) -> list[Path]:
        if not directory.exists():
            raise FileNotFoundError(directory)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def _iter_frames(self, source_path: Path, cv2: Any) -> Iterable[np.ndarray]:
        if source_path.is_dir():
            paths = self._image_paths(source_path)
            if not paths:
                raise ValueError(f"image sequence directory is empty: {source_path}")
            for path in paths:
                frame = cv2.imread(str(path))
                if frame is None:
                    raise ValueError(f"unable to read image frame: {path}")
                yield frame
            return

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise ValueError(f"unable to open video or image sequence: {source_path}")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
        finally:
            capture.release()

    def _frame_vector(self, result: Any) -> tuple[np.ndarray, bool]:
        pose = result.pose_landmarks.landmark if result.pose_landmarks else None
        origin_x, origin_y, origin_z, scale = self._normalization_from_pose(pose)

        left, left_present = self._xyz_landmarks(
            result.left_hand_landmarks.landmark if result.left_hand_landmarks else None,
            range(21),
            origin_x,
            origin_y,
            origin_z,
            scale,
        )
        right, right_present = self._xyz_landmarks(
            result.right_hand_landmarks.landmark if result.right_hand_landmarks else None,
            range(21),
            origin_x,
            origin_y,
            origin_z,
            scale,
        )
        pose_values, pose_present = self._pose_landmarks(
            pose, origin_x, origin_y, origin_z, scale
        )
        face_values, face_present = self._xyz_landmarks(
            result.face_landmarks.landmark if result.face_landmarks else None,
            FACE_INDICES,
            origin_x,
            origin_y,
            origin_z,
            scale,
        )

        base = np.concatenate((left, right, pose_values, face_values)).astype(np.float32)
        masks = np.asarray(
            [left_present, right_present, pose_present, face_present], dtype=np.float32
        )
        vector = np.concatenate((base, masks))
        if vector.shape[0] != BASE_FEATURE_SIZE + MASK_SIZE:
            raise RuntimeError(f"unexpected feature size: {vector.shape[0]}")
        valid = pose_present and (left_present or right_present)
        return vector, valid

    @staticmethod
    def _normalization_from_pose(
        pose: Sequence[Any] | None,
    ) -> tuple[float, float, float, float]:
        if pose is None or len(pose) <= 12:
            return 0.5, 0.5, 0.0, 1.0
        left = pose[11]
        right = pose[12]
        origin_x = (left.x + right.x) / 2
        origin_y = (left.y + right.y) / 2
        origin_z = (left.z + right.z) / 2
        scale = ((left.x - right.x) ** 2 + (left.y - right.y) ** 2) ** 0.5
        return origin_x, origin_y, origin_z, max(scale, 1e-6)

    @staticmethod
    def _xyz_landmarks(
        landmarks: Sequence[Any] | None,
        indices: Sequence[int],
        origin_x: float,
        origin_y: float,
        origin_z: float,
        scale: float,
    ) -> tuple[np.ndarray, bool]:
        if landmarks is None:
            return np.zeros(len(indices) * 3, dtype=np.float32), False
        values = []
        for index in indices:
            point = landmarks[index]
            values.extend(
                (
                    (point.x - origin_x) / scale,
                    (point.y - origin_y) / scale,
                    (point.z - origin_z) / scale,
                )
            )
        return np.asarray(values, dtype=np.float32), True

    @staticmethod
    def _pose_landmarks(
        landmarks: Sequence[Any] | None,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        scale: float,
    ) -> tuple[np.ndarray, bool]:
        if landmarks is None:
            return np.zeros(len(POSE_INDICES) * 4, dtype=np.float32), False
        values = []
        for index in POSE_INDICES:
            point = landmarks[index]
            values.extend(
                (
                    (point.x - origin_x) / scale,
                    (point.y - origin_y) / scale,
                    (point.z - origin_z) / scale,
                    point.visibility,
                )
            )
        return np.asarray(values, dtype=np.float32), True

    @staticmethod
    def _append_motion(sequence: np.ndarray) -> np.ndarray:
        base = sequence[:, :BASE_FEATURE_SIZE]
        masks = sequence[:, BASE_FEATURE_SIZE:]
        deltas = np.zeros_like(base)
        deltas[1:] = base[1:] - base[:-1]
        return np.concatenate((base, masks, deltas), axis=1).astype(np.float32)

    @staticmethod
    def _resample(sequence: np.ndarray, target_length: int) -> np.ndarray:
        if target_length <= 0:
            raise ValueError("target_length must be positive")
        if len(sequence) == target_length:
            return sequence
        indices = np.linspace(0, len(sequence) - 1, target_length).round().astype(int)
        return sequence[indices]
