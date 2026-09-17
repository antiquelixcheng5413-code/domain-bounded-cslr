"""Real-feature collation for Phase 2 (video -> CLIP RGB / motion / landmark).

Reads per-sample features cached by :mod:`cslr.translation.extract` and packs
them into model-ready batches, keeping the same output schema as the synthetic
``collate_part3`` so the rest of the pipeline is shared.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MODALITIES = ("rgb", "motion", "landmark")


def load_real_features(feature_root: Path, sample_id: str) -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}
    for mod in MODALITIES:
        path = feature_root / f"{sample_id}.{mod}.npy"
        if not path.exists():
            raise FileNotFoundError(f"missing cached feature: {path}")
        loaded[mod] = np.load(path).astype(np.float32)
    return loaded


def _pad(feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(f.shape[0] for f in feats)
    dim = feats[0].shape[1]
    out = torch.zeros(len(feats), max_len, dim, dtype=feats[0].dtype)
    mask = torch.zeros(len(feats), max_len, dtype=torch.bool)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f
        mask[i, : f.shape[0]] = True
    return out, mask


def collate_real_features(batch: list[dict[str, object]], feature_root: Path) -> dict[str, object]:
    sample_ids = [str(item["sample_id"]) for item in batch]
    texts = [str(item["chinese_sentences"]) for item in batch]
    raw_texts = [str(item["raw_texts"]) for item in batch]

    sources = [load_real_features(feature_root, sid) for sid in sample_ids]
    rgb_list = [torch.from_numpy(s["rgb"]) for s in sources]
    motion_list = [torch.from_numpy(s["motion"]) for s in sources]
    landmark_list = [torch.from_numpy(s["landmark"]) for s in sources]

    rgb, rgb_mask = _pad(rgb_list)
    motion, motion_mask = _pad(motion_list)
    landmark, landmark_mask = _pad(landmark_list)

    return {
        "sample_ids": sample_ids,
        "rgb_features": rgb,
        "motion_features": motion,
        "landmark_features": landmark,
        "feature_masks": {"rgb": rgb_mask, "motion": motion_mask, "landmark": landmark_mask},
        "target_texts": texts,
        "raw_texts": raw_texts,
    }


def build_real_eval_batch(dataset, feature_root: Path) -> list[dict[str, object]]:
    items = [dataset[i] for i in range(len(dataset))]
    if not items:
        return []
    return [collate_real_features(items, feature_root)]


def build_real_training_batches(
    dataset, feature_root: Path, *, batch_size: int
) -> list[dict[str, object]]:
    indices = list(range(len(dataset)))
    batches: list[dict[str, object]] = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        items = [dataset[i] for i in chunk]
        batches.append(collate_real_features(items, feature_root))
    return batches