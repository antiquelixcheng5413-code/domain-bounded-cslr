"""Dataset, collate, and CTC target construction for ordered gloss recognition."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cslr.contracts import SampleRecord
from cslr.data.manifest import read_manifest, validate_manifest
from cslr.recognition.gloss_sequence import GlossVocabulary, build_ordered_vocabulary


@dataclass(frozen=True)
class SequenceSample:
    sample_id: str
    features: np.ndarray  # [T, D]
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]


# Layout of the cached 368-dim CE-CSL landmark vector (see cslr.features.extractor):
#   [0:126]    left+right hand 21x3
#   [126:158]  pose 8x4
#   [158:182]  face 8x3
#   [182:186]  presence masks (left, right, pose, face)
#   [186:312]  deltas of [0:126]
#   [312:368]  deltas of [126:182]
FEATURE_BLOCKS: dict[str, slice] = {
    "hands": slice(0, 126),
    "pose": slice(126, 158),
    "face": slice(158, 182),
    "masks": slice(182, 186),
    "hand_deltas": slice(186, 312),
    "body_deltas": slice(312, 368),
}


def feature_view_indices(view: str) -> list[int] | None:
    """Column indices for a named feature view over the cached 368-dim vector.

    The diagnostics in the Part 4 handoff show the full 368-dim vector is close to
    uninformative about which gloss is being signed, partly because the hand block is only
    ~34% of the input. These views let the CTC head see a hand-dominated input without
    re-extracting features.
    """

    if view == "full":
        return None
    parts = [part.strip() for part in view.split("+") if part.strip()]
    if not parts:
        raise ValueError("feature view must name at least one block")
    indices: list[int] = []
    for part in parts:
        if part not in FEATURE_BLOCKS:
            raise ValueError(f"unknown feature block {part!r}; available: {sorted(FEATURE_BLOCKS)}")
        span = FEATURE_BLOCKS[part]
        indices.extend(range(span.start, span.stop))
    return indices


def load_feature(path: Path) -> np.ndarray:
    """Load one cached feature array and validate it."""

    if not path.exists():
        raise FileNotFoundError(f"feature file is missing: {path}")
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"{path}: expected a 2D array, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{path}: feature sequence is empty")
    features = array.astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: features contain non-finite values")
    return features


@dataclass(frozen=True)
class FeatureNormalizer:
    """Per-dimension standardisation of cached features.

    This matters more than it looks. The 368-dim vectors mix normalised coordinates (order 1)
    with frame-to-frame deltas (order 0.1) and binary presence masks, so an unstandardised input
    makes the CTC model collapse onto a single repeated token (measured: 1-2 distinct predictions
    for 8 samples). Standardising with train-split statistics removes that failure mode
    (measured: 8 distinct predictions for the same 8 samples, train loss 64 -> 1.7).
    """

    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, features: Iterable[np.ndarray]) -> FeatureNormalizer:
        arrays = [np.asarray(item, dtype=np.float64) for item in features]
        if not arrays:
            raise ValueError("cannot fit a normalizer without features")
        flattened = np.concatenate([item.reshape(-1, item.shape[-1]) for item in arrays], axis=0)
        mean = flattened.mean(axis=0)
        std = flattened.std(axis=0)
        # keep every dimension usable even if it is constant in the train split
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(float).tolist(), std=std.astype(float).tolist())

    def apply(self, features: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if features.shape[-1] != mean.shape[0]:
            raise ValueError(
                f"feature width {features.shape[-1]} does not match normalizer width {mean.shape[0]}"
            )
        return ((features - mean) / std).astype(np.float32)

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureNormalizer:
        return cls(mean=[float(v) for v in payload["mean"]], std=[float(v) for v in payload["std"]])


class GlossSequenceDataset:
    """Read ordered-gloss samples for one split from a manifest and feature root."""

    def __init__(
        self,
        records: list[SampleRecord],
        feature_root: Path,
        vocabulary: GlossVocabulary,
        normalizer: FeatureNormalizer | None = None,
        feature_view: str = "full",
    ) -> None:
        if not records:
            raise ValueError("no records for this split")
        self.records = list(records)
        self.feature_root = feature_root
        self.vocabulary = vocabulary
        self.normalizer = normalizer
        self.feature_view = feature_view
        self.columns = feature_view_indices(feature_view)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SequenceSample:
        record = self.records[index]
        features = load_feature(self.feature_root / f"{record.sample_id}.npy")
        if self.columns is not None:
            features = features[:, self.columns]
        if self.normalizer is not None:
            features = self.normalizer.apply(features)
        token_ids = tuple(self.vocabulary.encode(record.label))
        tokens = tuple(self.vocabulary.decode(token_ids))
        return SequenceSample(
            sample_id=record.sample_id,
            features=features,
            tokens=tokens,
            token_ids=token_ids,
        )

    def labels(self) -> list[str]:
        return [record.label for record in self.records]


def split_records(records: list[SampleRecord], split: str) -> list[SampleRecord]:
    return [record for record in records if record.split == split]


def filter_present(
    records: list[SampleRecord], feature_root: Path
) -> tuple[list[SampleRecord], list[str]]:
    """Keep only records whose cached feature file exists.

    Extraction runs for hours, so training iterations need to work on whatever is already on
    disk. Returns the kept records and the ids that were dropped, so a caller can report the
    coverage instead of silently training on a subset.
    """

    kept: list[SampleRecord] = []
    missing: list[str] = []
    for record in records:
        if (feature_root / f"{record.sample_id}.npy").exists():
            kept.append(record)
        else:
            missing.append(record.sample_id)
    return kept, missing


def collate_samples(samples: list[SequenceSample]) -> dict[str, Any]:
    """Right-pad features to the batch maximum and keep the true lengths.

    Padding never contributes to the CTC loss or to decoding: ``input_lengths``
    carries the real frame count for every sample, and ``target_lengths`` the
    real token count. A sample whose target is longer than its input is legal
    for CTC (the loss simply cannot align it), so it is reported instead of
    silently dropped.
    """

    if not samples:
        raise ValueError("cannot collate an empty batch")
    lengths = [sample.features.shape[0] for sample in samples]
    dimensions = {sample.features.shape[1] for sample in samples}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent feature dimensions in batch: {sorted(dimensions)}")
    max_length = max(lengths)
    batch_size = len(samples)
    width = dimensions.pop()
    features = np.zeros((batch_size, max_length, width), dtype=np.float32)
    mask = np.zeros((batch_size, max_length), dtype=bool)
    for row, sample in enumerate(samples):
        length = sample.features.shape[0]
        features[row, :length] = sample.features
        mask[row, :length] = True

    target_lengths = [len(sample.token_ids) for sample in samples]
    flattened: list[int] = []
    for sample in samples:
        flattened.extend(sample.token_ids)

    return {
        "sample_ids": [sample.sample_id for sample in samples],
        "features": features,
        "mask": mask,
        "input_lengths": lengths,
        "targets": flattened,
        "target_lengths": target_lengths,
        "tokens": [list(sample.tokens) for sample in samples],
    }


def build_vocabulary_from_records(
    records: list[SampleRecord],
    min_frequency: int = 2,
    max_tokens: int | None = None,
    config: GlossSequenceConfig | None = None,
) -> tuple[GlossVocabulary, dict[str, int]]:
    """Build the ordered vocabulary from the train split only."""

    train_records = split_records(records, "train")
    if not train_records:
        raise ValueError("manifest contains no train records")
    return build_ordered_vocabulary(
        [record.label for record in train_records],
        min_frequency=min_frequency,
        max_tokens=max_tokens,
        config=config,
    )


def load_records(manifest_path: Path) -> list[SampleRecord]:
    records = read_manifest(manifest_path)
    validate_manifest(records)
    return records


def describe_split(records: list[SampleRecord], feature_root: Path) -> dict[str, Any]:
    """Cheap integrity report: how many feature files exist, and sizes."""

    present = 0
    missing: list[str] = []
    dimensions: set[int] = set()
    lengths: list[int] = []
    for record in records:
        path = feature_root / f"{record.sample_id}.npy"
        if not path.exists():
            missing.append(record.sample_id)
            continue
        present += 1
        array = np.load(path, mmap_mode="r")
        if array.ndim == 2:
            dimensions.add(int(array.shape[1]))
            lengths.append(int(array.shape[0]))
    return {
        "records": len(records),
        "features_present": present,
        "features_missing": len(missing),
        "missing_examples": missing[:10],
        "feature_dimensions": sorted(dimensions),
        "sequence_lengths": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
        },
    }


def build_normalizer(
    records: list[SampleRecord],
    feature_root: Path,
    sample_limit: int | None = 600,
    feature_view: str = "full",
) -> FeatureNormalizer:
    """Fit the feature normalizer from the train split (never from validation)."""

    selected = records[:sample_limit] if sample_limit else records
    if not selected:
        raise ValueError("no records available to fit the normalizer")
    columns = feature_view_indices(feature_view)
    features = []
    for record in selected:
        array = load_feature(feature_root / f"{record.sample_id}.npy")
        features.append(array if columns is None else array[:, columns])
    return FeatureNormalizer.fit(features)


def vocabulary_to_json(vocabulary: GlossVocabulary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokens": list(vocabulary.tokens),
        "counts": dict(vocabulary.counts),
        "config": vocabulary.config.as_dict(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def vocabulary_from_json(path: Path) -> GlossVocabulary:
    return GlossVocabulary.from_json(json.loads(path.read_text(encoding="utf-8")))
