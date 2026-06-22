from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cslr.contracts import SampleRecord


def split_ctc_tokens(gloss: str) -> list[str]:
    """Keep ordered Gloss tokens, including terminal punctuation used by legacy CTC labels."""
    return [token.strip() for token in gloss.replace("\u3000", " ").split("/") if token.strip()]


@dataclass(frozen=True)
class CTCVocabulary:
    tokens: list[str]
    blank_index: int

    def __post_init__(self) -> None:
        if self.blank_index < 0:
            raise ValueError("blank_index must not be negative")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("CTC vocabulary contains duplicate tokens")
        if self.blank_index >= self.class_count:
            raise ValueError("blank_index must be within the CTC output class range")

    @property
    def token_to_index(self) -> dict[str, int]:
        return {token: index + 1 for index, token in enumerate(self.tokens)}

    @property
    def index_to_token(self) -> dict[int, str]:
        return {index + 1: token for index, token in enumerate(self.tokens)}

    @property
    def class_count(self) -> int:
        return len(self.tokens) + 1

    @classmethod
    def from_file(cls, path: Path, blank_index: int) -> CTCVocabulary:
        tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return cls(tokens=[token for token in tokens if token], blank_index=blank_index)


@dataclass(frozen=True)
class CTCSample:
    features: np.ndarray
    targets: np.ndarray
    reference_tokens: list[str]
    input_length: int
    target_length: int
    sample_id: str


class CTCDataset:
    """Load fixed-length landmark features with ordered CE-CSL Gloss targets."""

    def __init__(
        self,
        records: list[SampleRecord],
        feature_root: Path,
        vocabulary: CTCVocabulary,
    ) -> None:
        self.records = records
        self.feature_root = feature_root
        self.vocabulary = vocabulary
        self._token_to_index = vocabulary.token_to_index
        self.oov_records = [
            record.sample_id
            for record in records
            if any(token not in self._token_to_index for token in split_ctc_tokens(record.label))
        ]
        if self.oov_records:
            raise ValueError(
                "CTC vocabulary is missing Gloss tokens for "
                f"{len(self.oov_records)} records; first: {self.oov_records[0]}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> CTCSample:
        record = self.records[index]
        path = self.feature_root / f"{record.sample_id}.npy"
        features = np.load(path, allow_pickle=False).astype(np.float32)
        if features.ndim != 2:
            raise ValueError(f"{record.sample_id}: expected 2D features, got {features.shape}")
        if features.shape[1] != 368:
            raise ValueError(f"{record.sample_id}: expected 368 features, got {features.shape[1]}")
        tokens = split_ctc_tokens(record.label)
        targets = np.asarray([self._token_to_index[token] for token in tokens], dtype=np.int64)
        valid_frames = int(np.count_nonzero(np.any(features != 0.0, axis=1)))
        input_length = valid_frames if valid_frames else int(features.shape[0])
        if len(targets) > input_length:
            raise ValueError(
                f"{record.sample_id}: target length {len(targets)} "
                f"exceeds input length {input_length}"
            )
        return CTCSample(
            features=features,
            targets=targets,
            reference_tokens=tokens,
            input_length=input_length,
            target_length=len(targets),
            sample_id=record.sample_id,
        )


def collate_ctc_samples(samples: list[CTCSample]) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot collate an empty CTC batch")
    feature_shape = samples[0].features.shape
    if any(sample.features.shape != feature_shape for sample in samples):
        raise ValueError("CTC batch contains inconsistent feature shapes")
    return {
        "features": np.stack([sample.features for sample in samples]).astype(np.float32),
        "targets": np.concatenate([sample.targets for sample in samples]).astype(np.int64),
        "input_lengths": np.asarray([sample.input_length for sample in samples], dtype=np.int64),
        "target_lengths": np.asarray([sample.target_length for sample in samples], dtype=np.int64),
        "sample_ids": [sample.sample_id for sample in samples],
        "reference_tokens": [sample.reference_tokens for sample in samples],
    }
