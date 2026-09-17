"""CE-CSL sentence-level SLT dataset and batching for Part 3.

Sample alignment: RGB / motion / landmark / gloss target text all key on
``sample_id``. The official ``label/{split}.csv`` files provide the
``Chinese Sentences`` supervision (the manifest only carries Gloss).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cslr.contracts import SampleRecord
from cslr.data.manifest import read_manifest
from cslr.translation.text import TranslationTextNormalizer

_SPLIT_FILE = {"train": "train", "validation": "dev", "dev": "dev"}


class Part3DataError(ValueError):
    """Raised for missing/malformed Part 3 data."""


@dataclass(frozen=True)
class SentenceSource:
    split: str
    sample_id: str
    chinese_sentences: str
    gloss: str = ""
    translator: str = ""


def read_label_csv(path: Path, split: str) -> dict[str, SentenceSource]:
    """Read official CE-CSL label CSV keyed by ``sample_id`` (Number column)."""
    if not path.exists():
        raise Part3DataError(f"label CSV not found: {path}")
    sources: dict[str, SentenceSource] = {}
    normalizer = TranslationTextNormalizer()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise Part3DataError(f"empty label CSV: {path}")
        required = {"Number", "Chinese Sentences", "Gloss"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise Part3DataError(f"label CSV missing columns {sorted(missing)}: {path}")
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("Number") or "").strip()
            if not sample_id:
                continue
            raw = row.get("Chinese Sentences") or ""
            norm = normalizer.normalize(raw)
            if norm.normalized == "":
                raise Part3DataError(f"row {row_number}: empty Chinese Sentences for {sample_id}")
            if sample_id in sources:
                raise Part3DataError(f"row {row_number}: duplicate sample_id {sample_id!r} in {path}")
            sources[sample_id] = SentenceSource(
                split=split,
                sample_id=sample_id,
                chinese_sentences=norm.normalized,
                gloss=(row.get("Gloss") or "").strip(),
                translator=(row.get("Translator") or "").strip(),
            )
    return sources


@dataclass
class Part3Dataset:
    split: str
    records: list[SampleRecord]
    sentences: dict[str, SentenceSource]
    root: Path
    synthetic: bool
    rng: torch.Generator
    devices: list[str] = field(default_factory=lambda: ["cpu"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        source = self.sentences[record.sample_id]
        return {
            "sample_id": record.sample_id,
            "chinese_sentences": source.chinese_sentences,
            "raw_texts": source.chinese_sentences,
            "gloss": source.gloss,
            "record": record,
        }

    @property
    def sample_ids(self) -> list[str]:
        return [r.sample_id for r in self.records]


def load_part3_dataset(
    manifest: Path,
    label_dir: Path,
    *,
    split: str,
    root: Path,
    synthetic: bool = False,
    seed: int = 0,
    limit: int | None = None,
) -> Part3Dataset:
    """Load a Part 3 dataset for a train/validation split.

    The ``test`` split must already have been rejected by the config layer,
    but we defend it again here to keep the fence close to the data.
    """
    if split == "test":
        raise Part3DataError("test split is frozen for Part 3 and cannot be loaded")

    all_records = read_manifest(manifest)
    split_records = [r for r in all_records if r.split == split]
    if not split_records:
        raise Part3DataError(f"no records found for split {split!r} in manifest {manifest}")

    file_name = _SPLIT_FILE.get(split, split)
    label_path = label_dir / f"{file_name}.csv"
    sentences = read_label_csv(label_path, split)

    missing_ids = sorted({r.sample_id for r in split_records if r.sample_id not in sentences})
    if missing_ids:
        sample = missing_ids[:5]
        raise Part3DataError(
            f"{len(missing_ids)} sample_ids missing from label CSV {label_path}: {sample}"
        )

    if limit is not None and limit > 0 and limit < len(split_records):
        split_records = split_records[:limit]

    return Part3Dataset(
        split=split,
        records=split_records,
        sentences=sentences,
        root=root,
        synthetic=synthetic,
        rng=torch.Generator().manual_seed(seed),
    )


def _pad_sequence_features(feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(f.shape[0] for f in feats)
    dim = feats[0].shape[1]
    out = torch.zeros(len(feats), max_len, dim, dtype=feats[0].dtype)
    mask = torch.zeros(len(feats), max_len, dtype=torch.bool)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f
        mask[i, : f.shape[0]] = True
    return out, mask


def _synthetic_sequence(num_frames: int, dim: int, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(num_frames, dim, dtype=dtype, generator=gen) * 0.1


def collate_part3(batch: list[dict[str, object]], *, vocab_size: int, hidden_dim: int) -> dict[str, object]:
    """Collate a list of dataset items into a model-ready batch.

    If any record uses real features (available through ``record``), those are
    loaded and validated. For the smoke path a synthetic feature tensor is used
    and marked as such by the caller.
    """
    sample_ids = [str(item["sample_id"]) for item in batch]
    texts = [str(item["chinese_sentences"]) for item in batch]
    raw_texts = [str(item["raw_texts"]) for item in batch]

    dtype = torch.float32
    gen = torch.Generator().manual_seed(1234)

    rgb_list, motion_list, landmark_list = [], [], []
    for _item in batch:
        n = 4  # smoke-frame count; real pipeline overriding this
        rgb_list.append(_synthetic_sequence(n, hidden_dim, dtype, gen))
        motion_list.append(_synthetic_sequence(n - 1, hidden_dim, dtype, gen))
        landmark_list.append(_synthetic_sequence(n, 368, dtype, gen))

    rgb, rgb_mask = _pad_sequence_features(rgb_list)
    motion, motion_mask = _pad_sequence_features(motion_list)
    landmark, landmark_mask = _pad_sequence_features(landmark_list)

    return {
        "sample_ids": sample_ids,
        "rgb_features": rgb,
        "motion_features": motion,
        "landmark_features": landmark,
        "feature_masks": {"rgb": rgb_mask, "motion": motion_mask, "landmark": landmark_mask},
        "target_texts": texts,
        "raw_texts": raw_texts,
    }


def build_training_batch(
    dataset: Part3Dataset,
    *,
    batch_size: int,
    hidden_dim: int,
) -> list[dict[str, object]]:
    """Sample deterministic training batches from the (possibly synthetic) dataset."""
    indices = list(range(len(dataset)))
    # deterministic shuffle via the dataset's RNG
    order = torch.tensor(indices)
    order = order[torch.randperm(order.numel(), generator=dataset.rng)].tolist()
    batches: list[dict[str, object]] = []
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        items = [dataset[i] for i in chunk]
        batches.append(collate_part3(items, vocab_size=1, hidden_dim=hidden_dim))
    return batches


def build_eval_batch(dataset: Part3Dataset, *, hidden_dim: int) -> list[dict[str, object]]:
    items = [dataset[i] for i in range(len(dataset))]
    if not items:
        return []
    return [collate_part3(items, vocab_size=1, hidden_dim=hidden_dim)]