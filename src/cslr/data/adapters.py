from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cslr.contracts import SampleRecord
from cslr.data.manifest import read_manifest


class DatasetAdapter(ABC):
    """Convert a dataset-specific index into the shared SampleRecord contract."""

    @classmethod
    def from_config(
        cls, config: dict[str, Any], data_root_override: Path | None = None
    ) -> DatasetAdapter:
        raise NotImplementedError(f"{cls.__name__} does not support config construction")

    @abstractmethod
    def records(self) -> Iterable[SampleRecord]:
        raise NotImplementedError


class CsvManifestAdapter(DatasetAdapter):
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path

    @classmethod
    def from_config(
        cls, config: dict[str, Any], data_root_override: Path | None = None
    ) -> CsvManifestAdapter:
        manifest_path = Path(config["manifest"])
        if data_root_override is not None and not manifest_path.is_absolute():
            manifest_path = data_root_override / manifest_path
        return cls(manifest_path=manifest_path)

    def records(self) -> Iterable[SampleRecord]:
        return read_manifest(self.manifest_path)


class NationalCSLDPImageSequenceAdapter(DatasetAdapter):
    """Scan NationalCSL-DP image-frame directories into the shared manifest contract."""

    def __init__(
        self,
        data_root: Path,
        intent_labels: dict[str, list[str]],
        split_by_signer: dict[str, str],
        views: Iterable[str] = ("front",),
    ) -> None:
        self.data_root = data_root
        self.intent_labels = intent_labels
        self.split_by_signer = split_by_signer
        self.views = tuple(views)

    @classmethod
    def from_config(
        cls, config: dict[str, Any], data_root_override: Path | None = None
    ) -> NationalCSLDPImageSequenceAdapter:
        labels: dict[str, list[str]] = {}
        for label, details in config.get("intent_labels", {}).items():
            ids = [str(value).zfill(4) for value in details.get("ids", [])]
            if ids:
                labels[label] = ids

        split_by_signer = {}
        for split, signers in config.get("proposed_signer_split", {}).items():
            for signer in signers:
                split_by_signer[str(signer)] = split

        views = config.get("adapter_options", {}).get("views", ["front"])
        return cls(
            data_root=data_root_override or Path(config["data_root"]),
            intent_labels=labels,
            split_by_signer=split_by_signer,
            views=views,
        )

    def records(self) -> Iterable[SampleRecord]:
        for signer in sorted(self.split_by_signer):
            signer_dir = self.data_root / signer
            if not signer_dir.exists():
                continue
            split = self.split_by_signer[signer]
            for view in self.views:
                for label, ids in sorted(self.intent_labels.items()):
                    for gloss_id in ids:
                        sequence_dir = signer_dir / view / gloss_id
                        if not sequence_dir.is_dir():
                            continue
                        if not any(sequence_dir.glob("*.jpg")):
                            continue
                        yield SampleRecord(
                            sample_id=f"{signer}_{view}_{gloss_id}",
                            video=Path(signer) / view / gloss_id,
                            label=label,
                            signer=signer,
                            session=view,
                            split=split,
                        )


CE_CSL_DEFAULT_SPLIT_FILES = {
    "train": "label/train.csv",
    "validation": "label/dev.csv",
    "test": "label/test.csv",
}

CE_CSL_DEFAULT_VIDEO_SPLITS = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}


class CECSLVideoAdapter(DatasetAdapter):
    """Read CE-CSL label CSV files and map each row to its MP4 video."""

    def __init__(
        self,
        data_root: Path,
        split_files: dict[str, str],
        video_splits: dict[str, str],
        video_dir: str = "video",
        sample_id_column: str = "Number",
        signer_column: str = "Translator",
        label_column: str = "Chinese Sentences",
        require_videos: bool = True,
    ) -> None:
        self.data_root = data_root
        self.split_files = split_files
        self.video_splits = video_splits
        self.video_dir = video_dir
        self.sample_id_column = sample_id_column
        self.signer_column = signer_column
        self.label_column = label_column
        self.require_videos = require_videos

    @classmethod
    def from_config(
        cls, config: dict[str, Any], data_root_override: Path | None = None
    ) -> CECSLVideoAdapter:
        options = config.get("adapter_options", {})
        return cls(
            data_root=data_root_override or Path(config["data_root"]),
            split_files=dict(options.get("split_files", CE_CSL_DEFAULT_SPLIT_FILES)),
            video_splits=dict(options.get("video_splits", CE_CSL_DEFAULT_VIDEO_SPLITS)),
            video_dir=options.get("video_dir", "video"),
            sample_id_column=options.get("sample_id_column", "Number"),
            signer_column=options.get("signer_column", "Translator"),
            label_column=options.get("label_column", "Chinese Sentences"),
            require_videos=bool(options.get("require_videos", True)),
        )

    def records(self) -> Iterable[SampleRecord]:
        for split, label_file in self.split_files.items():
            video_split = self.video_splits.get(split, split)
            label_path = self.data_root / label_file
            if not label_path.exists():
                raise FileNotFoundError(label_path)

            with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self._validate_columns(reader.fieldnames or [], label_path)
                for row_number, row in enumerate(reader, start=2):
                    sample_id = (row[self.sample_id_column] or "").strip()
                    signer = (row[self.signer_column] or "").strip()
                    label = (row[self.label_column] or "").strip()
                    if not any((sample_id, signer, label)):
                        continue
                    video = Path(self.video_dir) / video_split / signer / f"{sample_id}.mp4"
                    if self.require_videos and not (self.data_root / video).is_file():
                        raise FileNotFoundError(
                            f"{label_path}:{row_number}: missing video {video.as_posix()}"
                        )
                    yield SampleRecord(
                        sample_id=sample_id,
                        video=video,
                        label=label,
                        signer=signer,
                        session=video_split,
                        split=split,
                    )

    def _validate_columns(self, columns: list[str], label_path: Path) -> None:
        required = [self.sample_id_column, self.signer_column, self.label_column]
        missing = [column for column in required if column not in columns]
        if missing:
            raise ValueError(f"{label_path}: missing columns: {', '.join(missing)}")


_ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "csv_manifest": CsvManifestAdapter,
    "ce_csl_video": CECSLVideoAdapter,
    "nationalcsl_dp_image_sequence": NationalCSLDPImageSequenceAdapter,
}


def register_adapter(name: str, adapter: type[DatasetAdapter]) -> None:
    if not name.strip():
        raise ValueError("adapter name must not be empty")
    if name in _ADAPTERS:
        raise ValueError(f"adapter already registered: {name}")
    _ADAPTERS[name] = adapter


def get_adapter(name: str) -> type[DatasetAdapter]:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown adapter {name!r}; available: {sorted(_ADAPTERS)}") from exc


def available_adapters() -> Iterable[str]:
    return sorted(_ADAPTERS)
