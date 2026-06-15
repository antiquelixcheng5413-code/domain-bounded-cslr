from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cslr.contracts import SampleRecord
from cslr.data.manifest import read_manifest


class DatasetAdapter(ABC):
    """Convert a dataset-specific index into the shared SampleRecord contract."""

    @abstractmethod
    def records(self) -> Iterable[SampleRecord]:
        raise NotImplementedError


class CsvManifestAdapter(DatasetAdapter):
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path

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
    ) -> "NationalCSLDPImageSequenceAdapter":
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


_ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "csv_manifest": CsvManifestAdapter,
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
