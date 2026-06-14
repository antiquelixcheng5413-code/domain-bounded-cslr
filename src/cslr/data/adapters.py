from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

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


_ADAPTERS: dict[str, type[DatasetAdapter]] = {"csv_manifest": CsvManifestAdapter}


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
