from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cslr.contracts import SampleRecord


@dataclass(frozen=True)
class ExtractionItem:
    sample_id: str
    split: str
    signer: str
    source: str
    output: str
    status: str
    accepted: bool
    total_frames: int
    valid_frames: int
    valid_ratio: float
    warnings: str
    error: str = ""


@dataclass(frozen=True)
class ExtractionBatchSummary:
    total: int
    extracted: int
    skipped: int
    failed: int
    accepted: int
    rejected: int
    items: list[ExtractionItem]

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "extracted": self.extracted,
            "skipped": self.skipped,
            "failed": self.failed,
            "accepted": self.accepted,
            "rejected": self.rejected,
        }


def extract_manifest_features(
    records: list[SampleRecord],
    data_root: Path,
    output_root: Path,
    extractor: Any,
    *,
    limit: int | None = None,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> ExtractionBatchSummary:
    selected = records[:limit] if limit is not None else records
    output_root.mkdir(parents=True, exist_ok=True)

    items: list[ExtractionItem] = []
    extracted = 0
    skipped = 0
    failed = 0
    accepted = 0
    rejected = 0

    for record in selected:
        source = data_root / record.video
        output = _safe_feature_path(output_root, record.sample_id)
        if output.exists() and not overwrite:
            skipped += 1
            items.append(
                ExtractionItem(
                    sample_id=record.sample_id,
                    split=record.split,
                    signer=record.signer,
                    source=source.as_posix(),
                    output=output.as_posix(),
                    status="skipped",
                    accepted=True,
                    total_frames=0,
                    valid_frames=0,
                    valid_ratio=0.0,
                    warnings="existing output",
                )
            )
            continue

        try:
            result = extractor.extract(source)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, result.features)
        except Exception as exc:
            failed += 1
            items.append(
                ExtractionItem(
                    sample_id=record.sample_id,
                    split=record.split,
                    signer=record.signer,
                    source=source.as_posix(),
                    output=output.as_posix(),
                    status="failed",
                    accepted=False,
                    total_frames=0,
                    valid_frames=0,
                    valid_ratio=0.0,
                    warnings="",
                    error=str(exc),
                )
            )
            if not continue_on_error:
                raise
            continue

        extracted += 1
        if result.quality.accepted:
            accepted += 1
        else:
            rejected += 1
        items.append(
            ExtractionItem(
                sample_id=record.sample_id,
                split=record.split,
                signer=record.signer,
                source=source.as_posix(),
                output=output.as_posix(),
                status="extracted",
                accepted=result.quality.accepted,
                total_frames=result.quality.total_frames,
                valid_frames=result.quality.valid_frames,
                valid_ratio=result.quality.valid_ratio,
                warnings="; ".join(result.quality.warnings),
            )
        )

    return ExtractionBatchSummary(
        total=len(selected),
        extracted=extracted,
        skipped=skipped,
        failed=failed,
        accepted=accepted,
        rejected=rejected,
        items=items,
    )


def write_extraction_report(path: Path, items: list[ExtractionItem]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "signer",
        "source",
        "output",
        "status",
        "accepted",
        "total_frames",
        "valid_frames",
        "valid_ratio",
        "warnings",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(item.__dict__)
    return len(items)


def _safe_feature_path(output_root: Path, sample_id: str) -> Path:
    output_root_resolved = output_root.resolve()
    output = (output_root / f"{sample_id}.npy").resolve()
    if output_root_resolved != output and output_root_resolved not in output.parents:
        raise ValueError(f"sample_id escapes output root: {sample_id}")
    return output
