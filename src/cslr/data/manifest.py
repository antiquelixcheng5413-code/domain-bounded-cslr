from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from cslr.contracts import SampleRecord

REQUIRED_COLUMNS = ("sample_id", "video", "label", "signer", "session", "split")


def read_manifest(path: Path) -> list[SampleRecord]:
    if not path.exists():
        raise FileNotFoundError(path)

    records: list[SampleRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            record = SampleRecord(
                sample_id=(row["sample_id"] or "").strip(),
                video=Path((row["video"] or "").strip()),
                label=(row["label"] or "").strip(),
                signer=(row["signer"] or "").strip(),
                session=(row["session"] or "").strip(),
                split=(row["split"] or "unassigned").strip().lower(),
            )
            try:
                record.validate()
            except ValueError as exc:
                raise ValueError(f"row {row_number}: {exc}") from exc
            records.append(record)
    return records


def validate_manifest(records: Iterable[SampleRecord]) -> None:
    seen = set()
    for record in records:
        record.validate()
        if record.sample_id in seen:
            raise ValueError(f"duplicate sample_id: {record.sample_id}")
        seen.add(record.sample_id)


def write_manifest(path: Path, records: Iterable[SampleRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for record in records:
            record.validate()
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "video": record.video.as_posix(),
                    "label": record.label,
                    "signer": record.signer,
                    "session": record.session,
                    "split": record.split,
                }
            )
            count += 1
    return count
