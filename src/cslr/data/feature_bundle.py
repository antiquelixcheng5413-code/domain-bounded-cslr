from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from cslr.contracts import SampleRecord


def verify_feature_bundle(
    records: Iterable[SampleRecord],
    feature_root: Path,
    expected_shape: tuple[int, int] = (48, 368),
    include_sha256: bool = False,
) -> dict[str, object]:
    """Validate a local landmark bundle against the versioned manifest."""
    record_list = list(records)
    missing: list[str] = []
    invalid_shape: list[dict[str, object]] = []
    invalid_dtype: list[dict[str, str]] = []
    unreadable: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    tree_hash = hashlib.sha256() if include_sha256 else None
    valid_count = 0

    for record in record_list:
        feature_path = feature_root / f"{record.sample_id}.npy"
        if not feature_path.is_file():
            missing.append(record.sample_id)
            continue
        try:
            feature = np.load(feature_path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            unreadable.append({"sample_id": record.sample_id, "error": str(exc)})
            continue
        if tuple(feature.shape) != expected_shape:
            invalid_shape.append({"sample_id": record.sample_id, "shape": list(feature.shape)})
            continue
        if feature.dtype != np.dtype("float32"):
            invalid_dtype.append({"sample_id": record.sample_id, "dtype": str(feature.dtype)})
            continue

        valid_count += 1
        split_counts[record.split] += 1
        if tree_hash is not None:
            file_hash = hashlib.sha256(feature_path.read_bytes()).hexdigest()
            tree_hash.update(f"{record.sample_id}:{file_hash}\n".encode())

    status = "ok" if not (missing or invalid_shape or invalid_dtype or unreadable) else "failed"
    return {
        "status": status,
        "feature_root": str(feature_root),
        "expected_records": len(record_list),
        "valid_records": valid_count,
        "expected_shape": list(expected_shape),
        "expected_dtype": "float32",
        "split_counts": dict(sorted(split_counts.items())),
        "missing_count": len(missing),
        "invalid_shape_count": len(invalid_shape),
        "invalid_dtype_count": len(invalid_dtype),
        "unreadable_count": len(unreadable),
        "missing_sample_ids": missing,
        "invalid_shapes": invalid_shape,
        "invalid_dtypes": invalid_dtype,
        "unreadable_files": unreadable,
        "integrity": {
            "algorithm": "sha256-tree" if tree_hash is not None else "not-computed",
            "digest": tree_hash.hexdigest() if tree_hash is not None else None,
        },
    }
