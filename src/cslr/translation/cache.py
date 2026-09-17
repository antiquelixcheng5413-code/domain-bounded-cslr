"""Offline feature caching and provenance receipts for Part 3.

Each cached feature file is paired with a SHA-256 digest receipt documenting
model name, version, frame sampling, shape, dtype and hash. Caches are keyed
one-per-``sample_id`` and must never overwrite existing landmark features.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_feature(
    root: Path,
    sample_id: str,
    modality: str,
    array: np.ndarray,
    *,
    model_name: str,
    version: str,
    frame_sampling: str,
    allow_overwrite: bool = False,
) -> Path:
    """Save a feature .npy plus a .receipt.json provenance sidecar.

    Refuses to clobber an existing landmark feature unless explicit.
    """
    root.mkdir(parents=True, exist_ok=True)
    feat_path = root / f"{sample_id}.{modality}.npy"
    if feat_path.exists() and not allow_overwrite:
        if modality == "landmark":
            raise FileExistsError(
                f"refusing to overwrite existing landmark feature {feat_path}"
            )
    np.save(feat_path, array.astype(np.float32))

    receipt = {
        "sample_id": sample_id,
        "modality": modality,
        "model_name": model_name,
        "version": version,
        "frame_sampling": frame_sampling,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256_file(feat_path),
    }
    receipt_path = root / f"{sample_id}.{modality}.receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return feat_path


def load_feature(
    root: Path,
    sample_id: str,
    modality: str,
    *,
    verify_hash: bool = True,
) -> np.ndarray:
    feat_path = root / f"{sample_id}.{modality}.npy"
    if not feat_path.exists():
        raise FileNotFoundError(f"feature not cached: {feat_path}")
    array = np.load(feat_path)
    if verify_hash:
        receipt = root / f"{sample_id}.{modality}.receipt.json"
        if receipt.exists():
            meta = json.loads(receipt.read_text(encoding="utf-8"))
            if meta.get("sha256") != sha256_file(feat_path):
                raise ValueError(f"SHA-256 mismatch for {feat_path}")
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite values in {feat_path}")
    return array


def write_smoke_receipt(
    report_dir: Path,
    *,
    status: str,
    split: str,
    synthetic: bool,
    formal_result: bool,
    test_split_read: bool,
    uses_external_weights: bool,
    git_commit: str,
    config: dict,
    extra: dict | None = None,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "split": split,
        "synthetic_features": synthetic,
        "formal_result": formal_result,
        "test_split_read": test_split_read,
        "uses_external_weights": uses_external_weights,
        "git_commit": git_commit,
        "config": config,
        **(extra or {}),
    }
    path = report_dir / "part3-smoke.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_per_sample_csv(report_dir: Path, rows: list[dict]) -> Path:
    import csv

    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "part3-smoke-per-sample.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path