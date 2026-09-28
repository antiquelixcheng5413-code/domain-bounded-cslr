"""Build an upload package of CE-CSL videos (train+dev only) for Kaggle.

Purpose: the §22 bottleneck (0.69 macro-AUC ceiling) can only be broken by
re-extracting features from raw pixels with a stronger vision tower. That work
runs on Kaggle (T4 16GB), so the videos must travel. This script assembles a
zip-ready folder under ``--out``:

    out/manifest.csv          # slim manifest: sample_id,video,label,split
    out/video/...             # copied videos, keeping the relative paths intact
    out/package.zip          # created when --zip is set

The frozen test split is never included (only records whose split is
train/dev are copied), honouring the standing freeze rule.

Usage (from repo root):

    python scripts/pack_kaggle_data.py --data-root data/raw/CE-CSL --out /tmp/kaggle_pkg --zip
    python scripts/pack_kaggle_data.py --data-root ... --out ... --limit 60   # small trial
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if REPO not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cslr.data.manifest import read_manifest  # noqa: E402


def pack(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    train = [r for r in records if r.split == "train"]
    dev = [r for r in records if r.split == "validation"]
    selected = train + dev
    if not selected:
        raise SystemExit("no train/dev records found")

    data_root = Path(args.data_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    video_out = out_root / "video"

    copied = skipped = missing = 0
    manifest_rows: list[dict[str, str]] = []
    limit_train = args.limit_train if args.limit_train else len(train)
    limit_dev = args.limit_dev if args.limit_dev else len(dev)

    def _emit(records: list, cap: int) -> None:
        nonlocal copied, skipped, missing
        for record in records[:cap]:
            src = data_root / record.video
            dst = video_out / record.video
            if not src.exists():
                missing += 1
                continue
            if dst.exists():
                skipped += 1
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
            manifest_rows.append(
                {"sample_id": record.sample_id, "video": record.video, "label": record.label, "split": record.split}
            )

    _emit(train, limit_train)
    _emit(dev, limit_dev)

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample_id", "video", "label", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        json_dumps(
            {
                "copied": copied,
                "already_present": skipped,
                "missing_source_videos": missing,
                "manifest_rows": len(manifest_rows),
                "train_rows": sum(1 for r in manifest_rows if r["split"] == "train"),
                "dev_rows": sum(1 for r in manifest_rows if r["split"] == "validation"),
                "out": str(out_root),
            }
        )
    )

    if args.zip:
        zip_path = out_root.with_suffix(".zip")
        write_zip(out_root, zip_path)
    return 0


def write_zip(src: Path, zip_path: Path) -> None:
    zip_path = Path(zip_path) if str(zip_path).endswith(".zip") else Path(str(zip_path) + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(src.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(src))
    print(json_dumps({"zip": str(zip_path)}))


def json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python scripts/pack_kaggle_data.py")
    parser.add_argument("--manifest", type=Path, default=REPO / "data/manifests/ce-csl.csv")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="CE-CSL root that the manifest 'video' paths are relative to")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-dev", type=int, default=0)
    parser.add_argument("--zip", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return pack(args)


if __name__ == "__main__":
    raise SystemExit(main())