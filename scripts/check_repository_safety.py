from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".tflite", ".mp4", ".mov", ".avi", ".webm"}
FORBIDDEN_PARTS = {
    ("data", "raw"),
    ("data", "interim"),
    ("data", "processed"),
    ("data", "participants"),
    ("artifacts", "checkpoints"),
    ("artifacts", "exports"),
    ("artifacts", "logs"),
    ("docs", "ethics", "consent-forms"),
}
SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def tracked_files() -> Iterable[Path]:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    for value in result.stdout.decode("utf-8").split("\0"):
        if value:
            yield ROOT / value


def is_gitkeep(path: Path) -> bool:
    return path.name == ".gitkeep"


def main() -> int:
    failures: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        lower_parts = tuple(part.lower() for part in relative.parts)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden tracked file type: {relative}")
        for forbidden in FORBIDDEN_PARTS:
            if lower_parts[: len(forbidden)] == forbidden and not is_gitkeep(path):
                failures.append(f"forbidden tracked location: {relative}")
        if path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
            failures.append(f"tracked file exceeds 10 MiB: {relative}")
        if path.is_file() and path.suffix.lower() not in {".docx", ".png", ".jpg", ".jpeg", ".pdf"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                failures.append(f"possible secret in: {relative}")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("repository safety check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
