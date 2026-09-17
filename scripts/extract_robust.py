"""Robust Phase-2 feature extraction for the remaining uncached samples.

The naive extractor dies on a corrupted video because OpenCV can raise a
segfault that Python cannot catch, killing the whole process. This runner
isolates each mini-batch in a subprocess; a crash only drops that window, and
leftover uncached samples are retried one at a time. Idempotent: cached
samples are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

from cslr.data.manifest import read_manifest

ROOT = Path(".").resolve()
BATCH = 6
BATCH_TIMEOUT = 60
PER_SAMPLE_TIMEOUT = 120


def _is_cached(split_cache: Path, sample_id: str) -> bool:
    return all((split_cache / f"{sample_id}.{mod}.npy").exists() for mod in ("rgb", "motion", "landmark"))


def _run(args: argparse.Namespace, start: int, limit: int, timeout: int) -> tuple[int | str, str]:
    cmd = [
        sys.executable, "-m", "cslr.translation", "extract",
        "--manifest", str(ROOT / args.manifest),
        "--video-root", str(ROOT / args.video_root),
        "--landmark-root", str(ROOT / args.landmark_root),
        "--cache-root", str(ROOT / args.cache_root),
        "--split", args.split,
        "--start", str(start),
        "--limit", str(limit),
        "--device", args.device,
        "--encoder", args.encoder,
    ]
    # Run the worker detached from this process's stdout inactivity: a worker that
    # decodes a corrupted video can hang. We pump its output on a thread and emit a
    # heartbeat so the job is never seen as idle, and hard-timeout hanging workers.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail: list[str] = []

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            tail.append(line.rstrip())

    threading.Thread(target=_pump, daemon=True).start()
    t0 = time.time()
    while proc.poll() is None:
        if time.time() - t0 > timeout:
            proc.kill()
            proc.wait()
            tail.append(f"[robust] worker TIMEOUT after {int(timeout)}s")
            return "timeout", "\n".join(tail[-10:])
        print(f"  [w{start} run={time.time()-t0:.0f}s l={len(tail)}]", flush=True)
        time.sleep(3)
    rc = proc.returncode
    print(f"  [worker start={start} done rc={rc} {time.time()-t0:.1f}s]", flush=True)
    return rc, "\n".join(tail[-10:])


def _single(args: argparse.Namespace, idx: int, split_cache: Path, rec) -> None:
    global FAILURES
    try:
        code, tail = _run(args, idx, 1, PER_SAMPLE_TIMEOUT)
    except Exception as exc:
        FAILURES.append((rec.sample_id, "err", repr(exc)))
        print(f"  retry FAIL {rec.sample_id} err {exc}", flush=True)
        return
    if _is_cached(split_cache, rec.sample_id):
        print(f"  retry ok {rec.sample_id} rc={code}", flush=True)
    else:
        FAILURES.append((rec.sample_id, code, tail.strip()[-200:]))
        print(f"  retry FAIL {rec.sample_id} rc={code}", flush=True)


FAILURES: list[tuple[str, int | str, str]] = []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="robust real feature extraction")
    parser.add_argument("--manifest", default="data/manifests/ce-csl.csv")
    parser.add_argument("--video-root", default="data/raw/CE-CSL")
    parser.add_argument("--landmark-root", default="data/processed/ce_csl")
    parser.add_argument("--cache-root", default="artifacts/part3_features")
    parser.add_argument("--split", required=True, choices=["train", "validation", "dev"])
    parser.add_argument("--max", type=int, default=0, help="max uncached samples to process (0=all)")
    parser.add_argument("--skip", default="", help="comma-separated manifest indices to skip (hang videos)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder", default="ViT-B-32")
    args = parser.parse_args(argv)

    norm_split = "validation" if args.split in ("dev", "validation") else args.split
    records = [r for r in read_manifest(ROOT / args.manifest) if r.split == norm_split]
    split_cache = ROOT / args.cache_root / norm_split
    split_cache.mkdir(parents=True, exist_ok=True)

    uncached = [i for i, r in enumerate(records) if not _is_cached(split_cache, r.sample_id)]
    skip = {int(x) for x in args.skip.split(",") if x.strip()}
    if skip:
        uncached = [i for i in uncached if i not in skip]
        print(f"[robust] skipped indices={sorted(skip)}", flush=True)
    if args.max and args.max > 0:
        uncached = uncached[: args.max]
    print(f"[robust] split={norm_split} total={len(records)} uncached={len(uncached)}", flush=True)
    t0 = time.time()

    for start in range(0, len(uncached), BATCH):
        window = uncached[start : start + BATCH]
        win_start, win_end = window[0], window[-1]
        code, tail = _run(args, win_start, win_end - win_start + 1, BATCH_TIMEOUT)
        now_cached = sum(1 for r in records[win_start : win_end + 1] if _is_cached(split_cache, r.sample_id))
        done = now_cached == (win_end - win_start + 1)
        print(f"[{records[win_start].sample_id}..{records[win_end].sample_id}] "
              f"rc={code} done={done} cached={now_cached}/{len(window)} "
              f"{time.time()-t0:.0f}s", flush=True)
        if not done:
            # per-sample retry on the leftover (timeout-isolated per video)
            rec_indices = [i for i in range(win_start, win_end + 1) if not _is_cached(split_cache, records[i].sample_id)]
            for i in rec_indices:
                _single(args, i, split_cache, records[i])

    print(f"[robust] failures={FAILURES}", flush=True)
    print(f"[robust] cached_total={sum(1 for r in records if _is_cached(split_cache, r.sample_id))}", flush=True)
    print("EXTRACT_ROBUST_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())