"""One-shot Phase-2 extraction that prints progress per sample to stay active.

The background runner SIGTERMs jobs that produce no output for a while. This
script emits a line per sample (flush=True) so the WSL job stays 'busy', and
it reuses one CLIP process instead of spawning a subprocess per window.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cslr.data.manifest import read_manifest
from cslr.translation.extract import RealFeatureExtractor

ROOT = Path(".").resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/ce-csl.csv")
    parser.add_argument("--video-root", default="data/raw/CE-CSL")
    parser.add_argument("--landmark-root", default="data/processed/ce_csl")
    parser.add_argument("--cache-root", default="artifacts/part3_features")
    parser.add_argument("--split", required=True, choices=["train", "validation", "dev"])
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--window", type=int, default=100000)
    parser.add_argument("--max", type=int, default=0, help="max uncached samples to process this run (0=all)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder", default="ViT-B-32")
    args = parser.parse_args(argv)

    norm_split = "validation" if args.split in ("dev", "validation") else args.split
    records = [r for r in read_manifest(ROOT / args.manifest) if r.split == norm_split]
    split_cache = ROOT / args.cache_root / norm_split
    split_cache.mkdir(parents=True, exist_ok=True)

    def cached(sid: str) -> bool:
        return all((split_cache / f"{sid}.{m}.npy").exists() for m in ("rgb", "motion", "landmark"))

    window_records = records[args.start_idx : args.start_idx + args.window]
    to_do = [r for r in window_records if not cached(r.sample_id)]
    if args.max and args.max > 0:
        to_do = to_do[: args.max]
    print(f"[oneshot] split={norm_split} window_start={args.start_idx} "
          f"in_window={len(window_records)} uncached={len(to_do)}", flush=True)

    extractor = RealFeatureExtractor(
        landmark_root=ROOT / args.landmark_root,
        cache_root=ROOT / args.cache_root,
        split=norm_split,
        encoder_name=args.encoder,
        device=args.device,
    )
    ok = fail = skip = 0
    t0 = time.time()
    for rec in to_do:
        try:
            saved = extractor.cache(rec.sample_id, ROOT / args.video_root / rec.video)
            ok += 1
            status = f"ok frames={sum(saved.values()):.0f}" if isinstance(saved, dict) else "ok"
            print(f"[{rec.sample_id}] {status} {time.time()-t0:.1f}s", flush=True)
        except Exception as exc:  # per-sample tolerance; resume next run
            fail += 1
            print(f"[{rec.sample_id}] FAIL: {type(exc).__name__}: {exc}", flush=True)
    print(f"[oneshot] done ok={ok} fail={fail} skip={skip} secs={time.time()-t0:.0f}", flush=True)
    print("ONESHOT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())