"""Route C: CTC-predicted gloss -> Chinese sentence via a local LLM (non-oracle).

Route A translates the *gold* gloss sequence and is therefore the oracle upper bound
(BLEU-1 0.671). Route C swaps in the gloss sequence produced by the Part 4 CTC recognizer
so we measure the real end-to-end chain end to end:

    video features  ->  CTC gloss recognizer  ->  predicted gloss  ->  LLM  ->  中文句子

It reuses Route A's prompt, LLM calling and metric/distinct audit verbatim so the result is
directly comparable against the oracle number.

The frozen ``test`` split is never read here: ``evaluate_checkpoint`` raises before opening
any file when ``--split test`` is requested.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = "/home/su127/FYP/domain-bounded-cslr"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "src"))
    sys.path.insert(0, os.path.join(REPO, "scripts"))

from cslr.recognition.inference import evaluate_checkpoint  # noqa: E402
from route_a_gloss_llm import PROMPT, build_rows, run_qwen, summarize  # noqa: E402


def predicted_gloss_rows(checkpoint: Path, manifest: Path, features: Path, split: str, limit: int | None) -> dict[str, str]:
    """Run the CTC recognizer over the split and return {sample_id: predicted_gloss_string}."""

    payload = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        feature_root=features,
        split=split,
        output_path=Path(os.devnull),
        beam_width=1,
        limit=limit,
        device="auto",
    )
    rows: dict[str, str] = {}
    for item in payload["predictions"]:
        tokens = [t for t in item["prediction"] if t != "<unk>"]
        rows[item["sample_id"]] = "/".join(tokens)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Part 4 self-describing CTC checkpoint")
    ap.add_argument("--manifest", default=str(Path(REPO) / "data/manifests/ce-csl.csv"))
    ap.add_argument("--features", required=True, help="feature root passed to evaluate_checkpoint")
    ap.add_argument("--label", required=True, help="CE-CSL label CSV (for reference Chinese)")
    ap.add_argument("--split", default="dev", choices=["train", "dev"])
    ap.add_argument("--model", required=True, help="local LLM HF dir (same as Route A)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preds = predicted_gloss_rows(
        Path(args.checkpoint), Path(args.manifest), Path(args.features), args.split, args.limit
    )

    gold_rows = build_rows(args.label)
    rows = []
    for r in gold_rows:
        sid = r["sample_id"]
        if sid not in preds:
            continue
        rows.append({"sample_id": sid, "gloss": preds[sid], "reference": r["reference"]})
    if not rows:
        raise SystemExit("no overlapping samples between CTC predictions and label CSV")

    records = run_qwen(rows, model_path=args.model, device=args.device, limit=args.limit)
    summary = summarize(records)
    summary["uses_external_weights"] = True
    summary["gloss_source"] = "ctc_predicted"
    summary["checkpoint"] = str(args.checkpoint)
    summary["test_split_read"] = False
    summary["prompt"] = PROMPT

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "per_sample": records}, fh, ensure_ascii=False, indent=2)

    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=== TOP PREDICTIONS ===")
    for r in records[:12]:
        print(f"  {r['sample_id']} | gloss={r['gloss']!r} | ref={r['reference']!r} -> {r['prediction']!r}")


if __name__ == "__main__":
    main()