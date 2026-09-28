"""LLM noise-robustness window (Route A): can Qwen reorganize corrupted gloss?

Measures how much gloss-recognition noise the local LLM can absorb before the
final translation degrades below the usable BLEU region. Corrupts the *gold*
dev gloss sequences to several noise levels (replace-with-wrong-gloss /
delete / shuffle), feeds them through the exact same route_a LLM call, and
reports BLEU-1/2, ROUGE-L, chrF, distinct per level.

Purpose: decide how accurate the P1 recognizer really must be. If BLEU stays
>0.3 under moderate noise, the LLM is a genuine reorganizer and we only need a
recognizer with decent coverage, not high per-token precision. If it collapses
fast, the LLM cannot rescue the current ~0.69 discriminator's errors, which
strengthens the "replace feature material" conclusion.

Protocol: same prompt/model as route_a (Qwen2.5-1.5B). test split frozen.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = "/home/su127/FYP/domain-bounded-cslr"
if REPO not in sys.path:
    sys.path.insert(0, str(Path(REPO) / "src"))

from scripts.route_a_gloss_llm import PROMPT, build_rows, run_qwen, summarize  # noqa: E402

# (display_key, replace_frac, delete_frac) ; shuffle handled separately
LEVELS: list[tuple[str, float | None, float | None]] = [
    ("clean", None, None),
    ("replace30", 0.30, 0.0),
    ("replace50", 0.50, 0.0),
    ("noisy_realistic", 0.35, 0.15),  # recognizer-ish: swap + drop
    ("shuffle", None, None),
]


def _tokens(gloss: str) -> list[str]:
    return [p for p in gloss.split("/") if p.strip() and p not in ("。", "，")]


def build_repl_pool(rows: list[dict]) -> list[str]:
    pool = set()
    for r in rows:
        pool.update(_tokens(r["gloss"]))
    return sorted(pool)


def corrupt(tokens: list[str], kind: str, replace: float | None, delete: float | None, rng: random.Random, pool: list[str]) -> list[str]:
    if kind == "shuffle":
        out = list(tokens)
        rng.shuffle(out)
        return out
    out = list(tokens)
    if replace and replace > 0 and out:
        n = max(1, round(replace * len(out)))
        for i in rng.sample(range(len(out)), n):
            cand = rng.choice(pool)
            tries = 0
            while cand == out[i] and tries < 6:
                cand = rng.choice(pool)
                tries += 1
            out[i] = cand
    if delete and delete > 0 and out:
        nd = max(1, round(delete * len(out)))
        drop = set(rng.sample(range(len(out)), min(nd, len(out) - 1)))
        out = [t for j, t in enumerate(out) if j not in drop]
    return out or [rng.choice(pool)]


def run(args: argparse.Namespace) -> int:
    all_rows = build_rows(args.label)
    pool = build_repl_pool(all_rows)
    sub = all_rows[: args.limit] if args.limit else all_rows

    results = {}
    for kind, replace, delete in LEVELS:
        if args.noise and kind not in args.noise:
            continue
        rng = random.Random(args.seed)
        rows_k = []
        for r in sub:
            corrupted = corrupt(_tokens(r["gloss"]), kind, replace, delete, rng, pool)
            rows_k.append({**r, "gloss": " ".join(corrupted)})
        records = run_qwen(rows_k, model_path=args.model, device=args.device, limit=len(rows_k), no_cache=args.no_cache)
        summ = summarize(records)
        summ["noise_level"] = kind
        results[kind] = summ
        print(f"[{kind}] bleu1={summ['bleu_1']} distinct={summ['distinct']}", flush=True)

    receipt = {
        "probe": "llm-noise-window",
        "protocol": f"corrupt gold dev gloss -> route_a Qwen2.5-1.5B -> metrics; limit={args.limit}",
        "model": args.model,
        "seed": args.seed,
        "prompt": PROMPT,
        "results": results,
        "test_split_read": False,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, ensure_ascii=False, indent=2)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.probe_llm_robustness")
    parser.add_argument("--label", type=Path, default=Path(REPO) / "data/raw/CE-CSL/label/dev.csv")
    parser.add_argument(
        "--model",
        default="/mnt/d/part3_models/ms_cache/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--noise", nargs="+", help="subset of levels to run")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path(REPO) / "artifacts/metrics/part4-llm-noise-window.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())