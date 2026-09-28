"""Split CTC dev WER by gloss function class.

Section 20 found content glosses (房子/休息/行李/告诉（我）) reach ROC-AUC 0.93-1.0
in every feature family while function words (了/又/好不好/走/决定) sit at 0.02-0.25.
This script re-scores the recognizer's dev predictions on the two subsets to answer:
did the CTC model learn the content glosses it *could* have learned, with the residual
error concentrated on function words (a data-property ceiling), or is content recognition
broken too (a training-problem ceiling)?

Pure-function helpers are unit tested; the model path reuses evaluate_checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from cslr.recognition.inference import evaluate_checkpoint
from cslr.recognition.metrics import edit_distance

REPO = Path(__file__).resolve().parents[1]
UNKNOWN = "<unk>"

# Function-word glosses: grammatical words with no lexical meaning (助词/连词/介词/副词/语气词).
# CE-CSL glosses are segmented words, so 了/的/又/都 appear as standalone gloss tokens.
FUNCTION_WORDS = frozenset(
    {
        "了", "的", "地", "得", "之", "其", "于", "而", "却", "虽", "但", "且",
        "因为", "所以", "如果", "但是", "但是2", "可是", "只要", "只有", "无论",
        "又", "也", "都", "还", "就", "才", "再", "很", "更", "太", "最", "真",
        "挺", "在", "把", "被", "从", "对", "给", "和", "与", "或", "等", "过",
        "着", "吗", "呢", "吧", "啊", "呀", "嘛", "哦", "哎", "嗯", "不", "没",
        "别", "请", "要", "会", "能", "可以", "应该", "必须", "一定", "一起",
        "一下", "刚", "马上", "立刻", "已经", "曾经", "正在", "常常", "经常",
        "一起2", "一起3", "以后", "以前", "然后", "于是", "接着", "一边",
    }
)


def filter_subsequence(tokens: Sequence[str], target: frozenset[str]) -> list[str]:
    """Keep only tokens in ``target`` (and drop the unknown marker)."""

    return [token for token in tokens if token != UNKNOWN and token in target]


def subset_stats(
    references: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str]],
    target: frozenset[str],
) -> dict[str, float | int]:
    """WER and token recall over the subset, counting only samples with non-empty references."""

    total_errors = 0
    total_length = 0
    covered_samples = 0
    reference_count: Counter[str] = Counter()
    matched_count: Counter[str] = Counter()
    for reference, prediction in zip(references, predictions, strict=False):
        ref_sub = filter_subsequence(reference, target)
        if not ref_sub:
            continue
        pred_sub = filter_subsequence(prediction, target)
        total_errors += edit_distance(ref_sub, pred_sub)
        total_length += len(ref_sub)
        covered_samples += 1
        reference_count.update(ref_sub)
        for token in set(ref_sub):
            matched_count[token] += min(ref_sub.count(token), pred_sub.count(token))
    reference_tokens = sum(reference_count.values())
    matched_tokens = sum(matched_count.values())
    return {
        "wer": (total_errors / total_length) if total_length else None,
        "reference_tokens": reference_tokens,
        "matched_tokens": matched_tokens,
        "token_recall": (matched_tokens / reference_tokens) if reference_tokens else None,
        "covered_samples": covered_samples,
        "distinct_tokens": len(reference_count),
    }


def load_probe_buckets(probe_path: Path) -> tuple[frozenset[str], frozenset[str]]:
    """Strong/weak content glosses from the landmark family's top/bottom ROC-AUC.

    The probe receipt stores only the 5 strongest and 5 weakest glosses per family;
    landmark is the strongest family overall (macro 0.6821).
    """

    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    landmark = payload["results"]["landmark"]
    strong = frozenset(entry["gloss"] for entry in landmark["top_auc"])
    weak = frozenset(entry["gloss"] for entry in landmark["bottom_auc"])
    return strong, weak


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(REPO) / "artifacts/checkpoints/ctc-vl48-cap300.pt",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path(REPO) / "data/manifests/ce-csl.csv"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(REPO) / "data/processed/ce-csl-qwenvl48",
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument("--probe", type=Path, help="part4-discrim-probe.json for AUC buckets")
    parser.add_argument(
        "--out", type=Path, default=Path(REPO) / "artifacts/metrics/part4-gloss-split-wer.json"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.split == "test":
        raise SystemExit("split 'test' is frozen")
    payload = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        feature_root=args.features,
        split=args.split,
        output_path=Path(os.devnull),
        beam_width=1,
        device="auto",
    )
    references = [item["reference"] for item in payload["predictions"]]
    predictions = [item["prediction"] for item in payload["predictions"]]

    all_tokens = {token for reference in references for token in reference if token != UNKNOWN}
    content_target = frozenset(all_tokens - FUNCTION_WORDS)
    full_target = frozenset(all_tokens)

    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "split": payload["split"],
        "samples": payload["samples"],
        "test_split_read": False,
        "buckets": {},
    }
    report["buckets"]["all"] = subset_stats(references, predictions, full_target)
    report["buckets"]["content"] = subset_stats(references, predictions, content_target)
    report["buckets"]["function"] = subset_stats(references, predictions, FUNCTION_WORDS)

    if args.probe is not None:
        strong, weak = load_probe_buckets(args.probe)
        strong = frozenset(token for token in strong if token not in FUNCTION_WORDS)
        weak = frozenset(token for token in weak if token not in FUNCTION_WORDS)
        report["buckets"]["probe_strong_content"] = subset_stats(references, predictions, strong)
        report["buckets"]["probe_weak_content"] = subset_stats(references, predictions, weak)
        report["probe_bucket_sizes"] = {"strong": len(strong), "weak": len(weak)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def line(label: str) -> None:
        stats = report["buckets"][label]
        print(
            f"{label:24s} WER={stats['wer']:.4f}  recall={stats['token_recall']:.4f}  "
            f"tokens={stats['reference_tokens']}  samples={stats['covered_samples']}"
        )

    line("all")
    line("content")
    line("function")
    if args.probe is not None:
        line("probe_strong_content")
        line("probe_weak_content")


if __name__ == "__main__":
    main()
