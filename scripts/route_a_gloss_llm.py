"""Route A1: gloss-sequence -> Chinese-sentence translation via a local LLM.

Reads CE-CSL label CSVs (split=train/dev), builds a prompt that asks the LLM
to translate a Chinese sign-language gloss sequence into a fluent Chinese
sentence, runs generation, and emits the same metrics + distinct-count audit
used everywhere else so results are directly comparable and not misread.

This is the fastest-possible check of "does handing tokens to a strong LLM
escape the repetition collapse we hit with tiny/mT5 decoders".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

# Make the repo importable.
REPO = "/home/su127/FYP/domain-bounded-cslr"
if REPO not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

from cslr.translation.metrics import bleu, chrf, rouge_l  # noqa: E402

PROMPT = (
    "你是一位中文手语翻译。请把下面的手语手势记号序列(Gloss)翻译成一个通顺、完整、"
    "符合中文语法的句子。只输出翻译结果本身，不要解释、不要加引号。\n"
    "Gloss: {gloss}\n中文翻译："
)


def _gloss_to_input(raw: str) -> str:
    # gloss stored as "10/年/鱼/禁止1/区/时间/长/不/。"
    parts = [p for p in raw.split("/") if p.strip()]
    # drop trailing punctuation, keep tokens
    parts = [p for p in parts if p != "。" and p != "，"]
    return " ".join(parts)


def build_rows(label_csv: str) -> list[dict]:
    rows = []
    with open(label_csv, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            sid = (r.get("Number") or "").strip()
            gloss = (r.get("Gloss") or "").strip()
            chinese = (r.get("Chinese Sentences") or "").strip()
            if not sid or not gloss or not chinese:
                continue
            rows.append({"sample_id": sid, "gloss": gloss, "reference": chinese})
    return rows


def run_qwen(rows: list[dict], *, model_path: str, device: str, limit: int, no_cache: bool = False) -> list[dict]:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype="auto"
    ).to(device).eval()

    out: list[dict] = []
    for i, row in enumerate(rows[:limit]):
        prompt = PROMPT.format(gloss=_gloss_to_input(row["gloss"]))
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=not no_cache,
            pad_token_id=tok.eos_token_id,
        )
        pred = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = pred.strip()
        ref = row["reference"]
        out.append(
            {
                "sample_id": row["sample_id"],
                "gloss": row["gloss"],
                "reference": ref,
                "prediction": pred,
                "bleu_1": round(bleu(ref, pred, 1), 4),
                "bleu_2": round(bleu(ref, pred, 2), 4),
                "rouge_l": round(rouge_l(ref, pred), 4),
                "chrf": round(chrf(ref, pred), 4),
                "exact_match": float(ref == pred),
            }
        )
        if (i + 1) % 20 == 0 or i + 1 == min(limit, len(rows)):
            print(f"[A1] {i+1}/{min(limit,len(rows))} done", flush=True)
    return out


def summarize(records: list[dict]) -> dict:
    n = max(1, len(records))
    avg = lambda k: round(sum(float(r[k]) for r in records) / n, 4)
    preds = [str(r["prediction"]) for r in records]
    cnt = Counter(preds)
    distinct = len(cnt)
    top_text, top_n = cnt.most_common(1)[0]
    return {
        "samples": len(records),
        "distinct": distinct,
        "collapsed": distinct <= 2,
        "top_prediction_share": round(top_n / n, 3),
        "top_prediction": top_text,
        "bleu_1": avg("bleu_1"),
        "bleu_2": avg("bleu_2"),
        "rouge_l": avg("rouge_l"),
        "chrf": avg("chrf"),
        "exact_match": avg("exact_match"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True, help="local HF dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = build_rows(args.label)
    records = run_qwen(rows, model_path=args.model, device=args.device, limit=args.limit, no_cache=args.no_cache)
    summary = summarize(records)
    summary["uses_external_weights"] = True
    summary["test_split_read"] = False
    summary["prompt"] = PROMPT

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "per_sample": records}, fh, ensure_ascii=False, indent=2)

    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=== TOP PREDICTIONS ===")
    for r in records[:12]:
        print(f"  {r['sample_id']} | ref={r['reference']!r} -> {r['prediction']!r}")


if __name__ == "__main__":
    main()