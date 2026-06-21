# Teammate CTC Error Analysis

Source: `artifacts/metrics/teammate-ctc-eval-detailed.csv` (1,015 official Dev/Test predictions).

## Verified Summary

| Split | Samples | Exact matches | Sequence accuracy | Whitespace-token WER |
|---|---:|---:|---:|---:|
| Dev | 515 | 137 | 26.60% | 30.12% |
| Test | 500 | 99 | 19.80% | 35.66% |

The exact-match counts reproduce the teammate-provided sequence accuracies. The WER values above
use standard Levenshtein distance after splitting the CSV's `reference` and `hypothesis` fields on
spaces. They do not match the reported 24.58% Dev and 26.56% Test WER, so the original evaluator
is required before the lower WER values can be treated as reproduced.

## Representative Errors

| Type | Sample | Reference | Prediction | Interpretation |
|---|---|---|---|---|
| Truncation | `dev-00004` | `上班2 时间 固定 是 。` | `上班2 时间 。` | The model drops the final semantic tokens. |
| Substitution | `dev-00001` | `10 年 鱼 禁止1 区 时间 长 不 。` | `世界 ？` | The sequence is unrelated to the reference. |
| Mixed substitution and deletion | `dev-00002` | `2 0 2 3 年 高 考试1 学生 一 千 多 万 。` | `2 0 2 3 五 高 千 。` | `年` is replaced by `五` and several tokens are omitted. This is not a pure insertion under standard edit-distance alignment. |
| Exact match | `dev-00009` | `下 次 大 雨 在 重庆 。` | `下 次 大 雨 在 重庆 。` | The complete sequence is correct. |

## Use in the Report

Use this table as qualitative CTC error analysis. Do not generalize it into an open-domain Chinese
translation claim: it evaluates only the CE-CSL dataset-bounded Gloss label space.
