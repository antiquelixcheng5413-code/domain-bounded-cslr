# Public Dataset Candidate Scorecard

Last verified: 2026-06-17.

Scoring: 0 = unsuitable, 1 = weak, 2 = acceptable, 3 = strong. Weighted maximum is 105. Scores
marked with `*` still require access or archive verification.

| Criterion | Weight | NationalCSL-DP | CSL-Daily | NMFs-CSL |
|---|---:|---:|---:|---:|
| Legal academic access and usable license | 5 | 3 | 1* | 2* |
| Signer IDs available | 5 | 3 | 3 | 1* |
| Data obtainable within Week 2 | 5 | 2* | 1* | 1* |
| Labels suitable for bounded baseline | 4 | 3 | 2 | 3 |
| Dataset-bounded semantic coverage | 3 | 3 | 3 | 1* |
| Hand, upper-body, and face visibility | 4 | 3 | 3 | 3 |
| Sufficient signer diversity | 5 | 2 | 2 | 1* |
| Existing benchmarks and documentation | 2 | 2 | 3 | 3 |
| Storage and preprocessing cost | 2 | 1 | 1* | 1* |
| **Weighted total** |  | **89** | **72** | **60** |

## Evidence Summary

### NationalCSL-DP

- Official paper: <https://www.nature.com/articles/s41597-025-04986-x>
- Official data: <https://figshare.com/articles/media/NationalCSL-DP/27261843>
- DOI: `10.6084/m9.figshare.27261843.v3`
- License: CC BY 4.0.
- Scale: 6,707 isolated signs, 134,140 videos, 10 signers, front and left views.
- Useful for isolated-sign benchmarking and historical comparison, but not the selected final
  dataset.

### CSL-Daily

- Paper: <https://arxiv.org/abs/2105.12397>
- Access page referenced by the authors:
  <http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/>
- Scale: 20,654 continuous videos, 10 signers, 2,000 gloss vocabulary.
- It is highly relevant to continuous recognition and translation, but access is request-based and
  the license was not verified within the project schedule.

### NMFs-CSL

- Official page: <https://ustc-slr.github.io/datasets/2020_nmfs_csl/>
- Scale: 1,067 isolated Chinese sign words recorded at 30 fps.
- Research use requires a release agreement signed by a full-time staff member; a student
  signature is explicitly not accepted.
- Signer metadata, archive size, semantic coverage, and expected approval time remain unverified.

## Hard Gates

A primary dataset must:

1. Be legally obtainable for this academic project.
2. Include signer identity or an official signer-independent split.
3. Provide usable labels and visual samples.
4. Be obtainable by the end of Week 2.

If the highest-scoring dataset fails an access gate, move immediately to the next candidate. Do not
delay implementation while waiting for an uncertain download approval.

## Decision Record

Final selected dataset as of 2026-06-17: CE-CSL.

- Local archive: `E:\Download\CE-CSL.zip`.
- Observed representation: MP4 videos plus official `train`, `dev`, and `test` CSV labels.
- Intended task: CE-CSL dataset-bounded Gloss token recognition with sentence references.
- NationalCSL-DP is retained as a historical candidate and audit record, but it is no longer the
  primary training/evaluation dataset.
