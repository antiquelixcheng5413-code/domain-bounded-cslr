# Teammate Handoff Record: CE-CSL Feature and CTC Work

Date: 2026-06-21

## Handoff Sources

| Source | Reference | Status |
|---|---|---|
| Teammate repository | `su1274167870-svg/domain-bounded-cslr`, `main` at `52b025b` | Reviewed as source code and process record |
| Feature delivery | `data/clean_datas/ce_csl.zip` | Received and retained locally |
| Extracted feature bundle | `data/clean_datas/ce_csl/` | Accepted after validation |
| CTC artifact delivery | `data/clean_datas/ctc-delivery-2026-06-21/` | Received, hashes checked, ONNX loaded |

The received ZIP SHA-256 is
`d0715eb8304cf86d97d664f35ab69f442a5acbd3f33e2c9968e6b954a9d6bec7`.
The extracted feature bundle SHA-256 tree digest is
`7c240ffa3d5493c51f0c019ad8b1d068bdc40645507d6b6ee68803a5ea634386`.

## Accepted Deliverables

1. CE-CSL landmark features: 5,988 `.npy` files.
2. Feature contract verified against `data/manifests/ce-csl.csv`:
   - 5,988 expected and valid records.
   - No missing, unreadable, invalid-shape, or invalid-dtype files.
   - Each array is `48 x 368`, `float32`.
3. Teammate implementation record, including the following files in the source repository:
   - `src/cslr/models/ctc.py`
   - `src/cslr/data/ctc_dataset.py`
   - `src/cslr/inference/ctc_service.py`
   - `scripts/train_ctc.py`, `scripts/export_ctc_onnx.py`, and `scripts/e2e_predict_ctc.py`
   - alignment scripts, a CTC vocabulary, and CE-CSL manifests.
4. The teammate repository also documents a TCN intent-classification implementation. It is retained
   as historical work, not imported into the current CE-CSL dataset-bounded main experiment.
5. Final CTC artifact bundle, retained outside Git:
   - `ctc_model_final.pt`, SHA-256
     `7a5f32d03cae509666aa1fa62addfecf2d554bcbef05306c1a3ee69d6a514747`.
   - `ctc_model.onnx` and required `ctc_model.onnx.data`, both hash-checked.
   - `vocab.txt`, SHA-256
     `85ea606a8a10cc1611a4e8755e875f09360727c8608ee0bbc14950c541d1abb3`.
   - `feature_extraction_report.csv`: 5,988 rows, all `success`.

The ONNX file was loaded in the project Docker environment. It accepts `features` with shape
`[batch, 48, 368]` and returns `logits` with shape `[batch, 48, 3863]`.

## Current Integration Decision

The current project keeps the accepted landmark bundle and the teammate repository as a read-only
implementation reference. Its CTC implementation is not merged directly into the active branch.
The active main experiment remains CE-CSL Gloss-token recognition with official train/dev/test
splits and constrained Chinese semantic reconstruction.

The teammate CTC code can become a later sequence-recognition comparison after the following are
addressed:

1. Use the official CE-CSL train/dev/test partition, rather than a new random 80/20 split.
2. Apply `log_softmax` before `CTCLoss` and reserve a non-conflicting CTC blank index.
3. Correct CTC decoding to reset repetition state after blank tokens.
4. Replace hard-coded confidence values with output-derived confidence and report sequence metrics.
5. Remove hospital-specific output as the default result path; CE-CSL labels define the current
   bounded domain.

## Evidence Still Required From the Teammate

The GitHub repository contains training scripts and a written training process; the final CTC
artifact bundle was delivered separately and is now retained locally. The teammate supplied the
following CE-CSL official-split evaluation summary:

| Split | Samples | Sequence accuracy | WER |
|---|---:|---:|---:|
| Train | 4,973 | Not reported | Not reported |
| Dev | 515 | 26.60% | 24.58% |
| Test | 500 | 19.80% | 26.56% |

Reported average CPU inference latency is approximately 12.5 ms. The split counts sum to 5,988
and match the accepted CE-CSL manifest. These values are recorded as teammate-provided results;
they are not yet independently reproduced in the current repository.

The following evidence is still required before the values can be treated as fully auditable
project results.

1. A quality report with valid-frame ratio and failure reason. The delivered report records only
   `sample_id`, `status`, and `size_bytes`; all rows are successful but it has no quality ratio.
2. Machine-readable predictions or evaluation results for the official CE-CSL train/dev/test
   split, including token-level F1, sequence outputs, failure counts, and latency per split.
3. The exact source manifest and split file used for the recorded CTC metrics.
4. Full command logs tying the delivered final checkpoint to the cited commit, configuration, seed,
   and Docker image.

These are evidence requirements, not a statement that the recorded training process is invalid.
The GitHub scripts document how the work was intended to run; the missing artifacts are required
to reproduce and report the claimed model result.

## Local Baseline Status

The project independently trained the current LSTM Gloss-token baseline from the accepted bundle.
It produced local-only `artifacts/checkpoints/lstm.pt` and stopped after 9 epochs. Its validation
micro-F1 and macro-F1 were both `0.0`, so it is a diagnostic run only and must not be presented as
a project result. The next model task is to investigate the label/threshold imbalance before ONNX
export or web deployment.
