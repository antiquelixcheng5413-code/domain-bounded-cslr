# CTC Sequence Recognition

## Scope

This document defines the continuous CE-CSL sequence-recognition comparison path. It is separate
from the existing LSTM multi-label Gloss-token baseline:

- `legacy_ctc` reproduces the teammate-delivered BiLSTM-CTC artifact for comparison only.
- `ctc_v2` is the formal implementation: official Train/Dev/Test, blank ID `0`, token IDs
  `1..3862`, `log_softmax` before `CTCLoss`, Dev early stopping, and corpus-level WER.

The Web service never treats a CTC path score as a calibrated probability. It only exposes
`legacy_ctc` when `CSLR_ENABLE_LEGACY_CTC=true` is explicitly set for internal reproduction.

## Vocabulary And Artifacts

The versioned vocabulary is
`data/manifests/ce-csl-ctc-vocab.txt`. The project Python runtime reads 3,862 ordered tokens and
its SHA-256 is:

```text
85ea606a8a10cc1611a4e8755e875f09360727c8608ee0bbc14950c541d1abb3
```

It is byte-identical to the vocabulary used by the teammate repository. The delivered models,
checkpoint, external ONNX sidecar, and local landmarks remain ignored under
`data/clean_datas/ctc-delivery-2026-06-21` and `data/clean_datas/ce_csl`.

## Reproduced Legacy Results

The legacy ONNX model was re-evaluated against the official manifest with the decoder corrected
to reset token collapse after a blank. All edit counts are accumulated before calculating WER.

| Split | Samples | Exact sequences | Sequence accuracy | S | D | I | N | Corpus WER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dev | 515 | 137 | 26.60% | 297 | 557 | 2 | 2,842 | 30.12% |
| Test | 500 | 99 | 19.80% | 350 | 733 | 1 | 3,040 | 35.66% |

The detailed files are `artifacts/metrics/legacy-ctc-dev.csv` and
`artifacts/metrics/legacy-ctc-test.csv`; their summary JSON files include the same aggregate
counts. The delivered PyTorch checkpoint also loaded through `LegacyCTCModel` and produced the
same Dev predictions and metrics. This confirms the ONNX reproduction path.

The legacy ONNX SHA-256 is
`d79541d9e4feaf1a471b3c15534c8619c1c1239b1b621c190d35f0ea9d5737a1`.

The teammate's previously stated 24.58% Dev WER and 26.56% Test WER are not used as reproduced
metrics because they do not match the detailed prediction files or this consistent corpus-level
recalculation.

## Commands

Reproduce the legacy ONNX evaluation:

```powershell
docker compose --profile dev run --rm dev evaluate-ctc `
  --manifest data/manifests/ce-csl.csv `
  --features data/clean_datas/ce_csl `
  --model data/clean_datas/ctc-delivery-2026-06-21/ctc_model.onnx `
  --model-kind legacy_ctc `
  --vocab data/manifests/ce-csl-ctc-vocab.txt `
  --split validation `
  --output artifacts/metrics/legacy-ctc-dev.csv
```

Train the formal CTC v2 model (not yet run as of this document):

```powershell
docker compose --profile dev run --rm dev train-ctc `
  --manifest data/manifests/ce-csl.csv `
  --features data/clean_datas/ce_csl `
  --vocab data/manifests/ce-csl-ctc-vocab.txt `
  --model-config configs/models/ctc_lstm.yaml `
  --training-config configs/training_ctc.yaml `
  --feature-receipt artifacts/metrics/ce-csl-features-receipt.json `
  --output artifacts/checkpoints/ctc_v2.pt

docker compose --profile dev run --rm dev export-ctc `
  artifacts/checkpoints/ctc_v2.pt `
  artifacts/exports/ctc_v2.onnx
```

Evaluate Dev during model selection, then evaluate Test once after selecting the Dev checkpoint:

```powershell
docker compose --profile dev run --rm dev evaluate-ctc `
  --manifest data/manifests/ce-csl.csv `
  --features data/clean_datas/ce_csl `
  --model artifacts/checkpoints/ctc_v2.pt `
  --model-kind ctc_v2 `
  --split validation `
  --output artifacts/metrics/ctc-v2-dev.csv
```

## Web Configuration

The normal Web default remains the existing `multilabel` baseline. After a formal CTC v2 export,
set the following environment values before `docker compose up --build app`:

```text
CSLR_MODEL_KIND=ctc_v2
CSLR_MODEL_PATH=/workspace/artifacts/exports/ctc_v2.onnx
CSLR_SEMANTIC_DATA_ROOT=/workspace/data/ce-csl
```

The result displays ordered Gloss tokens, an uncalibrated CTC path score, model version, latency,
and an exact-reference Chinese reconstruction state. A Chinese sentence is returned only when the
predicted complete Gloss sequence has a unique exact CE-CSL reference; otherwise the result states
that no reliable Chinese reconstruction is available. No hospital intent mapping is used.
