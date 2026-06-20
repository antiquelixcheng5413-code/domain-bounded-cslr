# CE-CSL Feature Artifact Protocol

## Scope

Git tracks source code, manifests, split definitions, configuration, tests, and small validation
receipts. It does not track raw videos, landmark arrays, checkpoints, ONNX files, extraction logs,
or backup archives.

Git LFS is not the default solution. Thousands of generated arrays still make clones slow and
consume hosted storage and bandwidth. DVC can be introduced later only after the team chooses a
durable shared storage remote and its access rules.

## Local Storage

The active CE-CSL feature bundle is stored outside version control at:

```text
E:\college\FYP\data\clean_datas\ce_csl\
```

The original delivery archive remains at:

```text
E:\college\FYP\data\clean_datas\ce_csl.zip
```

The ZIP is a transfer backup. The extracted directory is used for validation and training. Neither
location may be staged with Git.

## Required Delivery Metadata

For every feature bundle, the extraction owner provides:

1. One `<sample_id>.npy` file per manifest record.
2. A per-sample extraction report containing status, valid-frame ratio, and errors.
3. Extractor commit ID, `configs/features.yaml`, command, Docker image version, and dates.
4. A receipt generated with `verify-features --sha256`.

For every trained model, the model owner provides:

1. Checkpoint/ONNX file names and SHA-256 values.
2. Training commit ID, model config, training config, seed, and feature-receipt digest.
3. Metrics on the official CE-CSL train/dev/test split.
4. Vocabulary/output metadata matching the exported model.

## Acceptance

Run the following command before any experiment:

```powershell
docker compose run --rm dev verify-features data/manifests/ce-csl.csv `
  --features data/clean_datas/ce_csl `
  --sha256 `
  --receipt artifacts/metrics/ce-csl-features-receipt.json
```

Accept a bundle only when the command exits with code `0`, reports 5,988 valid records, and has no
missing, invalid-shape, invalid-dtype, or unreadable files. Preserve the receipt digest in every
training record. A changed feature extractor or source manifest requires a new bundle version,
not silent replacement of the existing data.

## Accepted Bundle

`ce_csl` was accepted on 2026-06-21 against `data/manifests/ce-csl.csv`: 5,988 expected and valid
records, no missing/invalid/unreadable files, shape `[48, 368]`, dtype `float32`, SHA-256 tree
digest `7c240ffa3d5493c51f0c019ad8b1d068bdc40645507d6b6ee68803a5ea634386`. The versioned receipt is
`artifacts/metrics/ce-csl-features-receipt.json`.
