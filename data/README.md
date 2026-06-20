# Data directory

Public datasets are the primary source for training and evaluation. This repository does not
redistribute them.

## Versioned content

- `manifests/`: anonymous CSV indexes.
- `splits/`: fixed random and signer-independent partitions.
- This README and dataset acquisition notes.

## Local-only content

- `raw/`: original videos.
- `interim/`: decoded frames and temporary extraction output.
- `processed/`: landmark arrays.
- `clean_datas/`: received, cleaned feature bundles and their backup archives.
- `participants/`: any identity or consent-related material.

## Feature Bundle Handoff

Landmarks are external research artifacts, not Git content. A complete CE-CSL bundle contains one
`float32` `<sample_id>.npy` file for every record in the versioned manifest, each with shape
`48 x 368`.

Validate a received bundle before training:

```powershell
docker compose run --rm dev verify-features data/manifests/ce-csl.csv `
  --features data/clean_datas/ce_csl `
  --sha256 `
  --receipt artifacts/metrics/ce-csl-features-receipt.json
```

The generated receipt records coverage, invalid files, split counts, and a deterministic SHA-256
tree digest. The full delivery rules are in
[`docs/datasets/feature-artifact-protocol.md`](../docs/datasets/feature-artifact-protocol.md).

Each manifest must contain:

```text
sample_id,video,label,signer,session,split
```

`video` is a path relative to the configured dataset root. Never place names, student IDs,
contact details, or consent records in a manifest.
