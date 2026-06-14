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
- `participants/`: any identity or consent-related material.

Each manifest must contain:

```text
sample_id,video,label,signer,session,split
```

`video` is a path relative to the configured dataset root. Never place names, student IDs,
contact details, or consent records in a manifest.
