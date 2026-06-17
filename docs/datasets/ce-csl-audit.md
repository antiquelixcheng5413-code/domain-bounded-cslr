# CE-CSL Final Dataset Audit

Last verified: 2026-06-17.

## Lock Status

CE-CSL is the final training and evaluation dataset for the main experiment.

This replaces NationalCSL-DP as the selected dataset. NationalCSL-DP remains a historical audit
record and possible future comparison source, but it is not the current primary experiment data.

## Local Archive

| Field | Value |
|---|---|
| Archive | `E:\Download\CE-CSL.zip` |
| Size | 9,838,930,301 bytes |
| SHA256 | `9E87B76D3D3A5A6AC24195445A302431A0F508BDA3C129B74592853731C4AC41` |
| Last write time | 2026-06-17 00:21:45 |
| Git status | Do not commit archive or extracted videos |

The source archive remains at `E:\Download\CE-CSL.zip`.

## Local Extraction

The working copy is:

```text
E:\college\FYP\data\ce-csl
```

This directory is ignored by Git through `data/ce-csl/**`. It is kept inside the project only so
Docker Compose can use the default `./data -> /workspace/data` mount.

Observed extracted contents:

| Item | Count |
|---|---:|
| MP4 videos | 5,988 |
| Label CSV files | 3 |
| Total files | 5,991 |
| Extracted bytes | 9,846,721,378 |

## Structure and Labels

```text
label/
  train.csv
  dev.csv
  test.csv
video/
  train/<Translator>/<Number>.mp4
  dev/<Translator>/<Number>.mp4
  test/<Translator>/<Number>.mp4
```

Label CSV columns:

```csv
Number,Translator,Chinese Sentences,Gloss,Note
```

Example decoded row:

```csv
train-00001,A,2023年高考到了。,2/0/2/3/高/考/时间/到/。,
```

## Versioned Project Files

| File | Purpose |
|---|---|
| `data/manifests/ce-csl.csv` | Manifest generated from official split CSVs using `Gloss` as `label` |
| `data/manifests/ce-csl-gloss-vocab.csv` | Train-split token-frequency table with `min_frequency=2` |
| `configs/datasets/ce_csl.yaml` | CE-CSL adapter configuration |

Current counts:

| Item | Count |
|---|---:|
| Manifest records | 5,988 |
| Train records | 4,973 |
| Validation records | 515 |
| Test records | 500 |
| Gloss tokens retained at min frequency 2 | 1,937 |

Translator IDs observed: `A` through `L`.

## Feature Extraction Smoke Test

MediaPipe extraction has been verified on the first two CE-CSL training videos through the
project Docker environment.

Command:

```powershell
docker compose run --rm dev extract-manifest data/manifests/ce-csl.csv `
  --data-root data/ce-csl `
  --output data/processed/ce-csl `
  --limit 2 `
  --overwrite `
  --report artifacts/logs/ce-csl-extraction-smoke.csv
```

Result:

| Sample | Source frames | Valid frames | Valid ratio | Output |
|---|---:|---:|---:|---|
| `train-00001` | 193 | 193 | 1.0 | `48 x 368` |
| `train-00002` | 185 | 185 | 1.0 | `48 x 368` |

The generated `.npy` files and smoke-test report are local artifacts and are ignored by Git.

## Task Impact

CE-CSL is sentence-level continuous Chinese Sign Language data. The final implementation therefore
uses a dataset-bounded recognition target:

- Primary supervision comes from `Gloss`, split into reproducible token labels.
- `Chinese Sentences` are semantic references for explanation and reporting.
- Full Chinese sentence labels are not used as the main closed-set classification target because
  official dev/test sentences rarely repeat train sentences.
- Domain-specific templates are future migration work and are not part of CE-CSL main metrics.

## Open Checks Before Final Report

- Record the official source URL and citation for CE-CSL.
- Verify the license/access terms.
- Complete full landmark extraction and failure-rate reporting.
- Train and compare LSTM, BiLSTM, TCN, and compact Transformer on the Gloss token task.
- Report micro-F1, macro-F1, per-token F1, latency, and failure examples.
