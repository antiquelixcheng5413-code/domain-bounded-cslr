# CE-CSL final dataset audit

Last verified: 2026-06-17.

## Lock Status

CE-CSL is the final experiment dataset selected by the team.

This replaces NationalCSL-DP as the primary dataset candidate. NationalCSL-DP remains a useful
reference and historical audit record, but it is no longer the selected training/evaluation
dataset.

## Local Archive

| Field | Value |
|---|---|
| Archive | `E:\Download\CE-CSL.zip` |
| Size | 9,838,930,301 bytes |
| SHA256 | `9E87B76D3D3A5A6AC24195445A302431A0F508BDA3C129B74592853731C4AC41` |
| Last write time | 2026-06-17 00:21:45 |
| Git status | Do not commit archive or extracted videos |

The archive should be extracted outside the Git repository, for example:

```text
D:\FYP_downloads\data\ce-csl
```

The Docker mount should then point to the parent directory:

```powershell
$env:CSLR_DATA_ROOT="D:\FYP_downloads\data"
```

## Observed Archive Structure

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

Example row:

```csv
train-00001,A,2023年高考到了。,2/0/2/3/高/考/时间/到/。,
```

## Counts

| Split | Manifest split | Videos |
|---|---|---:|
| `train` | `train` | 4,973 |
| `dev` | `validation` | 515 |
| `test` | `test` | 500 |
| Total |  | 5,988 |

Translator IDs observed: `A` through `L`.

## Task Impact

CE-CSL is sentence-level continuous Chinese Sign Language data. This changes the project
assumption from isolated hospital-intent classification to a sentence/gloss-level baseline.

Immediate implementation:

- Build a manifest from the official train/dev/test CSV files.
- Use `Chinese Sentences` as the initial label column for the baseline.
- Keep `Gloss` available for later sequence-aware or semantic reconstruction experiments.
- Report CE-CSL results separately from any hospital-intent demo templates.

Open checks before final report:

- Record the official source and citation for CE-CSL.
- Verify the license/access terms.
- Decide whether the final model predicts full sentence labels, gloss strings, or a filtered
  domain subset.
- Benchmark MediaPipe extraction time on representative videos from all three splits.
