# Team Handoff Guide

Last updated: 2026-06-17.

This guide is for teammates who need to understand the current framework and continue from the
selected CE-CSL dataset.

The completed teammate delivery review is recorded in
[`teammate-handoff-2026-06-21.md`](teammate-handoff-2026-06-21.md).

## Current Project Status

The repository already contains:

- FastAPI backend and browser UI.
- Docker Compose service for running the web app.
- MediaPipe-based landmark extraction code.
- LSTM, BiLSTM, TCN, and compact Transformer model definitions.
- Single-label legacy training plus CE-CSL Gloss token multi-label training.
- ONNX export, ONNX inference, evaluation metrics, tests, and CI workflow.
- CE-CSL manifest and train-split Gloss token vocabulary.

The repository does not contain:

- Raw CE-CSL videos.
- Trained PyTorch checkpoint.
- ONNX model file.
- Real F1, failure-analysis, or latency tables.

The received CE-CSL landmarks are local-only in `data/clean_datas/ce_csl`; they are not in Git.
They passed manifest validation on 2026-06-21: 5,988/5,988 files, `48 x 368 float32`, SHA-256 tree
digest `7c240ffa3d5493c51f0c019ad8b1d068bdc40645507d6b6ee68803a5ea634386`. Because there is no
trained model yet, the web app currently starts and honestly reports
`model_unavailable`. This is expected.

## What Each Teammate Should Install

Required:

- Windows 10 or Windows 11.
- Git.
- VS Code or another editor.
- WSL 2.
- Docker Desktop with WSL 2 backend.

Docker login is not required.

Storage rule:

```text
Code repository:       E:\college\FYP or another non-C project folder
Large downloads/data:  D:\FYP_downloads or another D: location
Docker program/data:   D:\DockerDesktop and D:\DockerDesktopData if possible
CE-CSL source archive: currently E:\Download\CE-CSL.zip
```

All newly downloaded software, installers, tools, and large archives should be placed on `D:`.
Do not put raw videos, extracted frames, landmarks, checkpoints, or ONNX models into Git.

## First Run

Clone the private GitHub repository:

```powershell
git clone https://github.com/antiquelixcheng5413-code/domain-bounded-cslr.git
cd domain-bounded-cslr
```

Start the app:

```powershell
docker compose up -d --build app
```

Open:

```text
http://localhost:8088
```

Check health:

```powershell
Invoke-RestMethod -Uri "http://localhost:8088/api/v1/health"
```

Expected before training:

```json
{
  "status": "ok",
  "model_ready": false,
  "demo_mode": false
}
```

Stop the app:

```powershell
docker compose down
```

## How the Framework Fits Together

```text
Browser UI
  -> FastAPI upload endpoint
  -> temporary video file
  -> MediaPipe landmark extraction
  -> temporal model inference through ONNX
  -> low-quality / low-confidence rejection
  -> CE-CSL label/token output and constrained Chinese explanation
```

Current behavior:

- If no model exists: return `model_unavailable`.
- If demo mode is enabled: return clearly marked `demo_only`.
- If a model exists but input quality is poor: return `low_quality`.
- If confidence is too low: return `low_confidence`.
- Only accepted real-model predictions should be used for reported results.

## Main Folders

| Path | Purpose |
|---|---|
| `app/backend` | FastAPI routes, schemas, and service creation |
| `app/frontend` | Browser UI for upload/camera/result display |
| `src/cslr/data` | Dataset adapters, manifests, Gloss token vocabulary tools |
| `src/cslr/features` | MediaPipe extraction, normalization, resampling |
| `src/cslr/models` | LSTM, BiLSTM, TCN, Transformer definitions |
| `src/cslr/training` | Training runner and ONNX export |
| `src/cslr/evaluation` | Single-label and multi-label metrics |
| `src/cslr/inference` | ONNX recognizer and recognition service |
| `src/cslr/semantic` | Optional template mapping, not the current default main output |
| `configs` | Dataset/model/training/feature configs |
| `data/manifests` | Anonymous sample indexes and token vocabulary |
| `docs` | Dataset decisions, setup notes, experiment records, reports |
| `artifacts` | Local-only model exports, checkpoints, logs, metrics |

## Team Roles for the Next Phase

| Role | Immediate work |
|---|---|
| Environment/Git owner | Verify clone, Docker start, tests, Git workflow |
| Dataset owner | Verify CE-CSL source/license, extraction location, manifest and vocabulary |
| Feature pipeline owner | Run full CE-CSL landmark extraction and quality report |
| Model owner | Train LSTM Gloss token baseline and export ONNX |
| Frontend/backend owner | Keep upload/camera flow and token result display stable |
| Report owner | Maintain Week 6 evidence and experiment record templates |

## Useful Checks

Run tests:

```powershell
docker compose --profile test run --rm test
```

Safety check:

```powershell
python scripts/check_repository_safety.py
```

List dataset adapters:

```powershell
docker compose run --rm dev list-adapters
```

Validate CE-CSL manifest:

```powershell
docker compose run --rm dev validate-manifest data/manifests/ce-csl.csv
```

Build CE-CSL Gloss token vocabulary:

```powershell
docker compose run --rm dev build-gloss-vocab data/manifests/ce-csl.csv `
  --output data/manifests/ce-csl-gloss-vocab.csv `
  --min-frequency 2
```

Full extraction command for the feature owner:

```powershell
docker compose run --rm dev extract-manifest data/manifests/ce-csl.csv `
  --data-root data/ce-csl `
  --output data/processed/ce-csl `
  --report artifacts/logs/ce-csl-extraction.csv `
  --continue-on-error
```

Validate the received feature bundle:

```powershell
docker compose run --rm dev verify-features data/manifests/ce-csl.csv `
  --features data/clean_datas/ce_csl `
  --sha256 `
  --receipt artifacts/metrics/ce-csl-features-receipt.json
```

## Current Dataset Note

CE-CSL is the final selected dataset. It contains sentence-level continuous CSL videos with
official `train`, `dev`, and `test` CSV labels.

The project target is now CE-CSL dataset-bounded recognition:

- Main supervision: `Gloss` split into token labels.
- Semantic reference: `Chinese Sentences`.
- Main metrics: micro-F1, macro-F1, per-token F1, latency, and failure analysis.
- Not a main metric: business-intent accuracy.
- Not a main metric: full Chinese sentence closed-set classification accuracy.

Manual team recordings are allowed only for UI and workflow demonstration, not as the main
training/evaluation evidence.
