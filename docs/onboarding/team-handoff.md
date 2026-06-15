# Team handoff guide

Last updated: 2026-06-16.

This guide is for teammates who need to understand the current framework and start working
without touching the final dataset decision yet.

## Current project status

The repository already contains:

- FastAPI backend and browser UI.
- Docker Compose service for running the web app.
- MediaPipe-based landmark extraction code.
- LSTM, BiLSTM, TCN, and compact Transformer model definitions.
- Training, ONNX export, inference, evaluation, and semantic-template modules.
- Tests and CI workflow.
- Dataset decision documents.

The repository does not yet contain:

- Final selected dataset.
- Extracted landmarks for training.
- Trained PyTorch checkpoint.
- ONNX model file.
- Real accuracy, F1, confusion matrix, or latency tables.

Because there is no trained model yet, the web app currently starts and honestly reports
`model_unavailable`. This is expected.

## What each teammate should install

Required:

- Windows 10 or Windows 11.
- Git.
- VS Code or another editor.
- WSL 2.
- Docker Desktop with WSL 2 backend.

Docker login is not required.

Recommended local storage layout:

```text
Code repository:       E:\college\FYP or another non-C project folder
Large downloads/data:  D:\FYP_downloads
Docker program/data:   D:\DockerDesktop and D:\DockerDesktopData if possible
```

Do not put raw videos, extracted frames, landmarks, checkpoints, or ONNX models into Git.

## First run

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

## How the framework fits together

```text
Browser UI
  -> FastAPI upload endpoint
  -> temporary video file
  -> MediaPipe landmark extraction
  -> temporal model inference through ONNX
  -> low-quality / low-confidence rejection
  -> deterministic Chinese template output
```

Current behavior:

- If no model exists: return `model_unavailable`.
- If demo mode is enabled: return clearly marked `demo_only`.
- If a model exists but input quality is poor: return `low_quality`.
- If confidence is too low: return `low_confidence`.
- Only accepted predictions should be used for reported results.

## Main folders

| Path | Purpose |
|---|---|
| `app/backend` | FastAPI routes, schemas, and service creation |
| `app/frontend` | Browser UI for upload/camera/demo result display |
| `src/cslr/features` | MediaPipe extraction, normalization, resampling |
| `src/cslr/models` | LSTM, BiLSTM, TCN, Transformer definitions |
| `src/cslr/training` | Training runner and ONNX export |
| `src/cslr/inference` | ONNX recognizer and recognition service |
| `src/cslr/semantic` | Intent-to-Chinese-template reconstruction |
| `configs` | Dataset/model/training/feature/intent configs |
| `data/manifests` | Anonymous sample indexes that can be committed |
| `docs` | Dataset decisions, setup notes, experiment records, reports |
| `artifacts` | Local-only model exports, checkpoints, logs, metrics |

## Team roles for the next phase

Suggested split:

| Role | Immediate work |
|---|---|
| Environment/Git owner | Verify clone, Docker start, tests, Git workflow |
| Dataset owner | Compare datasets, record license/access/storage facts |
| Feature pipeline owner | Validate single-video/image-sequence extraction |
| Model owner | Prepare LSTM baseline and ONNX export path |
| Frontend/backend owner | Keep web upload/camera flow stable |
| Report owner | Maintain Week 6 evidence and experiment record templates |

## Git workflow

Use short branches:

```powershell
git switch -c feature/short-name
git status
git add path/to/files
git commit -m "feat(scope): short description"
git push -u origin feature/short-name
```

Keep `main` clean. Avoid committing:

- raw videos
- extracted frame folders
- full landmark arrays
- `.pt`, `.pth`, `.ckpt`, `.onnx`
- logs
- `.env`
- keys or tokens
- identity/consent materials

## Useful checks

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

Validate a manifest:

```powershell
docker compose run --rm dev validate-manifest data/manifests/example.csv
```

## Current dataset note

NationalCSL-DP is only a temporary primary candidate. It is useful because it is public,
Chinese, isolated-sign based, signer-labeled, and has hospital-related glosses. The team must
still finalize the dataset choice before formal training and reporting.

Manual team recordings are allowed only for UI and workflow demonstration, not as the main
training/evaluation evidence.
