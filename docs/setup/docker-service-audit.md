# Docker Service Audit

Audit date: 2026-06-17.

This document records the current Docker service state. It is independent of model quality: the
web service can start, report that no trained model is installed, and refuse to fake predictions.

## Storage and Runtime

| Item | Verified value |
|---|---|
| Docker CLI | `D:\DockerDesktop\resources\bin\docker.exe` |
| Docker Desktop | `D:\DockerDesktop` |
| Docker data disk | `D:\DockerDesktopData` |
| Compose project | `E:\college\FYP` |
| Service | `app` |
| Port mapping | `0.0.0.0:8088->8000/tcp` |

## Build and Start

Run from `E:\college\FYP`:

```powershell
$env:PATH="D:\DockerDesktop\resources\bin;$env:PATH"
docker compose up -d --build app
```

Check service state:

```powershell
docker compose ps
```

## HTTP Checks

Health endpoint:

```powershell
Invoke-RestMethod -Uri "http://localhost:8088/api/v1/health"
```

Expected response before a model is trained:

```json
{
  "status": "ok",
  "model_ready": false,
  "demo_mode": false,
  "model_error": "model file does not exist: /workspace/artifacts/exports/lstm.onnx"
}
```

Page and static asset checks:

```powershell
Invoke-WebRequest -Uri "http://localhost:8088/" -UseBasicParsing
Invoke-WebRequest -Uri "http://localhost:8088/static/styles.css" -UseBasicParsing
```

Expected result: both return HTTP 200. The page contains `CE-CSL 数据集受限手语识别`.

Prediction endpoint before a model is trained:

```powershell
New-Item -ItemType Directory -Force -Path "artifacts\logs\service-smoke" | Out-Null
Set-Content -LiteralPath "artifacts\logs\service-smoke\dummy.webm" `
  -Value "not-a-real-video" -NoNewline -Encoding ASCII
curl.exe -s -S -F "video=@artifacts\logs\service-smoke\dummy.webm;type=video/webm" `
  http://localhost:8088/api/v1/predict
```

Expected response:

```json
{
  "status": "model_unavailable",
  "label": "unknown",
  "gloss_tokens": [],
  "intent": "unknown",
  "gloss": "UNKNOWN",
  "text_zh": "模型尚未安装或训练，当前不能进行真实识别。",
  "confidence": 0.0,
  "top_k": [],
  "warnings": [
    "model file does not exist: /workspace/artifacts/exports/lstm.onnx"
  ],
  "model_version": null
}
```

This is a pass condition for the service layer. It means the system is honest about missing
model artifacts and does not fabricate recognition results.

## Model Artifact Contract

Real prediction requires both files below under the project directory:

```text
artifacts/exports/lstm.onnx
artifacts/exports/lstm.labels.json
```

These are intentionally ignored by Git. The final ONNX model should be distributed through a
private GitHub Release or another approved artifact channel, not committed to normal Git history.
