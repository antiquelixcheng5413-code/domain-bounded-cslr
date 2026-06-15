# Local runbook

Last updated: 2026-06-16.

This is the personal runbook for starting, checking, and stopping the current framework on the
Windows machine used for development.

## Machine layout

Current local layout:

```text
Project repository:  E:\college\FYP
Docker Desktop:      D:\DockerDesktop
Docker data:         D:\DockerDesktopData
Docker config:       D:\DockerDesktopConfig
Download cache:      D:\FYP_downloads
Portable gh:         D:\FYP_tools
```

The project directory still contains `.downloads` and `.tools`, but they are junctions to D
drive locations.

## Start Docker Desktop

If Docker is not already running:

```powershell
Start-Process "D:\DockerDesktop\Docker Desktop.exe"
```

Make the current PowerShell session use the D-drive Docker CLI:

```powershell
$env:PATH="D:\DockerDesktop\resources\bin;$env:PATH"
```

Check Docker:

```powershell
docker version
docker compose version
```

## Start the web app

From the repository:

```powershell
cd E:\college\FYP
$env:PATH="D:\DockerDesktop\resources\bin;$env:PATH"
docker compose up -d --build app
```

Open:

```text
http://localhost:8088
```

## Check service health

```powershell
docker compose ps
docker inspect --format "{{.State.Health.Status}}" fyp-app-1
Invoke-RestMethod -Uri "http://localhost:8088/api/v1/health"
```

Expected now:

```json
{
  "status": "ok",
  "model_ready": false,
  "demo_mode": false,
  "model_error": "model file does not exist: /workspace/artifacts/exports/lstm.onnx"
}
```

This is correct because the real model has not been trained or exported.

## Check page assets

```powershell
$page = Invoke-WebRequest -Uri "http://localhost:8088/" -UseBasicParsing
$style = Invoke-WebRequest -Uri "http://localhost:8088/static/styles.css" -UseBasicParsing
$page.StatusCode
$page.Content -like "*GROUP 11*"
$style.StatusCode
```

Expected:

```text
200
True
200
```

## Check prediction behavior before model training

This verifies that the service does not fake predictions:

```powershell
New-Item -ItemType Directory -Force -Path "artifacts\logs\service-smoke" | Out-Null
Set-Content -LiteralPath "artifacts\logs\service-smoke\dummy.webm" `
  -Value "not-a-real-video" -NoNewline -Encoding ASCII
curl.exe -s -S -F "video=@artifacts\logs\service-smoke\dummy.webm;type=video/webm" `
  http://localhost:8088/api/v1/predict
```

Expected status:

```json
{
  "status": "model_unavailable",
  "confidence": 0.0
}
```

## Run tests

```powershell
docker compose --profile test run --rm test
```

Expected now: 20 tests pass.

## Stop services

```powershell
docker compose down
```

## Run in demo mode only for UI checking

Demo mode is not real recognition and must not be used in the report.

```powershell
$env:CSLR_DEMO_MODE="true"
docker compose up -d --build app
```

Turn it off afterwards:

```powershell
$env:CSLR_DEMO_MODE="false"
docker compose down
```

## Where the real model must go later

Real prediction needs:

```text
E:\college\FYP\artifacts\exports\lstm.onnx
E:\college\FYP\artifacts\exports\lstm.labels.json
```

These files are intentionally ignored by Git.

## Common issues

Port already in use:

```powershell
$env:CSLR_PORT="8090"
docker compose up -d app
```

Docker command not found:

```powershell
$env:PATH="D:\DockerDesktop\resources\bin;$env:PATH"
```

Docker stuck starting:

```powershell
wsl --shutdown
Start-Process "D:\DockerDesktop\Docker Desktop.exe"
```

Accidentally created large files in the repo:

```powershell
git status --short
python scripts/check_repository_safety.py
```

Do not commit data, logs, model weights, or secrets.
