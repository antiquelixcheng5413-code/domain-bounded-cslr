# Week 6 local progress review

## Current evidence as of 2026-06-16

- [x] WSL 2 and Docker Desktop are installed on the Windows development computer.
- [x] Docker Desktop program, data disk, config, downloads, and tools were moved to `D:`.
- [x] Docker starts from `D:\DockerDesktop`.
- [x] The application container is healthy at `http://localhost:8088`.
- [x] `docker compose up -d --build app` starts the `app` service.
- [x] Health endpoint returns `model_ready=false` while no ONNX model is installed.
- [x] Prediction endpoint returns `model_unavailable` rather than a fake result.
- [x] All 24 tests pass in the Docker test image.
- [x] CE-CSL final archive was identified and SHA256-audited.
- [x] NationalCSL-DP metadata and a raw-video sample archive passed MD5 verification as a
  historical candidate.
- [x] One official MP4 produced an accepted `48 x 368` MediaPipe feature sequence.
- [x] The CE-CSL archive structure and official split CSV files have been inspected.
- [x] The CE-CSL fixed manifest has been generated and validated.
- [x] CE-CSL batch extraction smoke test succeeded for two training videos.
- [ ] A real LSTM model has been trained and exported.
- [x] GitHub private remote deployment is complete.

## Required live demonstration

- [ ] `docker compose up app` starts on a clean Windows/Docker setup.
- [ ] Browser upload works.
- [ ] Browser camera records a 1–4 second clip.
- [ ] The backend decodes video and extracts a 48-frame feature sequence.
- [ ] A trained LSTM ONNX model produces a bounded prediction.
- [ ] Low-quality and low-confidence clips are rejected.
- [ ] The Chinese template preserves the recognized intent.
- [ ] Demo mode is disabled for reported results.

## Required evidence

- [ ] Dataset name, version, license, URL, checksum, and access date.
- [ ] Manifest schema and sample counts by label/signer/split.
- [ ] Landmark success and rejection rates.
- [ ] Random-split Top-1 and macro-F1.
- [ ] Signer-independent preliminary Top-1 and macro-F1.
- [ ] Confusion matrix and at least three failure examples.
- [ ] Extraction, inference, and end-to-end P50/P95 latency.
- [ ] Git commit and exact configuration for every table or figure.

## Presentation structure

1. Scope and non-claims.
2. Dataset decision.
3. Reproducible pipeline.
4. Live demonstration.
5. Preliminary results.
6. Failure analysis.
7. Weeks 7–12 decisions.

## Fallback rules

- If CE-CSL license/source metadata cannot be documented, record the gap before final report.
- If a real model is not available, show the upload pipeline but label the milestone incomplete.
- Never replace missing results with demo-mode predictions.
- If signer-independent performance is low, report it and investigate rather than changing the
  split.
