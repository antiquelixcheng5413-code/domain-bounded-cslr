# Week 6 local progress review

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

- If the primary dataset remains inaccessible by the end of Week 2, switch candidates.
- If a real model is not available, show the upload pipeline but label the milestone incomplete.
- Never replace missing results with demo-mode predictions.
- If signer-independent performance is low, report it and investigate rather than changing the
  split.
