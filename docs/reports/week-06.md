# Week 6 Local Progress Review

## Current Evidence as of 2026-06-21

- [x] WSL 2 and Docker Desktop are installed on the Windows development computer.
- [x] Docker Desktop program, data disk, config, downloads, and tools were moved to `D:`.
- [x] Docker starts from `D:\DockerDesktop`.
- [x] The application container is healthy at `http://localhost:8088`.
- [x] `docker compose up -d --build app` starts the `app` service.
- [x] Health endpoint returns `model_ready=false` while no ONNX model is installed.
- [x] Prediction endpoint returns `model_unavailable` rather than a fake result.
- [x] GitHub private remote deployment is complete.
- [x] CE-CSL final archive was identified and SHA256-audited.
- [x] The CE-CSL archive structure and official split CSV files have been inspected.
- [x] CE-CSL manifest has been generated from `Gloss` labels and validated.
- [x] Train-split Gloss token vocabulary has been generated with `min_frequency=2`.
- [x] CE-CSL batch extraction smoke test succeeded for two training videos.
- [x] Full CE-CSL landmark feature bundle received and validated: 5,988/5,988 files,
  `48 x 368 float32`, no missing or unreadable files.
- [ ] A real LSTM Gloss token model has been trained and exported.

## Teammate CTC Handoff (Not Yet Independently Reproduced)

The teammate delivered a CTC checkpoint and ONNX model, both retained outside Git and verified to
load in Docker. The reported official-split metrics are recorded in
[`teammate-ctc-official-split-summary.json`](../../artifacts/metrics/teammate-ctc-official-split-summary.json):

| Split | Samples | Sequence accuracy | WER |
|---|---:|---:|---:|
| Dev | 515 | 26.60% | 24.58% |
| Test | 500 | 19.80% | 26.56% |

Reported average CPU inference latency is 12.5 ms. The detailed prediction CSV independently
confirms the Dev/Test sequence accuracies, but whitespace-token WER recomputation gives 30.12%
Dev and 35.66% Test rather than the reported 24.58% and 26.56%. The CTC evaluation script's
normalization/tokenization rules are required before reporting WER as reproduced. These are CTC
sequence-recognition results, not the active LSTM multi-label baseline.

## Required Live Demonstration

- [ ] `docker compose up app` starts on a clean Windows/Docker setup.
- [ ] Browser upload works.
- [ ] Browser camera records a 1-3 second clip.
- [ ] The backend decodes video and extracts a 48-frame feature sequence.
- [ ] A trained LSTM ONNX model returns dataset-bounded gloss/token predictions.
- [ ] Low-quality and low-confidence clips are rejected.
- [ ] Web response shows predicted label, gloss tokens, confidence, and latency.
- [ ] Demo mode is disabled for reported results.

## CTC Sequence Comparison Update

- [x] Teammate legacy CTC checkpoint and ONNX were integrated through a compatibility layer.
- [x] Legacy ONNX Dev/Test evaluation was regenerated with official splits and corpus-level WER.
- [x] The 3,862-token ordered vocabulary is versioned and checksum recorded.
- [x] Legacy checkpoint and ONNX produce identical Dev predictions under the corrected decoder.
- [x] CTC v2 training, export, evaluation, and Web model-kind plumbing are implemented.
- [ ] GPU preflight and one-epoch CTC v2 smoke training.
- [ ] Train CTC v2 with official Train, select by Dev corpus WER, then evaluate Test once.

The reproducible legacy metrics are Dev sequence accuracy 26.60% / corpus WER 30.12% and Test
sequence accuracy 19.80% / corpus WER 35.66%. Do not use the earlier stated WER values of 24.58%
and 26.56%, which are not reproduced by the detailed predictions. Full commands and error counts
are in [the CTC experiment record](../experiments/ctc-sequence-recognition.md).

## Required Evidence

- [ ] Dataset name, version, license, URL, checksum, and access date.
- [ ] Manifest schema and sample counts by signer/split.
- [ ] Gloss token vocabulary size, frequency threshold, and split coverage.
- [x] Landmark file-success report: 5,988 success, 0 failure; valid-frame ratio is still missing.
- [ ] LSTM validation micro-F1, macro-F1, subset accuracy, and per-token F1.
- [ ] Later model comparison: BiLSTM, TCN, compact Transformer.
- [ ] Error analysis with at least three failure examples.
- [ ] Extraction, inference, and end-to-end P50/P95 latency.
- [ ] Git commit and exact configuration for every table or figure.

## Presentation Structure

1. Scope and non-claims: CE-CSL dataset-bounded recognition, not open-domain translation.
2. Dataset decision and why full sentence closed-set classification is not the main target.
3. Reproducible pipeline: manifest, Gloss vocabulary, landmark extraction, training.
4. Live Web demonstration.
5. Preliminary LSTM token-level results.
6. Failure analysis and next model comparison.
7. Weeks 7-12 decisions.

## Fallback Rules

- If CE-CSL license/source metadata cannot be documented, record the gap before final report.
- If a real model is not available, show the upload pipeline but label the milestone incomplete.
- Never replace missing results with demo-mode predictions.
- Do not report business-intent accuracy unless a separate validated intent-labeled dataset or
  explicit mapping experiment exists.
- Do not report full Chinese sentence closed-set accuracy as the main CE-CSL result.
