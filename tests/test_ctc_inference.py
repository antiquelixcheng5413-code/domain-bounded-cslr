import tempfile
import unittest
from pathlib import Path

import numpy as np

from cslr.recognition.dataset import GlossSequenceDataset, collate_samples, load_records, split_records
from cslr.recognition.gloss_sequence import GlossSequenceConfig, build_ordered_vocabulary
from cslr.recognition.inference import (
    evaluate_checkpoint,
    format_prediction,
    load_recognizer,
    predict_video,
)
from cslr.recognition.model import CTCConfig, CTCRecognizer, ctc_config_from_dict
from cslr.recognition.training import TrainingConfig, resolve_device, train_ctc

MANIFEST_HEADER = "sample_id,video,label,signer,session,split"


def _write_corpus(root: Path, train_count: int = 6, dev_count: int = 2, length: int = 24, width: int = 12):
    features = root / "features"
    features.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = [MANIFEST_HEADER]
    for index in range(train_count):
        sample_id = f"train-{index:05d}"
        array = rng.normal(0, 0.02, size=(length, width)).astype(np.float32)
        for token in range(3):
            start = token * (length // 3)
            array[start : start + length // 3, token] += 3.0
        np.save(features / f"{sample_id}.npy", array)
        lines.append(f"{sample_id},video/train/A/{sample_id}.mp4,一/二/三,A,train,train")
    for index in range(dev_count):
        sample_id = f"dev-{index:05d}"
        array = rng.normal(0, 0.02, size=(length, width)).astype(np.float32)
        for token in range(3):
            start = token * (length // 3)
            array[start : start + length // 3, token] += 3.0
        np.save(features / f"{sample_id}.npy", array)
        lines.append(f"{sample_id},video/dev/A/{sample_id}.mp4,一/二/三,A,dev,validation")
    manifest = root / "manifest.csv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest, features


def _train_small(manifest: Path, features: Path, root: Path, unit: str = "token") -> Path:
    vocabulary, _ = build_ordered_vocabulary(
        ["一/二/三"], min_frequency=1, config=GlossSequenceConfig(target_unit=unit)
    )
    checkpoint = root / "checkpoint.pt"
    train_ctc(
        manifest_path=manifest,
        feature_root=features,
        vocabulary=vocabulary,
        model_config=CTCConfig(
            input_size=12, vocabulary_size=vocabulary.size, hidden_size=16, num_layers=1, dropout=0.0
        ),
        training_config=TrainingConfig(
            epochs=2, batch_size=2, device="cpu", amp=False, seed=0, early_stopping_patience=5
        ),
        output_path=checkpoint,
    )
    return checkpoint


class CtcConfigRoundTripTests(unittest.TestCase):
    def test_serialised_blank_index_is_ignored(self) -> None:
        payload = {"input_size": 368, "vocabulary_size": 10, "blank_index": 0}
        config = ctc_config_from_dict(payload)
        self.assertEqual(config.input_size, 368)
        self.assertEqual(config.vocabulary_size, 10)

    def test_unknown_keys_are_dropped_not_crashed(self) -> None:
        config = ctc_config_from_dict({"input_size": 8, "vocabulary_size": 4, "whatever": 1})
        self.assertEqual(config.input_size, 8)


class RecognizerLoadingTests(unittest.TestCase):
    def test_checkpoint_round_trip_restores_vocabulary_and_normalizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, features = _write_corpus(root)
            checkpoint = _train_small(manifest, features, root)
            model, vocabulary, normalizer, training_config, device = load_recognizer(checkpoint)
            self.assertIsNotNone(normalizer)
            self.assertEqual(len(normalizer.mean), 12)
            self.assertIn("一", vocabulary)
            self.assertEqual(training_config.get("feature_view", "full"), "full")
            self.assertEqual(model.config.input_size, 12)
            self.assertEqual(device, resolve_device("auto"))

    def test_missing_checkpoint_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_recognizer(Path("does-not-exist.pt"))

    def test_character_checkpoint_restores_the_target_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, features = _write_corpus(root)
            checkpoint = _train_small(manifest, features, root, unit="char")
            _, vocabulary, _, _, _ = load_recognizer(checkpoint)
            self.assertEqual(vocabulary.config.target_unit, "char")
            self.assertEqual(vocabulary.units("一/二/三"), ["一", "二", "三"])


class EvaluateCheckpointTests(unittest.TestCase):
    def test_writes_predictions_and_metrics_for_dev(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, features = _write_corpus(root)
            checkpoint = _train_small(manifest, features, root)
            output = root / "eval.json"
            payload = evaluate_checkpoint(
                checkpoint_path=checkpoint,
                manifest_path=manifest,
                feature_root=features,
                split="dev",
                output_path=output,
                batch_size=2,
                device="cpu",
            )
            self.assertTrue(output.exists())
            self.assertEqual(payload["samples"], 2)
            self.assertFalse(payload["test_split_read"])
            self.assertEqual(payload["split"], "validation")
            self.assertIn("wer", payload["metrics"])
            self.assertEqual(len(payload["predictions"]), 2)
            first = payload["predictions"][0]
            self.assertIn("sample_id", first)
            self.assertIn("reference", first)
            self.assertIn("prediction", first)

    def test_frozen_split_is_refused_before_reading_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, features = _write_corpus(root)
            checkpoint = _train_small(manifest, features, root)
            with self.assertRaises(ValueError):
                evaluate_checkpoint(
                    checkpoint_path=checkpoint,
                    manifest_path=manifest,
                    feature_root=features,
                    split="test",
                    output_path=root / "eval.json",
                    device="cpu",
                )

    def test_missing_features_are_skipped_and_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, features = _write_corpus(root)
            checkpoint = _train_small(manifest, features, root)
            (features / "dev-00001.npy").unlink()
            payload = evaluate_checkpoint(
                checkpoint_path=checkpoint,
                manifest_path=manifest,
                feature_root=features,
                split="dev",
                output_path=root / "eval.json",
                device="cpu",
            )
            self.assertEqual(payload["samples"], 1)
            self.assertEqual(payload["skipped_without_features"], 1)

    def test_limit_caps_the_evaluated_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, features = _write_corpus(root, train_count=4, dev_count=2)
            checkpoint = _train_small(manifest, features, root)
            payload = evaluate_checkpoint(
                checkpoint_path=checkpoint,
                manifest_path=manifest,
                feature_root=features,
                split="dev",
                output_path=root / "eval.json",
                limit=1,
                device="cpu",
            )
            self.assertEqual(payload["samples"], 1)


class FormattingTests(unittest.TestCase):
    def test_format_prediction_skips_unknown(self) -> None:
        self.assertEqual(format_prediction(["我", "<unk>", "要"]), "我/要")
        self.assertEqual(format_prediction([]), "")


class PredictVideoTests(unittest.TestCase):
    def test_video_path_runs_extraction_then_decode(self) -> None:
        """The video path is exercised with a stubbed extractor (no MediaPipe needed)."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, features = _write_corpus(root)
            checkpoint = _train_small(manifest, features, root)
            video = root / "clip.mp4"
            video.write_bytes(b"not a real video")

            class StubExtractor:
                def __init__(self, sequence_length: int = 96, minimum_valid_ratio: float = 0.8):
                    self.sequence_length = sequence_length
                    self.minimum_valid_ratio = minimum_valid_ratio

                def extract(self, source):
                    import numpy as np

                    from cslr.contracts import QualityReport
                    from cslr.features.extractor import ExtractionResult

                    array = np.zeros((24, 12), dtype=np.float32)
                    for token in range(3):
                        array[token * 8 : (token + 1) * 8, token] = 3.0
                    return ExtractionResult(
                        features=array,
                        quality=QualityReport(24, 24, 1.0, True, []),
                        source_frames=24,
                    )

            import cslr.features.extractor as extractor_module

            original = extractor_module.MediaPipeHolisticExtractor
            extractor_module.MediaPipeHolisticExtractor = StubExtractor  # type: ignore[assignment]
            try:
                payload = predict_video(
                    checkpoint_path=checkpoint, video_path=video, device="cpu"
                )
            finally:
                extractor_module.MediaPipeHolisticExtractor = original  # type: ignore[assignment]

        self.assertEqual(payload["source_frames"], 24)
        self.assertTrue(payload["quality_accepted"])
        self.assertIn("gloss_tokens", payload)
        self.assertIsInstance(payload["gloss_tokens"], list)
        self.assertIn("extraction", payload["latency_ms"])
        self.assertIn("decode", payload["latency_ms"])

    def test_missing_video_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, features = _write_corpus(root)
            checkpoint = _train_small(manifest, features, root)
            # the real extractor is not reached because the file does not exist
            with self.assertRaises(Exception):
                predict_video(
                    checkpoint_path=checkpoint,
                    video_path=root / "missing.mp4",
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
