import tempfile
import unittest
from pathlib import Path

from cslr.inference.service import RecognitionService


class ServiceTests(unittest.TestCase):
    def test_missing_model_is_explicit(self) -> None:
        service = RecognitionService(Path("missing.onnx"), demo_mode=False)
        with tempfile.NamedTemporaryFile(suffix=".webm") as video:
            prediction = service.predict_video(Path(video.name))
        self.assertEqual(prediction.status, "model_unavailable")
        self.assertEqual(prediction.label, "unknown")
        self.assertEqual(prediction.gloss_tokens, [])
        self.assertEqual(prediction.confidence, 0.0)

    def test_demo_mode_is_marked(self) -> None:
        service = RecognitionService(None, demo_mode=True)
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "interface.webm"
            video_path.write_bytes(b"interface-test")
            prediction = service.predict_video(video_path)
        self.assertEqual(prediction.status, "demo_only")
        self.assertTrue(prediction.gloss_tokens)
        self.assertTrue(any("界面演示模式" in warning for warning in prediction.warnings))

    def test_legacy_ctc_is_disabled_by_default_for_web_service(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".onnx") as model:
            service = RecognitionService(Path(model.name), model_kind="legacy_ctc")

        self.assertFalse(service.ready)
        self.assertIn("disabled for the formal Web service", service.model_error or "")


if __name__ == "__main__":
    unittest.main()
