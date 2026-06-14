import tempfile
import unittest
from pathlib import Path

from cslr.inference.service import RecognitionService
from cslr.semantic import IntentCatalog

ROOT = Path(__file__).resolve().parents[1]


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = IntentCatalog.from_yaml(ROOT / "configs/hospital_intents.yaml")

    def test_missing_model_is_explicit(self) -> None:
        service = RecognitionService(self.catalog, Path("missing.onnx"), demo_mode=False)
        with tempfile.NamedTemporaryFile(suffix=".webm") as video:
            prediction = service.predict_video(Path(video.name))
        self.assertEqual(prediction.status, "model_unavailable")
        self.assertEqual(prediction.confidence, 0.0)

    def test_demo_mode_is_marked(self) -> None:
        service = RecognitionService(self.catalog, None, demo_mode=True)
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "interface.webm"
            video_path.write_bytes(b"interface-test")
            prediction = service.predict_video(video_path)
        self.assertEqual(prediction.status, "demo_only")
        self.assertTrue(any("演示模式" in warning for warning in prediction.warnings))


if __name__ == "__main__":
    unittest.main()
