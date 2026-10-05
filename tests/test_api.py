import os
import unittest

from fastapi.testclient import TestClient


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Force the "no recogniser assets" scenario so these contract tests pass on any machine,
        # including a dev box where the real VL48 models are installed on disk. The UI-level
        # behaviour that genuinely matters here is *honesty*: absent a trained model, the API must
        # not fabricate a prediction and must advertise readiness as False.
        os.environ["CSLR_CTC_CHECKPOINT"] = os.path.join("definitely", "missing", "model.pt")
        os.environ["CSLR_VISION_MODEL"] = os.path.join("definitely", "missing", "vision")
        os.environ["CSLR_LLM_MODEL"] = os.path.join("definitely", "missing", "llm")

        # routes.py calls create_recognition_service() at import time and caches it, so the
        # no-model env vars above must be in place before the FastAPI app is first imported.
        import app.backend.main as backend_main

        cls.client = TestClient(backend_main.app)

    @classmethod
    def tearDownClass(cls) -> None:
        for key in ("CSLR_CTC_CHECKPOINT", "CSLR_VISION_MODEL", "CSLR_LLM_MODEL"):
            os.environ.pop(key, None)

    def test_health_reports_model_state(self) -> None:
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertFalse(body["model_ready"])

    def test_homepage_and_styles_are_served(self) -> None:
        page = self.client.get("/")
        styles = self.client.get("/static/styles.css")
        self.assertEqual(page.status_code, 200)
        self.assertIn("CE-CSL 数据集受限手语识别", page.text)
        self.assertEqual(styles.status_code, 200)
        self.assertIn(".result-heading", styles.text)

    def test_rejects_non_video_extension(self) -> None:
        response = self.client.post(
            "/api/v1/predict",
            files={"video": ("notes.txt", b"not a video", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)

    def test_missing_model_does_not_fake_prediction(self) -> None:
        response = self.client.post(
            "/api/v1/predict",
            files={"video": ("sample.webm", b"test-video", "video/webm")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "model_unavailable")
        self.assertEqual(body["label"], "unknown")
        self.assertEqual(body["gloss_tokens"], [])
        self.assertEqual(body["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
