import unittest

from fastapi.testclient import TestClient

from app.backend.main import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

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
        self.assertIn("医院前台手语识别", page.text)
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
        self.assertEqual(body["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
