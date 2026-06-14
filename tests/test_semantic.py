import unittest
from pathlib import Path

from cslr.semantic import IntentCatalog

ROOT = Path(__file__).resolve().parents[1]


class SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = IntentCatalog.from_yaml(ROOT / "configs/hospital_intents.yaml")

    def test_known_intent_above_threshold(self) -> None:
        result = self.catalog.reconstruct("registration", 0.9, 0.65)
        self.assertEqual(result.gloss, "REGISTER")

    def test_low_confidence_uses_fallback(self) -> None:
        result = self.catalog.reconstruct("registration", 0.4, 0.65)
        self.assertEqual(result.intent, "unknown")

    def test_gloss_maps_to_intent(self) -> None:
        result = self.catalog.reconstruct("REGISTER", 0.9, 0.65)
        self.assertEqual(result.intent, "registration")


if __name__ == "__main__":
    unittest.main()
