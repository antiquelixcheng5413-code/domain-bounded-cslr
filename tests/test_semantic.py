import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cslr.semantic import IntentCatalog
from cslr.semantic.references import ExactSemanticResolver

ROOT = Path(__file__).resolve().parents[1]


class SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = IntentCatalog.from_yaml(ROOT / "configs/example_intents.yaml")

    def test_known_intent_above_threshold(self) -> None:
        result = self.catalog.reconstruct("registration", 0.9, 0.65)
        self.assertEqual(result.gloss, "REGISTER")

    def test_low_confidence_uses_fallback(self) -> None:
        result = self.catalog.reconstruct("registration", 0.4, 0.65)
        self.assertEqual(result.intent, "unknown")

    def test_gloss_maps_to_intent(self) -> None:
        result = self.catalog.reconstruct("REGISTER", 0.9, 0.65)
        self.assertEqual(result.intent, "registration")

    def test_exact_ce_csl_reference_requires_one_unique_sentence(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "label"
            labels.mkdir()
            header = "Number,Translator,Chinese Sentences,Gloss,Note\n"
            (labels / "train.csv").write_text(
                header + "s1,A,Sentence one,A/B,\n" + "s2,A,First C,C,\n",
                encoding="utf-8",
            )
            (labels / "dev.csv").write_text(
                header + "s3,A,Second C,C,\n",
                encoding="utf-8",
            )
            (labels / "test.csv").write_text(header, encoding="utf-8")
            resolver = ExactSemanticResolver.from_ce_csl(root)

            exact = resolver.resolve(["A", "B"])
            unknown = resolver.resolve(["X"])
            ambiguous = resolver.resolve(["C"])

        self.assertEqual(exact.status, "exact_reference")
        self.assertEqual(exact.text_zh, "Sentence one")
        self.assertEqual(unknown.status, "no_exact_reference")
        self.assertEqual(ambiguous.status, "ambiguous_reference")


if __name__ == "__main__":
    unittest.main()
