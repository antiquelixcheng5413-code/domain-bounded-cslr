import unittest
from pathlib import Path

from cslr.contracts import SampleRecord
from cslr.data.gloss import build_gloss_vocabulary, encode_gloss_tokens, split_gloss_tokens


class GlossTests(unittest.TestCase):
    def test_split_gloss_tokens_ignores_punctuation(self) -> None:
        self.assertEqual(split_gloss_tokens("我/要/挂号/。"), ["我", "要", "挂号"])

    def test_build_vocabulary_uses_frequency_threshold(self) -> None:
        records = [
            SampleRecord("s1", Path("a.mp4"), "我/要/挂号", "A", "s", "train"),
            SampleRecord("s2", Path("b.mp4"), "我/要/买/票", "B", "s", "train"),
        ]

        tokens, counts = build_gloss_vocabulary(records, min_frequency=2)

        self.assertEqual(tokens, ["我", "要"])
        self.assertEqual(counts, {"我": 2, "要": 2})

    def test_encode_gloss_tokens_ignores_unknown_tokens(self) -> None:
        encoded = encode_gloss_tokens("我/要/挂号", {"我": 0, "挂号": 1})
        self.assertEqual(encoded, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
