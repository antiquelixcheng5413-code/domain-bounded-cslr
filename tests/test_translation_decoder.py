"""Unit tests for the tiny Chinese decoder and gloss auxiliary head."""

from __future__ import annotations

import unittest

import torch

from cslr.translation.decoder import TinyTransformerChineseDecoder


class DecoderTest(unittest.TestCase):
    def _vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "高": 4, "考": 5, "到": 6, "了": 7}

    def _decoder(self, **kw) -> TinyTransformerChineseDecoder:
        defaults = dict(
            vocab_size=8,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            feedforward_dim=32,
            max_target_len=32,
            gloss_vocab_size=None,
        )
        defaults.update(kw)
        return TinyTransformerChineseDecoder(**defaults)

    def test_teacher_forcing_logits(self) -> None:
        dec = self._decoder()
        visual = torch.randn(2, 8, 16)
        vmask = torch.ones(2, 8, dtype=torch.bool)
        target = torch.tensor([[1, 4, 5, 2], [1, 4, 6, 2]])
        result = dec.decode_from_visual(visual, vmask, target)
        logits = result[0]
        self.assertEqual(logits.shape, (2, 4, 8))

    def test_gloss_head_created_when_requested(self) -> None:
        dec = self._decoder(gloss_vocab_size=50)
        self.assertIsNotNone(dec.gloss_head)
        self.assertEqual(dec.gloss_head.out_features, 50)

    def test_gloss_head_none_by_default(self) -> None:
        dec = self._decoder()
        self.assertIsNone(dec.gloss_head)

    def test_generate_shape(self) -> None:
        dec = self._decoder()
        vocab = self._vocab()
        visual = torch.randn(1, 8, 16)
        vmask = torch.ones(1, 8, dtype=torch.bool)
        texts = dec.generate(visual, vmask, vocab, max_len=8)
        self.assertEqual(len(texts), 1)
        self.assertIsInstance(texts[0], str)


if __name__ == "__main__":
    unittest.main()