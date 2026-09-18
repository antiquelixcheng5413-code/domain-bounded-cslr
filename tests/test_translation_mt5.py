"""Unit tests for the mT5-backed Chinese decoder adapter.

These run against a tiny synthetic fake T5 object so they do not require the
pretrained weights (which live on the D-drive cache) or a network.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from cslr.translation.mt5_decoder import MT5ChineseDecoder


class _FakeShared(nn.Module):
    def __init__(self, n_e: int, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_e, d))  # [V, d]


class _FakeDecoder(nn.Module):
    def __init__(self, shared: _FakeShared) -> None:
        super().__init__()
        self.shared = shared

    def forward(self, input_ids, encoder_hidden_states=None, encoder_attention_mask=None):
        hidden = torch.nn.functional.embedding(
            input_ids.to(torch.long), self.shared.weight.to(input_ids.device)
        )  # [B, L, d]
        return (hidden,)


class _FakeT5(nn.Module):
    def __init__(self, shared: _FakeShared) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=shared.weight.shape[1], vocab_size=shared.weight.shape[0])
        self.shared = shared
        self.decoder = _FakeDecoder(shared)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 0

    def __init__(self) -> None:
        self.id2c = {0: "", 1: "", 2: "我", 3: "爱", 4: "好", 5: "你"}
        self.c2id = {v: k for k, v in self.id2c.items() if v}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.c2id.get(ch, 0) for ch in list(text)]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self.id2c.get(int(i), "") for i in ids)


def _make_decoder(device: str = "cpu") -> MT5ChineseDecoder:
    d, vocab = 16, 64
    shared = _FakeShared(vocab, d)
    t5 = _FakeT5(shared)
    return MT5ChineseDecoder(t5, _FakeTokenizer(), visual_dim=8, device=device)


class MT5DecoderTest(unittest.TestCase):
    def test_teacher_forcing_shape(self) -> None:
        dec = _make_decoder()
        vt = torch.randn(2, 5, 8)
        vm = torch.ones(2, 5, dtype=torch.bool)
        tgt = dec._encode_targets(["我爱", "你好"])
        logits, meta = dec.decode_from_visual(vt, vm, tgt)
        self.assertEqual(logits.shape, (2, tgt.shape[1], 64))
        self.assertIn("logits", meta)

    def test_encode_targets_bos_eos(self) -> None:
        dec = _make_decoder()
        tgt = dec._encode_targets(["我爱"])
        # decoder start token = pad(0); last = eos(1); about the padded width
        self.assertTrue(tgt.shape[1] >= 3)
        self.assertEqual(int(tgt[0, 0]), 0)
        self.assertEqual(int(tgt[0, -1]), 1)

    def test_generate_returns_per_sample_strings(self) -> None:
        dec = _make_decoder()
        vt = torch.randn(2, 5, 8)
        vm = torch.ones(2, 5, dtype=torch.bool)
        texts = dec.generate(vt, vm, max_len=6)
        self.assertEqual(len(texts), 2)
        self.assertTrue(all(isinstance(t, str) for t in texts))

    def test_forward_updates_params(self) -> None:
        dec = _make_decoder()
        vt = torch.randn(2, 5, 8)
        vm = torch.ones(2, 5, dtype=torch.bool)
        tgt = dec._encode_targets(["我爱"])
        criterion = nn.CrossEntropyLoss(ignore_index=dec.pad_id)
        opt = torch.optim.SGD(dec.parameters(), lr=0.01)
        logits, _ = dec.decode_from_visual(vt, vm, tgt)
        loss = criterion(logits[:, :-1].reshape(-1, 64), tgt[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        self.assertTrue(loss.item() > 0)

    def test_freeze_cross_only_keeps_only_visual_proj(self) -> None:
        dec = _make_decoder()
        dec.set_freeze_mode("cross_only")
        # no EncDecAttention params exist in the fake -> entire LM path frozen
        self.assertFalse(dec.t5.shared.weight.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in dec.t5.parameters()))
        self.assertTrue(all(p.requires_grad for p in dec.visual_proj.parameters()))
        # back to none: shared embedding learnable again
        dec.set_freeze_mode("none")
        self.assertTrue(dec.t5.shared.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in dec.parameters()))

    def test_freeze_unknown_mode_raises(self) -> None:
        dec = _make_decoder()
        with self.assertRaises(ValueError):
            dec.set_freeze_mode("bogus")

    def test_visual_text_align_scalar_and_trainable(self) -> None:
        dec = _make_decoder()
        vt = torch.randn(2, 5, 8)
        vm = torch.ones(2, 5, dtype=torch.bool)
        tgt = dec._encode_targets(["我爱", "你好"])
        loss = dec.visual_text_align(vt, vm, tgt)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(-1.0 <= loss.item() <= 2.0)
        opt = torch.optim.SGD(dec.visual_proj.parameters(), lr=0.1)
        loss.backward()
        opt.step()
        self.assertIsNotNone(dec.visual_proj[1].weight.grad)


if __name__ == "__main__":
    unittest.main()