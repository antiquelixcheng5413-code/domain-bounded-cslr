"""mT5-decoder-backed Chinese decoder adapter.

Uses a pretrained mT5 decoder as the language prior: the fusion visual tokens
are fed as cross-attention memory; the target side is subword-tokenized with
the mT5 tokenizer, and the LM head is the shared embedding transpose (tied).

The pretrained weights are loaded once at construction (a ``T5Model`` and a
``T5Tokenizer``); training only updates the visual projection layer and the
decoder weights on top of the pretrained checkpoint on the CE-CSL split.
"""

from __future__ import annotations

import torch
from torch import nn

from cslr.translation.decoder import ChineseDecoder


class MT5ChineseDecoder(ChineseDecoder):
    def __init__(
        self,
        t5_model,
        tokenizer,
        visual_dim: int,
        max_target_tokens: int = 64,
        dropout: float = 0.0,
        device: str = "cpu",
        freeze: str = "none",
    ) -> None:
        super().__init__()
        self.t5 = t5_model
        self.tokenizer = tokenizer
        self.visual_dim = visual_dim
        self.max_target_tokens = max_target_tokens

        self.d_model = t5_model.config.d_model
        self.vocab_size = t5_model.config.vocab_size
        self.pad_id = int(tokenizer.pad_token_id)
        self.eos_id = int(tokenizer.eos_token_id)
        # mT5 uses <pad> as the decoder start token; expose as bos for contract.
        self.bos_id = self.pad_id
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, self.d_model),
        )
        self.set_freeze_mode(freeze)

        dev = torch.device(device)
        self.to(dev)
        self.device = dev

    def set_freeze_mode(self, mode: str) -> None:
        """Control which mT5 parameters are trainable.

        ``none``: train everything (default; pairs a strong pretrained LM prior
        with a tiny data split, which can degenerate to a pure-LM collapse).

        ``cross_only``: freeze the language path (shared embedding, self-attn,
        feed-forward, layer norms) and train only the visual projection plus the
        decoder cross-attention.  This forces visual conditioning and prevents
        the decoder from ignoring its encoder memory.
        """
        if mode == "none":
            for p in self.t5.parameters():
                p.requires_grad = True
            return
        if mode == "cross_only":
            for name, p in self.t5.named_parameters():
                # EncDecAttention is the cross-attention between decoder and the
                # visual memory; everything else is the fixed language prior.
                p.requires_grad = "EncDecAttention" in name
            return
        raise ValueError(f"unknown mt5 freeze mode {mode!r} (expected 'none'|'cross_only')")

    def _lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        # Tied weight-sharing LM head (transpose of the shared embedding).
        w = self.t5.shared.weight  # [V, d_model]
        return torch.matmul(hidden, w.t())  # [B, L, V]

    def encode_visual(
        self, visual_tokens: torch.Tensor, visual_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        enc = self.dropout(self.visual_proj(visual_tokens.to(self.device)))
        if visual_mask is not None:
            visual_mask = visual_mask.to(self.device)
        return enc, visual_mask

    def forward_teacher_forcing(
        self,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor | None,
        target_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        enc, enc_mask = self.encode_visual(visual_tokens, visual_mask)
        out = self.t5.decoder(
            input_ids=target_ids.to(self.device),
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_mask,
        )[0]  # [B, L, d_model]
        logits = self._lm_head(out)
        return {"logits": logits}

    def visual_text_align(
        self,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor | None,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine distance between the mean projected visual repr and the mean
        target-token embedding — a direct visual<->text supervision signal that
        does not flow through the decoder's LM prior."""
        enc, enc_mask = self.encode_visual(visual_tokens, visual_mask)
        if enc_mask is None:
            enc_mask = torch.ones(enc.shape[:2], dtype=torch.bool, device=enc.device)
        valid = enc_mask.unsqueeze(-1).to(enc.dtype)
        h_v = (enc * valid).sum(1) / valid.sum(1).clamp(min=1.0)  # [B, d]
        emb = self.t5.shared.weight.to(target_ids.device)  # [V, d]
        tgt_emb = emb[target_ids]  # [B, L, d]
        tgt_mask = (target_ids != self.pad_id).unsqueeze(-1).to(tgt_emb.dtype)
        h_t = (tgt_emb * tgt_mask).sum(1) / tgt_mask.sum(1).clamp(min=1.0)  # [B, d]
        return 1.0 - torch.nn.functional.cosine_similarity(h_v, h_t, dim=-1).mean()

    def _encode_targets(self, texts: list[str]) -> torch.Tensor:
        """Subword-encode sentences; decoder_start (pad) prepended, eos appended."""
        batch: list[list[int]] = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            batch.append([self.bos_id] + ids[: self.max_target_tokens - 2] + [self.eos_id])
        max_len = max(len(s) for s in batch)
        padded = torch.full(
            (len(batch), max_len), self.pad_id, dtype=torch.long, device=self.device
        )
        for i, seq in enumerate(batch):
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
        return padded

    def generate(
        self,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor | None,
        vocab: dict[str, int] | None = None,
        max_len: int | None = None,
        beam_size: int = 1,
    ) -> list[str]:
        del vocab, beam_size  # mT5 decodes through its tokenizer, greedy by default
        max_len = max_len or self.max_target_tokens
        enc, enc_mask = self.encode_visual(visual_tokens, visual_mask)
        batch_size = visual_tokens.shape[0]
        dev = self.device
        out_ids = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=dev)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=dev)

        with torch.no_grad():
            for _ in range(max_len):
                out = self.t5.decoder(
                    input_ids=out_ids,
                    encoder_hidden_states=enc,
                    encoder_attention_mask=enc_mask,
                )[0]
                logits = self._lm_head(out)[:, -1]  # [B, V]
                next_id = torch.argmax(logits, dim=-1)
                next_id = torch.where(
                    finished,
                    torch.tensor(self.eos_id, device=dev),
                    next_id,
                )
                out_ids = torch.cat([out_ids, next_id.unsqueeze(1)], dim=1)
                finished = finished | (next_id == self.eos_id)
                if bool(finished.all()):
                    break

        texts: list[str] = []
        for row in out_ids:
            ids = [t for t in row.tolist() if t != self.pad_id and t != self.bos_id]
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True).replace(" ", ""))
        return texts

    def decode_from_visual(
        self,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor | None,
        target_ids: torch.Tensor | None = None,
        max_len: int | None = None,
        vocab: dict[str, int] | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if target_ids is not None:
            result = self.forward_teacher_forcing(visual_tokens, visual_mask, target_ids)
            return result["logits"], result
        return None, {}