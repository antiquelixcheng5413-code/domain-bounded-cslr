"""Replaceable Chinese decoder interface with a tiny Transformer decoder.

A configurable Gloss auxiliary head is attached (default weight 0.0); it is
used only as optional auxiliary supervision and never required at inference.
"""

from __future__ import annotations

import torch
from torch import nn

from cslr.translation.text import character_tokenize


class ChineseDecoder(nn.Module):
    vocab_size: int
    bos_id: int
    eos_id: int
    pad_id: int

    def decode_from_visual(
        self,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor | None,
        target_ids: torch.Tensor | None = None,
        max_len: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        raise NotImplementedError


class TinyTransformerChineseDecoder(ChineseDecoder):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        max_target_len: int = 128,
        dropout: float = 0.1,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        gloss_vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_target_len = max_target_len

        self.tok_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_target_len, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        self.gloss_head = nn.Linear(hidden_dim, gloss_vocab_size) if gloss_vocab_size else None

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        # lower-triangular; True = ignore (upper triangle)
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0).expand(ids.shape[0], -1)
        return self.tok_embed(ids) + self.pos_embed(pos)

    def forward_teacher_forcing(
        self,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor | None,
        target_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tgt_emb = self._embed_tokens(target_ids)
        tgt_mask = self._causal_mask(target_ids.shape[1], target_ids.device)
        tgt_key_padding = target_ids == self.pad_id

        memory_key_padding = ~visual_mask.to(tgt_emb.dtype).bool() if visual_mask is not None else None

        out = self.decoder(
            tgt_emb,
            visual_tokens,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding if tgt_key_padding.any() else None,
            memory_key_padding_mask=memory_key_padding,
        )
        logits = self.output_proj(out)
        return {"logits": logits}

    def generate(
        self,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor | None,
        vocab: dict[str, int],
        max_len: int | None = None,
        beam_size: int = 1,
    ) -> list[str]:
        if beam_size > 1:
            return self.generate_beam(visual_tokens, visual_mask, vocab, max_len=max_len, beam_size=beam_size)
        max_len = max_len or self.max_target_len
        batch_size = visual_tokens.shape[0]
        device = visual_tokens.device
        out_ids = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        memory_key_padding = ~visual_mask.bool() if visual_mask is not None else None

        id_to_char = {v: k for k, v in vocab.items()}
        for _ in range(max_len):
            emb = self._embed_tokens(out_ids)
            tgt_mask = self._causal_mask(out_ids.shape[1], device)
            decoded = self.decoder(emb, visual_tokens, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding)
            logits = self.output_proj(decoded[:, -1])  # [B, V]
            if out_ids.shape[1] >= 2:
                seq = out_ids.tolist()
                prev = [r[-1] for r in seq]
                banned = [set(zip(r[:-1], r[1:])) for r in seq]
                for b in range(batch_size):
                    if finished[b]:
                        continue
                    for nid in range(logits.shape[1]):
                        if (prev[b], nid) in banned[b]:
                            logits[b, nid] = float("-inf")
            next_id = torch.argmax(logits, dim=-1)  # greedy
            next_id = torch.where(finished, torch.tensor(self.eos_id, device=device), next_id)
            out_ids = torch.cat([out_ids, next_id.unsqueeze(1)], dim=1)
            finished = finished | (next_id == self.eos_id)
            if bool(finished.all()):
                break

        texts: list[str] = []
        for row in out_ids:
            chars: list[str] = []
            for token in row.tolist():
                if token == self.bos_id or token == self.pad_id:
                    continue
                if token == self.eos_id:
                    break
                chars.append(id_to_char.get(token, ""))
            texts.append("".join(chars))
        return texts

    def generate_beam(
        self,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor | None,
        vocab: dict[str, int],
        max_len: int | None = None,
        beam_size: int = 4,
    ) -> list[str]:
        """Beam-search decoding, per sample. beam_size=1 degenerates to greedy."""
        max_len = max_len or self.max_target_len
        id_to_char = {v: k for k, v in vocab.items()}
        device = visual_tokens.device
        outputs: list[str] = []
        for si in range(visual_tokens.shape[0]):
            visual = visual_tokens[si : si + 1]
            mem_key = (
                ~visual_mask[si : si + 1].bool()
                if visual_mask is not None else None
            )
            beams: list[tuple[list[int], float]] = [([self.bos_id], 0.0)]
            completed: list[tuple[list[int], float]] = []
            for _ in range(max_len):
                if not beams:
                    break
                lengths = [len(seq) for seq, _ in beams]
                width = max(lengths)
                ids = torch.full(
                    (len(beams), width), self.pad_id, dtype=torch.long, device=device
                )
                for i, (seq, _sc) in enumerate(beams):
                    ids[i, : len(seq)] = torch.tensor(seq, device=device)
                emb = self._embed_tokens(ids)
                tgt_mask = self._causal_mask(width, device)
                decoded = self.decoder(
                    emb, visual, tgt_mask=tgt_mask, memory_key_padding_mask=mem_key
                )
                logp = torch.log_softmax(self.output_proj(decoded[:, -1]), dim=-1)

                candidates: list[tuple[list[int], float]] = []
                for i, (seq, score) in enumerate(beams):
                    if seq[-1] == self.eos_id:
                        candidates.append((seq, score))
                        continue
                    topk = torch.topk(logp[i], min(beam_size, logp.shape[1]))
                    for tok, delta in zip(
                        topk.indices.tolist(), topk.values.tolist()
                    ):
                        candidates.append((seq + [tok], score + delta))

                candidates.sort(key=lambda x: -x[1])
                beams = []
                seen: set[tuple[int, ...]] = set()
                for seq, sc in candidates:
                    key = tuple(seq)
                    if key in seen:
                        continue
                    seen.add(key)
                    beams.append((seq, sc))
                    if len(beams) >= beam_size:
                        break

                completed += [b for b in beams if b[0][-1] == self.eos_id]
                beams = [b for b in beams if b[0][-1] != self.eos_id]

            if not completed:
                completed = [max(beams, key=lambda x: x[1])] if beams else []
            if not completed:  # empty decode fallback
                best_seq: list[int] = [self.bos_id]
            else:
                best_seq = sorted(completed, key=lambda x: -x[1])[0][0]
            chars: list[str] = []
            for token in best_seq:
                if token == self.bos_id or token == self.pad_id:
                    continue
                if token == self.eos_id:
                    break
                chars.append(id_to_char.get(token, ""))
            outputs.append("".join(chars))
        return outputs

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