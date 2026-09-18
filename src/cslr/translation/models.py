"""Whole-pipeline Part 3 model assembly with a unified output contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cslr.translation.dataset import collate_part3
from cslr.translation.decoder import TinyTransformerChineseDecoder
from cslr.translation.fusion import LightSpaMoFusion
from cslr.translation.text import build_vocab


@dataclass
class TranslationModelOutput:
    loss: torch.Tensor | None = None
    translation_loss: torch.Tensor | None = None
    gloss_aux_loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    generated_texts: list[str] | None = None
    generated_token_ids: torch.Tensor | None = None
    visual_token_count: torch.Tensor | None = None
    stats: dict[str, float] | None = None


class Part3SpaMoModel(nn.Module):
    def __init__(
        self,
        *,
        modalities: tuple[str, ...],
        hidden_dim: int,
        fusion_heads: int,
        fusion_layers: int,
        fusion_ff: int,
        dropout: float,
        vocab: dict[str, int],
        gloss_vocab_size: int | None,
        max_target_len: int = 128,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        decoder_ff: int | None = None,
        gloss_aux_weight: float = 0.0,
        feature_dim: dict[str, int] | None = None,
        fusion_num_pool_tokens: int | None = None,
        fusion_align_frames: bool = False,
        label_smoothing: float = 0.0,
        beam_size: int = 1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gloss_aux_weight = gloss_aux_weight
        self.device_device = torch.device(device)
        if feature_dim is None:
            feature_dim = {"rgb": hidden_dim, "motion": hidden_dim, "landmark": 368}

        self.fusion = LightSpaMoFusion(
            modalities=modalities,
            hidden_dim=hidden_dim,
            num_heads=fusion_heads,
            num_layers=fusion_layers,
            feedforward_dim=fusion_ff,
            dropout=dropout,
            feature_dim=feature_dim,
            num_pool_tokens=fusion_num_pool_tokens,
            align_frames=fusion_align_frames,
        )
        self.decoder = TinyTransformerChineseDecoder(
            vocab_size=self.vocab_size,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            feedforward_dim=decoder_ff or hidden_dim * 4,
            max_target_len=max_target_len,
            dropout=dropout,
            gloss_vocab_size=gloss_vocab_size,
        ).to(self.device_device)
        self.to(self.device_device)
        self.label_smoothing = label_smoothing
        self.beam_size = beam_size
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.decoder.pad_id, label_smoothing=label_smoothing
        )

    def label_to_target_ids(self, texts: list[str]) -> torch.Tensor:
        import torch as _t

        unk = self.vocab["<unk>"]
        bos, eos = self.vocab["<bos>"], self.vocab["<eos>"]
        pad = self.vocab["<pad>"]
        sequences: list[list[int]] = []
        for text in texts:
            ids = [bos] + [self.vocab.get(ch, unk) for ch in list(text) if ch != " "] + [eos]
            sequences.append(ids)
        max_len = max(len(s) for s in sequences)
        padded = [_t.full((max_len,), pad, dtype=_t.long) for _ in sequences]
        for i, seq in enumerate(sequences):
            padded[i][: len(seq)] = _t.tensor(seq, dtype=_t.long)
        return _t.stack(padded).to(self.device_device)

    def extract_modality_features(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {mod: t.to(self.device_device) for mod, t in features.items() if t is not None}

    def forward(
        self,
        features: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None,
        target_texts: list[str] | None = None,
        generate: bool = False,
        max_gen_len: int | None = None,
        beam_size: int | None = None,
    ) -> TranslationModelOutput:
        active = {k: v for k, v in features.items() if v is not None}
        if not active:
            raise ValueError("no input modalities provided to the model")

        visual_tokens, visual_mask, token_count = self.fusion(active, masks)
        target_ids = self.label_to_target_ids(target_texts) if target_texts else None

        loss = None
        translation_loss = None
        gloss_aux_loss = None
        logits = None
        generated_texts = None
        generated_token_ids = None

        if target_ids is not None:
            dec_meta = self.decoder.decode_from_visual(visual_tokens, visual_mask, target_ids)
            logits = dec_meta[0]
            if logits is not None:
                trg = target_ids[:, 1:].contiguous()
                logg = logits[:, :-1, :].reshape(-1, self.vocab_size)
                translation_loss = self.cross_entropy(logg, trg.reshape(-1))
                loss = translation_loss
                # optional gloss aux head: accept an empty heuristic (weight 0 by default)
                if self.gloss_head_enabled and self.gloss_aux_weight > 0:
                    gloss_aux_loss = torch.tensor(0.0, device=self.device_device)
                    loss = translation_loss + self.gloss_aux_weight * gloss_aux_loss

        if generate:
            bz = beam_size if beam_size is not None else self.beam_size
            generated_texts = self.decoder.generate(
                visual_tokens, visual_mask, self.vocab, max_len=max_gen_len, beam_size=bz
            )

        return TranslationModelOutput(
            loss=loss,
            translation_loss=translation_loss,
            gloss_aux_loss=gloss_aux_loss,
            logits=logits,
            generated_texts=generated_texts,
            visual_token_count=torch.tensor([token_count], dtype=torch.long, device=self.device_device),
        )

    @property
    def gloss_head_enabled(self) -> bool:
        return getattr(self.decoder, "gloss_head", None) is not None


def build_model_from_config(
    cfg, vocab: dict[str, int], *, gloss_vocab_size: int | None = None,
    feature_dim: dict[str, int] | None = None,
) -> Part3SpaMoModel:
    return Part3SpaMoModel(
        modalities=cfg.fusion.modalities,
        hidden_dim=cfg.model.hidden_dim,
        fusion_heads=cfg.fusion.num_heads,
        fusion_layers=cfg.fusion.num_layers,
        fusion_ff=cfg.fusion.feedforward_dim,
        dropout=cfg.model.dropout,
        vocab=vocab,
        gloss_vocab_size=gloss_vocab_size,
        max_target_len=cfg.model.decoder.max_target_len,
        decoder_layers=cfg.model.decoder.num_layers,
        decoder_heads=cfg.model.decoder.num_heads,
        decoder_ff=cfg.model.decoder.feedforward_dim,
        gloss_aux_weight=cfg.model.gloss_aux_weight,
        feature_dim=feature_dim,
        fusion_num_pool_tokens=getattr(cfg.fusion, "num_pool_tokens", None),
        fusion_align_frames=getattr(cfg.fusion, "align_frames", False),
        label_smoothing=getattr(cfg.model, "label_smoothing", 0.0),
        beam_size=getattr(cfg.model, "beam_size", 1),
        device=cfg.device,
    )


def build_vocab_from_dataset(dataset) -> dict[str, int]:
    sentences = [str(dataset.sentences[r.sample_id].chinese_sentences) for r in dataset.records]
    return build_vocab(sentences)