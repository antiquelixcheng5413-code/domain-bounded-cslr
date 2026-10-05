"""New-route recognition service for the FastAPI backend.

This replaces the legacy LSTM-onnx path with the current best model chain:

    uploaded video -> Qwen2.5-VL vision tower (VL48) -> CTC -> gloss -> Qwen2.5-1.5B -> 中文

The frozen ``test`` split is never read (this service only ingests user uploads).
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


class VL48CtcLlmsService:
    """Video -> VL48 features -> CTC gloss -> Chinese via local models."""

    def __init__(
        self,
        vision_model: str | None = None,
        checkpoint: str | None = None,
        llm_model: str | None = None,
        frames: int = 48,
        beam_width: int = 1,
    ) -> None:
        self.frames = frames
        self.beam_width = beam_width
        self.vision_model = vision_model or os.getenv(
            "CSLR_VISION_MODEL",
            "/mnt/d/cslr-tools/models/Qwen2.5-VL-3B-Instruct",
        )
        self.checkpoint = Path(
            checkpoint
            or os.getenv(
                "CSLR_CTC_CHECKPOINT",
                str(REPO / "artifacts/checkpoints/ctc-vl48-cap300.pt"),
            )
        )
        self.llm_model = llm_model or os.getenv(
            "CSLR_LLM_MODEL",
            "/mnt/d/part3_models/ms_cache/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master",
        )
        self.model_error: str | None = self._check_resources()

        # lazy-loaded, so /health works even before models are up
        self._encoder = None
        self._recognizer = None
        self._vocabulary = None
        self._normalizer = None
        self._llm = None

    def _check_resources(self) -> str | None:
        """Report the first missing asset, so the service can surface readiness honestly."""
        from pathlib import Path

        missing = [str(path) for path in (self.checkpoint, Path(self.vision_model), Path(self.llm_model)) if not path.exists()]
        if not missing:
            return None
        return "missing model assets: " + ", ".join(missing)

    # ---- loading ---------------------------------------------------------
    def _load_encoder(self):
        if self._encoder is not None:
            return self._encoder
        import torch
        from cslr.recognition.qwen_vl_features import FrozenQwenVisionEncoder

        if not Path(self.vision_model).exists():
            raise FileNotFoundError(f"vision model missing: {self.vision_model}")
        self._encoder = FrozenQwenVisionEncoder(self.vision_model)
        return self._encoder

    def _load_recognizer(self):
        if self._recognizer is not None:
            return self._recognizer, self._vocabulary, self._normalizer
        from cslr.recognition.inference import load_recognizer

        if not self.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint missing: {self.checkpoint}")
        model, vocabulary, normalizer, _config, _device = load_recognizer(self.checkpoint, "auto")
        self._recognizer, self._vocabulary, self._normalizer = model, vocabulary, normalizer
        return model, vocabulary, normalizer

    def _load_llm(self):
        if self._llm is not None:
            return self._llm
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not Path(self.llm_model).exists():
            raise FileNotFoundError(f"llm missing: {self.llm_model}")
        self._tok = AutoTokenizer.from_pretrained(self.llm_model, trust_remote_code=True)
        self._llm = (
            AutoModelForCausalLM.from_pretrained(
                self.llm_model, trust_remote_code=True, torch_dtype="auto"
            )
            .to("cuda" if torch.cuda.is_available() else "cpu")
            .eval()
        )
        return self._llm

    # ---- pipeline ---------------------------------------------------------
    def _prompt(self, gloss: str) -> str:
        return (
            "你是一位中文手语翻译。请把下面的手语手势记号序列(Gloss)翻译成一个通顺、完整、"
            "符合中文语法的句子。只输出翻译结果本身，不要解释、不要加引号。\n"
            f"Gloss: {gloss}\n中文翻译："
        )

    def predict_video(self, video_path: Path) -> dict:
        started = time.perf_counter()
        latency: dict[str, float] = {}
        if self.model_error is not None:
            return {
                "status": "model_unavailable",
                "label": "unknown",
                "gloss_tokens": [],
                "intent": "unknown",
                "gloss": "UNKNOWN",
                "text_zh": "模型尚未安装或训练，当前不能进行真实识别。",
                "confidence": 0.0,
                "top_k": [],
                "warnings": [self.model_error],
                "latency_ms": {"total": _elapsed_ms(started)},
                "model_version": "vl48-ctc-qwen1.5b",
            }
        try:
            encoder_started = time.perf_counter()
            encoder = self._load_encoder()
            from cslr.recognition.clip_features import sample_frames

            frames = sample_frames(video_path, self.frames)
            features = encoder.encode_frames(frames)  # [T, D]
            latency["extract_features"] = _elapsed_ms(encoder_started)

            decode_started = time.perf_counter()
            model, vocabulary, normalizer = self._load_recognizer()
            feature_view = "full"
            from cslr.recognition.dataset import feature_view_indices

            columns = feature_view_indices(feature_view)
            if columns is not None:
                features = features[:, columns]
            if normalizer is not None:
                features = normalizer.apply(features)
            if features.shape[0] == 0:
                raise ValueError("no frames extracted")
            import torch
            from cslr.recognition.decode import (
                classes_to_token_ids,
                greedy_decode,
                prefix_beam_search,
            )
            from cslr.recognition.model import BLANK_INDEX

            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = torch.from_numpy(features[None].astype(np.float32)).to(torch_device)
            lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=torch_device)
            with torch.no_grad():
                logits = model(tensor, lengths)
                output_lengths = model.output_lengths(lengths)
                steps = torch.log_softmax(logits.float(), dim=-1)[0, : int(output_lengths[0])].cpu().numpy()
            result = (
                prefix_beam_search(steps, beam_width=self.beam_width)
                if self.beam_width > 1
                else greedy_decode(steps)
            )
            tokens = vocabulary.decode(classes_to_token_ids(result.classes))
            gloss = "/".join(t for t in tokens if t != "<unk>")
            latency["ctc_decode"] = _elapsed_ms(decode_started)

            llm_started = time.perf_counter()
            text_zh = self._translate(gloss)
            latency["llm_generate"] = _elapsed_ms(llm_started)

            latency["total"] = _elapsed_ms(started)
            return {
                "status": "ok",
                "label": gloss or "unknown",
                "gloss_tokens": [t for t in tokens if t != "<unk>"],
                "intent": gloss or "unknown",
                "gloss": gloss or "unknown",
                "text_zh": text_zh,
                "confidence": 1.0,
                "top_k": [],
                "warnings": [],
                "latency_ms": latency,
                "model_version": "vl48-ctc-qwen1.5b",
            }
        except Exception as exc:  # noqa: BLE001 - report upstream
            latency["total"] = _elapsed_ms(started)
            return {
                "status": "error",
                "label": "unknown",
                "gloss_tokens": [],
                "intent": "unknown",
                "gloss": "UNKNOWN",
                "text_zh": f"识别失败：{exc}",
                "confidence": 0.0,
                "top_k": [],
                "warnings": [str(exc)],
                "latency_ms": latency,
                "model_version": "vl48-ctc-qwen1.5b",
            }

    def _translate(self, gloss: str) -> str:
        if not gloss.strip():
            return "（未能识别出有效的手势序列）"
        llm = self._load_llm()
        import torch

        prompt = self._prompt(gloss)
        msgs = [{"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(llm.device)
        gen = llm.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            pad_token_id=self._tok.eos_token_id,
        )
        pred = self._tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return pred.strip()

    @property
    def ready(self) -> bool:
        return self.model_error is None

    @property
    def demo_mode(self) -> bool:
        return False


def create_vl48_cslr_service() -> VL48CtcLlmsService:
    return VL48CtcLlmsService()