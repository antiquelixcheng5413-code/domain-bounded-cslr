#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gloss -> 中文句子：本地 Qwen2.5-1.5B-Instruct（非 oracle 端到端链）。

复用 scripts/route_a_gloss_llm.py 的 prompt 与预处理思路：
    video -> 识别器(gloss) -> Qwen2.5-1.5B -> 通顺中文句子
单例惰性加载，避免每个请求重载 ~3GB 权重。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

QWEN_PATH = os.getenv(
    "CSLR_QWEN_WEIGHTS",
    "/mnt/d/part3_models/ms_cache/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master",
)
DEVICE_ENV = "CSLR_QWEN_DEVICE"

PROMPT = (
    "你是一位中文手语翻译。请把下面的手语手势记号序列(Gloss)翻译成一个通顺、完整、"
    "符合中文语法的句子。只输出翻译结果本身，不要解释、不要加引号。\n"
    "Gloss: {gloss}\n中文翻译："
)


def _gloss_to_input(raw: str) -> str:
    parts = [p for p in raw.split("/") if p.strip()]
    parts = [p for p in parts if p not in ("。", "，")]
    return " ".join(parts)


class GlossTranslator:
    """单例：本地 Qwen2.5-1.5B，gloss -> 中文句子。"""

    _instance = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.device = self._resolve_device()
        self._tok = None
        self._model = None
        self.ready = False
        self.error: str | None = None
        self._gen_lock = threading.Lock()
        if not Path(QWEN_PATH).exists():
            self.error = "missing Qwen weights: " + QWEN_PATH
            return
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001
            self.error = f"Qwen load failed: {exc}"

    @staticmethod
    def _resolve_device() -> str:
        dev = os.getenv(DEVICE_ENV, "auto")
        if dev and dev != "auto":
            return dev
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                QWEN_PATH, trust_remote_code=True, dtype=dtype
            )
            .to(self.device)
            .eval()
        )
        self.ready = True

    @classmethod
    def get(cls) -> "GlossTranslator":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def translate(self, gloss: str) -> str:
        if not self.ready:
            return ""
        g = _gloss_to_input(gloss)
        if not g.strip():
            return ""
        prompt = PROMPT.format(gloss=g)
        msgs = [{"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        import torch

        inputs = self._tok(text, return_tensors="pt").to(self.device)
        with self._gen_lock, torch.no_grad():
            gen = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        out = self._tok.decode(
            gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return out.strip()


def create_translator() -> GlossTranslator:
    return GlossTranslator.get()
