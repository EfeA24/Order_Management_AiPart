from __future__ import annotations

import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import DEVICE, TranslationConfig, get_translation_config

_PASSTHROUGH_RE = re.compile(
    r"^\s*("
    r"[-+]?\d+([.,]\d+)?"
    r"|[A-Za-z0-9]+[-_/][A-Za-z0-9][-_/A-Za-z0-9]*"
    r"|\d+\s*[xX×]\s*\d+(\s*[xX×]\s*\d+)?"
    r"|\d+([.,]\d+)?\s*(mm|cm|m|kg|g|lt|l|pcs|adet)?"
    r")\s*$",
    re.IGNORECASE,
)


def should_passthrough(text: str) -> bool:
    if not text or not text.strip():
        return True
    return bool(_PASSTHROUGH_RE.match(text.strip()))


class Translator:
    def __init__(
        self,
        model_path: str | Path | None = None,
        config: TranslationConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or get_translation_config()
        self.device = device or DEVICE
        source = str(model_path) if model_path else self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(source)
        self.model.to(self.device)
        self.model.eval()

    def translate(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        results: list[str | None] = [None] * len(texts)
        to_translate: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            if should_passthrough(text):
                results[i] = text
            else:
                to_translate.append((i, text))
        batch_size = self.config.batch_size
        for start in range(0, len(to_translate), batch_size):
            chunk = to_translate[start : start + batch_size]
            indices = [i for i, _ in chunk]
            batch_texts = [t for _, t in chunk]
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_length=self.config.max_length,
                    num_beams=4,
                )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for idx, out in zip(indices, decoded, strict=True):
                results[idx] = out
        return [r if r is not None else "" for r in results]


def load_translation_model(model_path=None):
    return Translator(model_path=model_path)


def translate(text: str, model=None, tokenizer=None) -> str:
    if isinstance(model, Translator):
        return model.translate([text])[0]
    return Translator().translate([text])[0]
