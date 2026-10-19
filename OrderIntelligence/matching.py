from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DEVICE, MatchingConfig, get_matching_config


class HeaderMatcher:
    def __init__(
        self,
        model_path: str | Path | None = None,
        config: MatchingConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or get_matching_config()
        self.device = device or DEVICE
        source = str(model_path) if model_path else self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def match(
        self,
        header_text: str,
        candidate_text: str,
        row_context: str,
        column_distance: int = 0,
    ) -> dict[str, Any]:
        text_b = (
            f"{candidate_text} [SEP] {row_context} [SEP] col_dist={column_distance}"
        )
        encoded = self.tokenizer(
            header_text,
            text_b,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1)
            score = float(probs[1].item())
            is_match = bool(score >= 0.5)
        return {
            "is_match": is_match,
            "confidence": score,
            "header_text": header_text,
            "candidate_text": candidate_text,
        }

    def match_batch(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            self.match(
                p["header_text"],
                p["candidate_text"],
                p.get("row_context", ""),
                int(p.get("column_distance", 0)),
            )
            for p in pairs
        ]


def match_header_product_value(entities: list) -> list:
    raise NotImplementedError("Use HeaderMatcher with OrderPipeline.")
