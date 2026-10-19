from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import (
    CELL_MARKER,
    DEVICE,
    EXTRACTION_LABELS,
    ROW_MARKER,
    ExtractionConfig,
    get_extraction_config,
)


class OrderExtractor:
    def __init__(
        self,
        model_path: str | Path | None = None,
        config: ExtractionConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or get_extraction_config()
        self.device = device or DEVICE
        source = str(model_path) if model_path else self.config.model_name
        label_path = Path(source) / "label_map.json"
        if label_path.exists():
            with label_path.open(encoding="utf-8") as f:
                mapping = json.load(f)
            self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
            self.label2id = mapping["label2id"]
        else:
            self.label2id = {label: i for i, label in enumerate(self.config.labels)}
            self.id2label = {i: label for label, i in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        if ROW_MARKER not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [ROW_MARKER, CELL_MARKER]}
            )
        self.model = AutoModelForTokenClassification.from_pretrained(
            source,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()

    def extract(
        self, text: str, cell_spans: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        cell_spans = cell_spans or []
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=-1).tolist()
            confs = probs.max(dim=-1).values.tolist()
        entities: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for idx, label_id in enumerate(pred_ids):
            start, end = offsets[idx]
            if start == end:
                continue
            label = self.id2label.get(label_id, "O")
            conf = float(confs[idx])
            if label.startswith("B-"):
                if current:
                    entities.append(current)
                etype = label[2:]
                current = {
                    "type": etype,
                    "start": start,
                    "end": end,
                    "confidence": conf,
                }
            elif label.startswith("I-") and current and current["type"] == label[2:]:
                current["end"] = end
                current["confidence"] = min(current["confidence"], conf)
            else:
                if current:
                    entities.append(current)
                    current = None
        if current:
            entities.append(current)
        results: list[dict[str, Any]] = []
        for ent in entities:
            if ent["type"] not in {"HEADER", "PRODUCT"}:
                continue
            span_start, span_end = ent["start"], ent["end"]
            matched_cell = None
            for cell in cell_spans:
                if span_start >= cell["start"] and span_end <= cell["end"]:
                    matched_cell = cell
                    break
                if span_start < cell["end"] and span_end > cell["start"]:
                    span_start = max(span_start, cell["start"])
                    span_end = min(span_end, cell["end"])
                    matched_cell = cell
                    break
            if matched_cell is None:
                continue
            piece = text[span_start:span_end].strip()
            if not piece:
                continue
            results.append(
                {
                    "type": ent["type"],
                    "text": piece,
                    "cell_id": matched_cell["cell_id"],
                    "confidence": ent["confidence"],
                    "start": span_start,
                    "end": span_end,
                }
            )
        return results

    def save_label_map(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "labels": EXTRACTION_LABELS,
        }
        with (directory / "label_map.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def load_extraction_model(model_path=None):
    return OrderExtractor(model_path=model_path)


def extract_entities(text: str, model=None, tokenizer=None) -> list:
    if isinstance(model, OrderExtractor):
        return model.extract(text)
    return OrderExtractor().extract(text)
