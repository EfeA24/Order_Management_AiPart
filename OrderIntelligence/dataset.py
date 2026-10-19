from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from config import (
    CELL_MARKER,
    PROCESSED_DIR,
    ROW_MARKER,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    ExtractionConfig,
    MatchingConfig,
    TranslationConfig,
    get_extraction_config,
    get_matching_config,
    get_translation_config,
)


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def load_translation_pairs(
    tr_path: Path | None = None,
    en_path: Path | None = None,
) -> list[tuple[str, str]]:
    cfg = get_translation_config()
    tr_path = tr_path or cfg.tr_file
    en_path = en_path or cfg.en_file
    if not tr_path.exists():
        raise FileNotFoundError(f"Turkish file not found: {tr_path}")
    if not en_path.exists():
        raise FileNotFoundError(f"English file not found: {en_path}")
    tr_lines = _read_lines(tr_path)
    en_lines = _read_lines(en_path)
    if len(tr_lines) != len(en_lines):
        raise ValueError(
            f"Line counts differ: tr={len(tr_lines)} en={len(en_lines)}"
        )
    return list(zip(tr_lines, en_lines, strict=True))


def clean_translation_pairs(
    pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    cleaned: list[tuple[str, str]] = []
    for tr, en in pairs:
        tr_s = tr.strip()
        en_s = en.strip()
        if not tr_s or not en_s:
            continue
        key = (tr_s, en_s)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned


def _stable_bucket(text: str, seed: int) -> float:
    digest = hashlib.md5(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_translation_pairs(
    pairs: list[tuple[str, str]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> dict[str, list[tuple[str, str]]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1, got {total}")
    by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for tr, en in pairs:
        by_source[tr].append((tr, en))
    train: list[tuple[str, str]] = []
    val: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    train_end = train_ratio
    val_end = train_ratio + val_ratio
    for tr, group in by_source.items():
        r = _stable_bucket(tr, seed)
        if r < train_end:
            train.extend(group)
        elif r < val_end:
            val.extend(group)
        else:
            test.extend(group)
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def prepare_translation_splits(
    config: TranslationConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config = config or get_translation_config()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        split: PROCESSED_DIR / f"{config.processed_prefix}_{split}.jsonl"
        for split in ("train", "val", "test")
    }
    meta_path = PROCESSED_DIR / f"{config.processed_prefix}_split_meta.json"
    if not force and all(p.exists() for p in paths.values()) and meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            return json.load(f)
    raw = load_translation_pairs(config.tr_file, config.en_file)
    cleaned = clean_translation_pairs(raw)
    splits = split_translation_pairs(cleaned)
    for split, pairs in splits.items():
        _write_jsonl(paths[split], [{"tr": tr, "en": en} for tr, en in pairs])
    meta = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "raw_pairs": len(raw),
        "cleaned_pairs": len(cleaned),
        "train_pairs": len(splits["train"]),
        "val_pairs": len(splits["val"]),
        "test_pairs": len(splits["test"]),
        "unique_tr_sources": len({tr for tr, _ in cleaned}),
        "files": {k: str(v) for k, v in paths.items()},
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _split_by_document_id(
    records: list[dict[str, Any]],
    id_key: str = "document_id",
    seed: int = SEED,
) -> dict[str, list[dict[str, Any]]]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_doc[str(rec[id_key])].append(rec)
    doc_ids = sorted(by_doc.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    n = len(doc_ids)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        only = by_doc[doc_ids[0]]
        return {"train": list(only), "val": list(only), "test": list(only)}
    if n == 2:
        return {
            "train": list(by_doc[doc_ids[0]]),
            "val": list(by_doc[doc_ids[1]]),
            "test": list(by_doc[doc_ids[1]]),
        }
    n_test = max(1, int(round(n * TEST_RATIO)))
    n_val = max(1, int(round(n * VAL_RATIO)))
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1
    n_train = n - n_test - n_val
    train_ids = doc_ids[:n_train]
    val_ids = doc_ids[n_train : n_train + n_val]
    test_ids = doc_ids[n_train + n_val :]
    train = [r for d in train_ids for r in by_doc[d]]
    val = [r for d in val_ids for r in by_doc[d]]
    test = [r for d in test_ids for r in by_doc[d]]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def prepare_extraction_splits(
    config: ExtractionConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config = config or get_extraction_config()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        split: PROCESSED_DIR / f"{config.processed_prefix}_{split}.jsonl"
        for split in ("train", "val", "test")
    }
    meta_path = PROCESSED_DIR / f"{config.processed_prefix}_split_meta.json"
    if not force and all(p.exists() for p in paths.values()) and meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            return json.load(f)
    if not config.annotations_file.exists():
        raise FileNotFoundError(
            f"Extraction annotations not found: {config.annotations_file}"
        )
    records = _read_jsonl(config.annotations_file)
    splits = _split_by_document_id(records)
    for split, rows in splits.items():
        _write_jsonl(paths[split], rows)
    meta = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "total_docs": len({r["document_id"] for r in records}),
        "train_docs": len({r["document_id"] for r in splits["train"]}),
        "val_docs": len({r["document_id"] for r in splits["val"]}),
        "test_docs": len({r["document_id"] for r in splits["test"]}),
        "files": {k: str(v) for k, v in paths.items()},
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def prepare_matching_splits(
    config: MatchingConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config = config or get_matching_config()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        split: PROCESSED_DIR / f"{config.processed_prefix}_{split}.jsonl"
        for split in ("train", "val", "test")
    }
    meta_path = PROCESSED_DIR / f"{config.processed_prefix}_split_meta.json"
    if not force and all(p.exists() for p in paths.values()) and meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            return json.load(f)
    if not config.annotations_file.exists():
        raise FileNotFoundError(
            f"Matching annotations not found: {config.annotations_file}"
        )
    records = _read_jsonl(config.annotations_file)
    splits = _split_by_document_id(records)
    for split, rows in splits.items():
        _write_jsonl(paths[split], rows)
    meta = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "total_examples": len(records),
        "train_examples": len(splits["train"]),
        "val_examples": len(splits["val"]),
        "test_examples": len(splits["test"]),
        "files": {k: str(v) for k, v in paths.items()},
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _align_labels(
    text: str,
    entities: list[dict[str, Any]],
    offset_mapping: list[tuple[int, int]],
    label2id: dict[str, int],
) -> list[int]:
    char_labels = ["O"] * len(text)
    for ent in entities:
        start, end = int(ent["start"]), int(ent["end"])
        etype = ent["type"]
        if start < 0 or end > len(text) or start >= end:
            continue
        char_labels[start] = f"B-{etype}"
        for i in range(start + 1, end):
            char_labels[i] = f"I-{etype}"
    labels: list[int] = []
    for start, end in offset_mapping:
        if start == end:
            labels.append(-100)
            continue
        span_labels = char_labels[start:end]
        label = "O"
        for candidate in span_labels:
            if candidate.startswith("B-"):
                label = candidate
                break
            if candidate.startswith("I-"):
                label = candidate
        labels.append(label2id.get(label, label2id["O"]))
    return labels


def _chunk_extraction_record(
    record: dict[str, Any],
    max_length: int,
    tokenizer,
) -> list[dict[str, Any]]:
    text = record["text"]
    entities = record.get("entities", [])
    cell_spans = record.get("cell_spans", [])
    if not cell_spans:
        return [record]
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=False,
    )
    if len(encoded["input_ids"]) <= max_length:
        return [record]
    chunks: list[dict[str, Any]] = []
    current_cells: list[dict[str, Any]] = []
    for span in cell_spans:
        trial = current_cells + [span]
        start = trial[0]["start"]
        end = trial[-1]["end"]
        piece = text[start:end]
        n_tokens = len(
            tokenizer(piece, add_special_tokens=True, truncation=False)["input_ids"]
        )
        if n_tokens > max_length and current_cells:
            c_start = current_cells[0]["start"]
            c_end = current_cells[-1]["end"]
            chunk_text = text[c_start:c_end]
            chunk_entities = []
            for ent in entities:
                if ent["start"] >= c_start and ent["end"] <= c_end:
                    chunk_entities.append(
                        {
                            **ent,
                            "start": ent["start"] - c_start,
                            "end": ent["end"] - c_start,
                        }
                    )
            chunks.append(
                {
                    "document_id": record["document_id"],
                    "text": chunk_text,
                    "entities": chunk_entities,
                    "cell_spans": [
                        {
                            **c,
                            "start": c["start"] - c_start,
                            "end": c["end"] - c_start,
                        }
                        for c in current_cells
                    ],
                }
            )
            current_cells = [span]
        else:
            current_cells = trial
    if current_cells:
        c_start = current_cells[0]["start"]
        c_end = current_cells[-1]["end"]
        chunk_text = text[c_start:c_end]
        chunk_entities = []
        for ent in entities:
            if ent["start"] >= c_start and ent["end"] <= c_end:
                chunk_entities.append(
                    {
                        **ent,
                        "start": ent["start"] - c_start,
                        "end": ent["end"] - c_start,
                    }
                )
        chunks.append(
            {
                "document_id": record["document_id"],
                "text": chunk_text,
                "entities": chunk_entities,
                "cell_spans": [
                    {
                        **c,
                        "start": c["start"] - c_start,
                        "end": c["end"] - c_start,
                    }
                    for c in current_cells
                ],
            }
        )
    return chunks or [record]


class TranslationDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        config: TranslationConfig | None = None,
        max_samples: int | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        self.config = config or get_translation_config()
        prepare_translation_splits(self.config)
        path = PROCESSED_DIR / f"{self.config.processed_prefix}_{split}.jsonl"
        rows = _read_jsonl(path)
        self.pairs = [(r["tr"], r["en"]) for r in rows]
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]
        self.tokenizer = tokenizer
        self.max_length = self.config.max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tr, en = self.pairs[idx]
        if self.tokenizer is None:
            return {"tr": tr, "en": en}
        model_inputs = self.tokenizer(
            tr,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=en,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "tr": tr,
            "en": en,
        }


class ExtractionDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        config: ExtractionConfig | None = None,
        max_samples: int | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        self.config = config or get_extraction_config()
        prepare_extraction_splits(self.config)
        path = PROCESSED_DIR / f"{self.config.processed_prefix}_{split}.jsonl"
        records = _read_jsonl(path)
        self.tokenizer = tokenizer
        self.label2id = {label: i for i, label in enumerate(self.config.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.max_length = self.config.max_length
        expanded: list[dict[str, Any]] = []
        if tokenizer is not None:
            if ROW_MARKER not in tokenizer.get_vocab():
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": [ROW_MARKER, CELL_MARKER]}
                )
            for rec in records:
                expanded.extend(
                    _chunk_extraction_record(rec, self.max_length, tokenizer)
                )
        else:
            expanded = records
        if max_samples is not None:
            expanded = expanded[:max_samples]
        self.records = expanded

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        if self.tokenizer is None:
            return rec
        encoded = self.tokenizer(
            rec["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded["offset_mapping"].squeeze(0).tolist()
        labels = _align_labels(
            rec["text"], rec.get("entities", []), offsets, self.label2id
        )
        while len(labels) < self.max_length:
            labels.append(-100)
        labels = labels[: self.max_length]
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "document_id": rec["document_id"],
            "text": rec["text"],
            "entities": rec.get("entities", []),
            "cell_spans": rec.get("cell_spans", []),
        }


class MatchingDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        config: MatchingConfig | None = None,
        max_samples: int | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        self.config = config or get_matching_config()
        prepare_matching_splits(self.config)
        path = PROCESSED_DIR / f"{self.config.processed_prefix}_{split}.jsonl"
        self.records = _read_jsonl(path)
        if max_samples is not None:
            self.records = self.records[:max_samples]
        self.tokenizer = tokenizer
        self.max_length = self.config.max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        if self.tokenizer is None:
            return rec
        text_a = rec["header_text"]
        text_b = (
            f"{rec['candidate_text']} [SEP] {rec['row_context']} "
            f"[SEP] col_dist={rec['column_distance']}"
        )
        encoded = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        if "token_type_ids" in encoded:
            token_type_ids = encoded["token_type_ids"].squeeze(0)
        else:
            token_type_ids = torch.zeros_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": int(bool(rec["is_match"])),
            "document_id": rec["document_id"],
            "header_text": rec["header_text"],
            "candidate_text": rec["candidate_text"],
        }
