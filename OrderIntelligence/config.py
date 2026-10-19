from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
TRANSLATIONS_DIR = DATA_DIR / "translations"
ORDERS_DIR = DATA_DIR / "orders"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
PROCESSED_DIR = DATA_DIR / "processed"

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
TRANSLATION_CHECKPOINT_DIR = CHECKPOINTS_DIR / "translation"
EXTRACTION_CHECKPOINT_DIR = CHECKPOINTS_DIR / "extraction"
MATCHING_CHECKPOINT_DIR = CHECKPOINTS_DIR / "matching"

OUTPUTS_DIR = BASE_DIR / "outputs"

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROW_MARKER = "[row]"
CELL_MARKER = "[cell]"

EXTRACTION_LABELS = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-PRODUCT",
    "I-PRODUCT",
]

REVIEW_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class TranslationConfig:
    model_name: str = "Helsinki-NLP/opus-mt-tr-en"
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_length: int = 128
    eval_max_samples: int = 500
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    tr_file: Path = field(default_factory=lambda: TRANSLATIONS_DIR / "KDE4.en-tr.tr")
    en_file: Path = field(default_factory=lambda: TRANSLATIONS_DIR / "KDE4.en-tr.en")
    checkpoint_dir: Path = field(default_factory=lambda: TRANSLATION_CHECKPOINT_DIR)
    processed_prefix: str = "translation"


@dataclass
class ExtractionConfig:
    model_name: str = "bert-base-multilingual-cased"
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 5
    max_length: int = 256
    eval_max_samples: int = 200
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    labels: list[str] = field(default_factory=lambda: list(EXTRACTION_LABELS))
    annotations_file: Path = field(
        default_factory=lambda: ANNOTATIONS_DIR / "extraction.jsonl"
    )
    checkpoint_dir: Path = field(default_factory=lambda: EXTRACTION_CHECKPOINT_DIR)
    processed_prefix: str = "extraction"


@dataclass
class MatchingConfig:
    model_name: str = "bert-base-multilingual-cased"
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 5
    max_length: int = 128
    eval_max_samples: int = 500
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    annotations_file: Path = field(
        default_factory=lambda: ANNOTATIONS_DIR / "matching.jsonl"
    )
    checkpoint_dir: Path = field(default_factory=lambda: MATCHING_CHECKPOINT_DIR)
    processed_prefix: str = "matching"


def get_translation_config() -> TranslationConfig:
    return TranslationConfig()


def get_extraction_config() -> ExtractionConfig:
    return ExtractionConfig()


def get_matching_config() -> MatchingConfig:
    return MatchingConfig()
