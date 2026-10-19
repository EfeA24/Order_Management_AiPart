from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from config import (
    CELL_MARKER,
    DEVICE,
    OUTPUTS_DIR,
    ROW_MARKER,
    SEED,
    get_extraction_config,
    get_matching_config,
    get_translation_config,
)
from dataset import ExtractionDataset, MatchingDataset, TranslationDataset


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_stack(batch: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        elif isinstance(values[0], list):
            out[key] = torch.tensor(values)
        else:
            out[key] = torch.tensor(values)
    return out


def translation_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return _collate_stack(batch, ["input_ids", "attention_mask", "labels"])


def extraction_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return _collate_stack(batch, ["input_ids", "attention_mask", "labels"])


def matching_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    out = _collate_stack(
        batch, ["input_ids", "attention_mask", "token_type_ids", "labels"]
    )
    return out


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _train_loop(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    num_epochs: int,
    grad_accum: int,
    use_amp: bool,
    checkpoint_dir: Path,
    tokenizer,
    extra_save=None,
    task_name: str = "task",
) -> dict[str, Any]:
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device == "cuda")
    best_val = float("inf")
    history: list[dict[str, float]] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp and device == "cuda"):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            scaler.scale(loss).backward()
            train_loss += outputs.loss.item()
            if step % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp and device == "cuda"):
                    outputs = model(**batch)
                val_loss += outputs.loss.item()
        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        )
        print(
            f"[{task_name}] epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )
        if val_loss <= best_val:
            best_val = val_loss
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val,
                },
                checkpoint_dir / "training_state.pt",
            )
            if extra_save:
                extra_save(checkpoint_dir)
    metrics = {
        "task": task_name,
        "best_val_loss": best_val,
        "history": history,
        "checkpoint_dir": str(checkpoint_dir),
    }
    _save_json(OUTPUTS_DIR / f"{task_name}_train_metrics.json", metrics)
    return metrics


def train_translation(max_samples: int | None = None, num_epochs: int | None = None):
    set_seed()
    cfg = get_translation_config()
    if num_epochs is not None:
        cfg.num_epochs = num_epochs
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    model.to(DEVICE)
    train_ds = TranslationDataset(
        "train", tokenizer=tokenizer, config=cfg, max_samples=max_samples
    )
    val_ds = TranslationDataset(
        "val", tokenizer=tokenizer, config=cfg, max_samples=max_samples
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=translation_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=translation_collate
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = max(
        1, (len(train_loader) // cfg.gradient_accumulation_steps) * cfg.num_epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    settings = {
        "model_name": cfg.model_name,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "num_epochs": cfg.num_epochs,
        "max_length": cfg.max_length,
        "max_samples": max_samples,
        "device": DEVICE,
    }
    _save_json(cfg.checkpoint_dir / "train_settings.json", settings)
    return _train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        DEVICE,
        cfg.num_epochs,
        cfg.gradient_accumulation_steps,
        cfg.use_mixed_precision,
        cfg.checkpoint_dir,
        tokenizer,
        task_name="translation",
    )


def train_extraction(max_samples: int | None = None, num_epochs: int | None = None):
    set_seed()
    cfg = get_extraction_config()
    if num_epochs is not None:
        cfg.num_epochs = num_epochs
    label2id = {label: i for i, label in enumerate(cfg.labels)}
    id2label = {i: label for label, i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [ROW_MARKER, CELL_MARKER]}
    )
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)
    train_ds = ExtractionDataset(
        "train", tokenizer=tokenizer, config=cfg, max_samples=max_samples
    )
    val_ds = ExtractionDataset(
        "val", tokenizer=tokenizer, config=cfg, max_samples=max_samples
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=extraction_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=extraction_collate
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = max(
        1, (len(train_loader) // cfg.gradient_accumulation_steps) * cfg.num_epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    def extra_save(directory: Path) -> None:
        payload = {
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
            "labels": cfg.labels,
        }
        _save_json(directory / "label_map.json", payload)

    settings = {
        "model_name": cfg.model_name,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "num_epochs": cfg.num_epochs,
        "max_length": cfg.max_length,
        "labels": cfg.labels,
        "max_samples": max_samples,
        "device": DEVICE,
    }
    _save_json(cfg.checkpoint_dir / "train_settings.json", settings)
    return _train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        DEVICE,
        cfg.num_epochs,
        cfg.gradient_accumulation_steps,
        cfg.use_mixed_precision,
        cfg.checkpoint_dir,
        tokenizer,
        extra_save=extra_save,
        task_name="extraction",
    )


def train_matching(max_samples: int | None = None, num_epochs: int | None = None):
    set_seed()
    cfg = get_matching_config()
    if num_epochs is not None:
        cfg.num_epochs = num_epochs
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2
    )
    model.to(DEVICE)
    train_ds = MatchingDataset(
        "train", tokenizer=tokenizer, config=cfg, max_samples=max_samples
    )
    val_ds = MatchingDataset(
        "val", tokenizer=tokenizer, config=cfg, max_samples=max_samples
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=matching_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=matching_collate
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = max(
        1, (len(train_loader) // cfg.gradient_accumulation_steps) * cfg.num_epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    settings = {
        "model_name": cfg.model_name,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "num_epochs": cfg.num_epochs,
        "max_length": cfg.max_length,
        "max_samples": max_samples,
        "device": DEVICE,
    }
    _save_json(cfg.checkpoint_dir / "train_settings.json", settings)
    return _train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        DEVICE,
        cfg.num_epochs,
        cfg.gradient_accumulation_steps,
        cfg.use_mixed_precision,
        cfg.checkpoint_dir,
        tokenizer,
        task_name="matching",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OrderIntelligence training")
    parser.add_argument(
        "--task",
        choices=["translation", "extraction", "matching"],
        required=True,
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.task == "translation":
        metrics = train_translation(args.max_samples, args.epochs)
    elif args.task == "extraction":
        metrics = train_extraction(args.max_samples, args.epochs)
    else:
        metrics = train_matching(args.max_samples, args.epochs)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
