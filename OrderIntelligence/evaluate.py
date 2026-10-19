from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from config import (
    OUTPUTS_DIR,
    ORDERS_DIR,
    get_extraction_config,
    get_matching_config,
    get_translation_config,
)
from dataset import (
    ExtractionDataset,
    MatchingDataset,
    TranslationDataset,
    prepare_extraction_splits,
    prepare_matching_splits,
    prepare_translation_splits,
)
from extraction import OrderExtractor
from matching import HeaderMatcher
from translation import Translator

_CODE_OR_NUMBER_RE = re.compile(
    r"\d+(?:[.,]\d+)?|[A-Za-z0-9]+[-_/][A-Za-z0-9][-_/A-Za-z0-9]*",
    re.IGNORECASE,
)


def extract_codes_and_numbers(text: str) -> set[str]:
    return {m.group(0).lower() for m in _CODE_OR_NUMBER_RE.finditer(text or "")}


def code_preservation_rate(sources: list[str], hypotheses: list[str]) -> float:
    total = 0
    preserved = 0
    for src, hyp in zip(sources, hypotheses, strict=True):
        tokens = extract_codes_and_numbers(src)
        if not tokens:
            continue
        hyp_tokens = extract_codes_and_numbers(hyp)
        total += len(tokens)
        preserved += sum(1 for t in tokens if t in hyp_tokens)
    if total == 0:
        return 1.0
    return preserved / total


def _entity_key(ent: dict[str, Any]) -> tuple:
    return (
        ent.get("type"),
        ent.get("text", "").strip().lower(),
        ent.get("cell_id"),
    )


def entity_prf(
    gold: list[dict[str, Any]], pred: list[dict[str, Any]]
) -> dict[str, float]:
    gold_set = {_entity_key(e) for e in gold}
    pred_set = {_entity_key(e) for e in pred}
    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_translation(
    pretrained: bool = True,
    checkpoint: str | Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    import sacrebleu

    cfg = get_translation_config()
    meta = prepare_translation_splits(cfg)
    n = max_samples if max_samples is not None else cfg.eval_max_samples
    dataset = TranslationDataset(split="test", tokenizer=None, config=cfg, max_samples=n)
    sources = [tr for tr, _ in dataset.pairs]
    references = [en for _, en in dataset.pairs]
    if pretrained or checkpoint is None:
        translator = Translator(model_path=None, config=cfg)
        model_id = cfg.model_name
    else:
        translator = Translator(model_path=checkpoint, config=cfg)
        model_id = str(checkpoint)
    hypotheses = translator.translate(sources)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    preservation = code_preservation_rate(sources, hypotheses)
    result = {
        "task": "translation",
        "model": model_id,
        "pretrained": pretrained or checkpoint is None,
        "num_samples": len(sources),
        "split_meta": {
            "cleaned_pairs": meta.get("cleaned_pairs"),
            "test_pairs": meta.get("test_pairs"),
        },
        "bleu": bleu.score,
        "chrf": chrf.score,
        "code_number_preservation": preservation,
        "samples": [
            {"tr": s, "ref": r, "hyp": h}
            for s, r, h in zip(sources[:5], references[:5], hypotheses[:5], strict=True)
        ],
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "translation_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    result["output_path"] = str(out_path)
    return result


def evaluate_extraction(
    pretrained: bool = False,
    checkpoint: str | Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    cfg = get_extraction_config()
    meta = prepare_extraction_splits(cfg)
    n = max_samples if max_samples is not None else cfg.eval_max_samples
    dataset = ExtractionDataset(split="test", tokenizer=None, config=cfg, max_samples=n)
    if checkpoint is None and not pretrained:
        ckpt = cfg.checkpoint_dir
        checkpoint = ckpt if (ckpt / "config.json").exists() else None
    model_path = None if pretrained else checkpoint
    extractor = OrderExtractor(model_path=model_path, config=cfg)
    all_gold: list[dict[str, Any]] = []
    all_pred: list[dict[str, Any]] = []
    for rec in dataset.records:
        gold = [
            {
                "type": e["type"],
                "text": e.get("text") or rec["text"][e["start"] : e["end"]],
                "cell_id": e.get("cell_id"),
            }
            for e in rec.get("entities", [])
        ]
        pred = extractor.extract(rec["text"], rec.get("cell_spans"))
        all_gold.extend(gold)
        all_pred.extend(pred)
    scores = entity_prf(all_gold, all_pred)
    result = {
        "task": "extraction",
        "model": str(model_path or cfg.model_name),
        "pretrained": model_path is None,
        "num_docs": len(dataset.records),
        "split_meta": meta,
        **scores,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "extraction_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    result["output_path"] = str(out_path)
    return result


def evaluate_matching(
    pretrained: bool = False,
    checkpoint: str | Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    cfg = get_matching_config()
    meta = prepare_matching_splits(cfg)
    n = max_samples if max_samples is not None else cfg.eval_max_samples
    dataset = MatchingDataset(split="test", tokenizer=None, config=cfg, max_samples=n)
    if checkpoint is None and not pretrained:
        ckpt = cfg.checkpoint_dir
        checkpoint = ckpt if (ckpt / "config.json").exists() else None
    model_path = None if pretrained else checkpoint
    matcher = HeaderMatcher(model_path=model_path, config=cfg)
    tp = fp = fn = 0
    field_correct = 0
    field_total = 0
    for rec in dataset.records:
        out = matcher.match(
            rec["header_text"],
            rec["candidate_text"],
            rec.get("row_context", ""),
            int(rec.get("column_distance", 0)),
        )
        gold = bool(rec["is_match"])
        pred = bool(out["is_match"])
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        field_total += 1
        if pred == gold:
            field_correct += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    result = {
        "task": "matching",
        "model": str(model_path or cfg.model_name),
        "pretrained": model_path is None,
        "num_examples": len(dataset.records),
        "split_meta": meta,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "field_match_accuracy": field_correct / field_total if field_total else 0.0,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "matching_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    result["output_path"] = str(out_path)
    return result


def evaluate_e2e() -> dict[str, Any]:
    from pipeline import OrderPipeline

    sample = ORDERS_DIR / "sample.xlsx"
    if not sample.exists():
        return {"task": "e2e", "error": f"Sample order not found: {sample}"}
    pipeline = OrderPipeline()
    result = pipeline.parse(sample, document_id="sample")
    payload = {
        "task": "e2e",
        "document_id": result.document_id,
        "num_products": len(result.products),
        "num_review_items": len(result.review_items),
        "result": result.model_dump(),
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "e2e_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    payload["output_path"] = str(out_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="OrderIntelligence evaluation")
    parser.add_argument(
        "--task",
        choices=["translation", "extraction", "matching", "e2e"],
        required=True,
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if args.task == "translation":
        result = evaluate_translation(
            pretrained=args.pretrained or args.checkpoint is None,
            checkpoint=args.checkpoint,
            max_samples=args.max_samples,
        )
    elif args.task == "extraction":
        result = evaluate_extraction(
            pretrained=args.pretrained,
            checkpoint=args.checkpoint,
            max_samples=args.max_samples,
        )
    elif args.task == "matching":
        result = evaluate_matching(
            pretrained=args.pretrained,
            checkpoint=args.checkpoint,
            max_samples=args.max_samples,
        )
    else:
        result = evaluate_e2e()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
