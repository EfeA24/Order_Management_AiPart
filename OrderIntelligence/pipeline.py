from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from config import (
    EXTRACTION_CHECKPOINT_DIR,
    MATCHING_CHECKPOINT_DIR,
    REVIEW_CONFIDENCE_THRESHOLD,
    TRANSLATION_CHECKPOINT_DIR,
)
from extraction import OrderExtractor
from matching import HeaderMatcher
from reader import build_model_input, read_xlsx, sort_cells
from schemas import FieldPrediction, OrderResult, ProductResult
from translation import Translator


def _checkpoint_or_none(directory: Path) -> Path | None:
    if (directory / "config.json").exists():
        return directory
    return None


class OrderPipeline:
    def __init__(
        self,
        translation_checkpoint: str | Path | None = None,
        extraction_checkpoint: str | Path | None = None,
        matching_checkpoint: str | Path | None = None,
    ) -> None:
        tr_path = (
            Path(translation_checkpoint)
            if translation_checkpoint
            else _checkpoint_or_none(TRANSLATION_CHECKPOINT_DIR)
        )
        ex_path = (
            Path(extraction_checkpoint)
            if extraction_checkpoint
            else _checkpoint_or_none(EXTRACTION_CHECKPOINT_DIR)
        )
        ma_path = (
            Path(matching_checkpoint)
            if matching_checkpoint
            else _checkpoint_or_none(MATCHING_CHECKPOINT_DIR)
        )
        self.translator = Translator(model_path=tr_path)
        self.extractor = OrderExtractor(model_path=ex_path)
        self.matcher = HeaderMatcher(model_path=ma_path)
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def parse(
        self, xlsx_path: str | Path, document_id: str | None = None
    ) -> OrderResult:
        path = Path(xlsx_path)
        doc_id = document_id or path.stem
        cells = sort_cells(read_xlsx(path))
        texts = [c["original_text"] for c in cells]
        english = self.translator.translate(texts)
        for cell, en in zip(cells, english, strict=True):
            cell["english_text"] = en
        model_text, cell_spans = build_model_input(cells, use_english=True)
        entities = self.extractor.extract(model_text, cell_spans)
        headers = [e for e in entities if e["type"] == "HEADER"]
        products = [e for e in entities if e["type"] == "PRODUCT"]
        used_structural = False
        if not headers or not products:
            structural = _structural_entities(cells)
            if not headers:
                headers = [e for e in structural if e["type"] == "HEADER"]
            if not products:
                products = [e for e in structural if e["type"] == "PRODUCT"]
            used_structural = True
        review_items: list[dict[str, Any]] = []
        if used_structural:
            review_items.append(
                {
                    "reason": "structural_fallback",
                    "detail": "Extraction model returned incomplete entities; row structure was used and needs review.",
                }
            )
        cells_by_id = {c["cell_id"]: c for c in cells}
        cells_by_row: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for cell in cells:
            cells_by_row[(cell["sheet"], cell["row"])].append(cell)
        product_results: list[ProductResult] = []
        header_meta: list[dict[str, Any]] = []
        for header in headers:
            h_cell = cells_by_id.get(header["cell_id"])
            if not h_cell:
                review_items.append(
                    {
                        "reason": "header_cell_missing",
                        "entity": header,
                    }
                )
                continue
            header_meta.append(
                {
                    "entity": header,
                    "cell": h_cell,
                    "column": h_cell["column"],
                    "sheet": h_cell["sheet"],
                }
            )
        for product in products:
            p_cell = cells_by_id.get(product["cell_id"])
            if not p_cell:
                review_items.append(
                    {
                        "reason": "product_cell_missing",
                        "entity": product,
                    }
                )
                continue
            row_cells = cells_by_row[(p_cell["sheet"], p_cell["row"])]
            row_context = " | ".join(
                (c.get("english_text") or c["original_text"]) for c in row_cells
            )
            fields: list[FieldPrediction] = []
            for hm in header_meta:
                if hm["sheet"] != p_cell["sheet"]:
                    continue
                if hm["column"] == p_cell["column"]:
                    continue
                same_col = next(
                    (c for c in row_cells if c["column"] == hm["column"]), None
                )
                candidates = [
                    c
                    for c in row_cells
                    if c["cell_id"] != product["cell_id"]
                    and c["cell_id"] != hm["cell"]["cell_id"]
                ]
                best: dict[str, Any] | None = None
                for cand in candidates:
                    col_dist = abs(cand["column"] - hm["column"])
                    pred = self.matcher.match(
                        hm["entity"]["text"],
                        cand.get("english_text") or cand["original_text"],
                        row_context,
                        col_dist,
                    )
                    if best is None or pred["confidence"] > best["confidence"]:
                        best = {**pred, "cell": cand, "column_distance": col_dist}
                used_column_fallback = False
                if (best is None or not best["is_match"]) and same_col is not None:
                    best = {
                        "is_match": True,
                        "confidence": 0.0,
                        "header_text": hm["entity"]["text"],
                        "candidate_text": same_col.get("english_text")
                        or same_col["original_text"],
                        "cell": same_col,
                        "column_distance": 0,
                    }
                    used_column_fallback = True
                if best is None or not best["is_match"]:
                    review_items.append(
                        {
                            "reason": "unmatched_header",
                            "product_cell_id": product["cell_id"],
                            "header": hm["entity"]["text"],
                            "best": best,
                        }
                    )
                    continue
                needs_review = (
                    used_column_fallback
                    or best["confidence"] < REVIEW_CONFIDENCE_THRESHOLD
                )
                if needs_review:
                    review_items.append(
                        {
                            "reason": (
                                "column_aligned_fallback"
                                if used_column_fallback
                                else "low_confidence_match"
                            ),
                            "product_cell_id": product["cell_id"],
                            "header": hm["entity"]["text"],
                            "match": best,
                        }
                    )
                fields.append(
                    FieldPrediction(
                        header_original=hm["cell"]["original_text"],
                        header_english=hm["cell"].get("english_text")
                        or hm["entity"]["text"],
                        value_original=best["cell"]["original_text"],
                        value_english=best["cell"].get("english_text")
                        or best["cell"]["original_text"],
                        source_cell_id=best["cell"]["cell_id"],
                        confidence=best["confidence"],
                        needs_review=needs_review,
                    )
                )
            product_results.append(
                ProductResult(
                    source_cell_id=product["cell_id"],
                    original_name=p_cell["original_text"],
                    english_name=p_cell.get("english_text") or product["text"],
                    fields=fields,
                )
            )
        if not products:
            review_items.append({"reason": "no_products_detected"})
        if not headers:
            review_items.append({"reason": "no_headers_detected"})
        return OrderResult(
            document_id=doc_id,
            products=product_results,
            review_items=review_items,
        )


def _structural_entities(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not cells:
        return []
    first_sheet = cells[0]["sheet"]
    sheet_cells = [c for c in cells if c["sheet"] == first_sheet]
    rows = sorted({c["row"] for c in sheet_cells})
    if not rows:
        return []
    header_row = rows[0]
    entities: list[dict[str, Any]] = []
    for cell in sheet_cells:
        text = cell.get("english_text") or cell["original_text"]
        if cell["row"] == header_row:
            entities.append(
                {
                    "type": "HEADER",
                    "text": text,
                    "cell_id": cell["cell_id"],
                    "confidence": 0.0,
                }
            )
        elif cell["column"] == min(
            c["column"] for c in sheet_cells if c["row"] == cell["row"]
        ):
            entities.append(
                {
                    "type": "PRODUCT",
                    "text": text,
                    "cell_id": cell["cell_id"],
                    "confidence": 0.0,
                }
            )
    return entities


def run_pipeline(text: str) -> dict:
    raise NotImplementedError("Use OrderPipeline.parse with an .xlsx path.")
