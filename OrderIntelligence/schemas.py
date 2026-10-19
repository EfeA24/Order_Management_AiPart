from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Cell(BaseModel):
    cell_id: str
    sheet: str
    row: int
    column: int
    original_text: str
    english_text: str | None = None
    merge_range: str | None = None


class EntityPrediction(BaseModel):
    type: Literal["HEADER", "PRODUCT"]
    text: str
    cell_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class FieldPrediction(BaseModel):
    header_original: str
    header_english: str
    value_original: str
    value_english: str
    source_cell_id: str
    confidence: float | None = None
    needs_review: bool = False


class ProductResult(BaseModel):
    source_cell_id: str
    original_name: str
    english_name: str
    fields: list[FieldPrediction] = Field(default_factory=list)


class OrderResult(BaseModel):
    document_id: str
    products: list[ProductResult] = Field(default_factory=list)
    review_items: list[dict] = Field(default_factory=list)
