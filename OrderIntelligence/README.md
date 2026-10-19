# OrderIntelligence

Three-stage ML pipeline that turns order Excel files into structured JSON:

1. **Translation** — Turkish to English (`Helsinki-NLP/opus-mt-tr-en`)
2. **Extraction** — HEADER / PRODUCT token classification
3. **Matching** — dynamic header-to-value classification

## Setup

```bash
cd OrderIntelligence
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### CUDA PyTorch

Install a CUDA build of PyTorch for your platform, then install the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Confirm the CUDA index URL at [pytorch.org](https://pytorch.org/get-started/locally/).

## Data layout

| Path | Purpose |
|------|---------|
| `data/translations/` | KDE4 parallel `.tr` / `.en` files (translation only) |
| `data/orders/` | Order `.xlsx` files (`sample.xlsx` included) |
| `data/annotations/` | `extraction.jsonl`, `matching.jsonl` |
| `data/processed/` | Generated train/val/test splits |

KDE4 is not used for extraction or matching.

### Annotation formats

Extraction (one document per line):

```json
{
  "document_id": "order-001",
  "text": "[row] [cell] Product [cell] Quantity ...",
  "entities": [
    {"type": "HEADER", "start": 14, "end": 21, "text": "Product", "cell_id": "Sheet1!A1"}
  ],
  "cell_spans": [{"start": 14, "end": 21, "cell_id": "Sheet1!A1"}]
}
```

Matching (one pair per line):

```json
{
  "document_id": "order-001",
  "header_text": "Quantity",
  "candidate_text": "100",
  "row_context": "Bolt | 100 | Stainless steel",
  "column_distance": 0,
  "is_match": true
}
```

## Train

```bash
python train.py --task translation --max-samples 512 --epochs 1
python train.py --task extraction --epochs 3
python train.py --task matching --epochs 3
```

Checkpoints are written under `checkpoints/<task>/`. Metrics go to `outputs/`.

## Evaluate

```bash
python evaluate.py --task translation --pretrained --max-samples 200
python evaluate.py --task extraction
python evaluate.py --task matching
python evaluate.py --task e2e
```

## API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

- `GET /health` — service and model readiness
- `POST /orders/parse` — upload `.xlsx`, receive `OrderResult` JSON

A .NET client can POST the file and use the synchronous JSON response. No callback service is required.

## Limits

- `.xlsx` only (no PDF/OCR yet)
- One product per row for matching
- Numbers stay as text (no unit conversion)
- Unmatched or low-confidence fields are listed in `review_items` / `needs_review`
