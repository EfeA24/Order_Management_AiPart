# Order Management - AiPart
A modular PyTorch reference project for converting Turkish order spreadsheets into structured JSON.
OrderIntelligence separates spreadsheet reading, Turkish-to-English translation, entity extraction, and header–value matching into independent components. The resulting JSON contract provides a foundation for future integration with an order management backend.
Project status: Reference implementation with training, evaluation, and API code. No project-specific fine-tuned checkpoints or benchmark results are included in the repository. Extraction and matching require task-specific training before their predictions can be considered meaningful.

🎯 Purpose
Incoming orders may use different column names, product descriptions, and spreadsheet layouts. This project explores a processing pipeline that will:
- Read nonempty Excel cells while retaining their source coordinates.
- Translate Turkish cell text into English.
- Identify HEADER and PRODUCT entities.
- Match candidate values to document-specific headers.
- Return products and their fields as structured JSON, with review information.
The current scope covers the Python/AI component. A .NET business application is a separate future integration.
🧠 Pipeline and models
Stage	Implementation	Output
Spreadsheet ingestion	openpyxl	Cells with sheet, row, column, and merge metadata
Translation	Helsinki-NLP/opus-mt-tr-en	English cell text alongside the original text
Entity extraction	bert-base-multilingual-cased with a token classification head	Header and product spans linked to cells
Header matching	bert-base-multilingual-cased with a binary classification head	Candidate match decisions and scores
Assembly	Pipeline logic and Pydantic schemas	An OrderResult JSON document


Translation
Translation runs in batches at the cell level. Cell identity remains outside the translation model, preserving the link to the source spreadsheet. A regex-based passthrough handles selected numeric, code, dimension, and unit patterns; it does not guarantee preservation of every possible identifier or mixed text format.
Entity extraction
The extraction module uses BIO labels:
O
B-HEADER
I-HEADER
B-PRODUCT
I-PRODUCT
[row] and [cell] markers provide boundaries in the serialized table text. Predicted spans are mapped back to source cells.
Dynamic header matching
Headers are supplied as input text rather than registered as new vocabulary tokens for every document. A matching example contains:
- Header text, such as Quantity.
- Candidate value, such as 100.
- Row context, such as Bolt | 100 | Stainless steel.
- Column distance, encoded as text in the model input.
The classifier predicts whether the candidate belongs to that header. Handling unseen header names remains an evaluation question, not a guaranteed capability.
Checkpoints and fallback behavior
The pipeline looks for saved models under checkpoints/<task>/, using the presence of config.json to select a local checkpoint. Otherwise, it loads the configured base model.
The base BERT models are not trained for this project's labels. Their newly initialized task heads cannot provide reliable order extraction or matching without training. A model loading successfully does not establish task readiness.
When extraction returns no headers or no products, a structural fallback uses the first nonempty row as headers and the leftmost nonempty cell in subsequent rows as the product. This fallback currently operates on the first sheet in the sorted cell list. Matching also has a same-column fallback when no positive candidate is selected. Fallback events are recorded for review; they are heuristics, not learned predictions.
📁 Repository structure
Application code is located in OrderIntelligence/ on the main branch.
Order_Management_AiPart/
├── README.md
└── OrderIntelligence/
    ├── config.py          # Paths, model names, labels, and hyperparameters
    ├── reader.py          # Excel reading and cell-aware serialization
    ├── dataset.py         # Data preparation and PyTorch datasets
    ├── translation.py     # Seq2seq translation wrapper
    ├── extraction.py      # HEADER / PRODUCT token classification
    ├── matching.py        # Header–candidate binary classification
    ├── train.py           # Optional fine-tuning entry point
    ├── evaluate.py        # Evaluation and pipeline inspection
    ├── pipeline.py        # Processing orchestration and result assembly
    ├── schemas.py         # Pydantic data contracts
    ├── api.py             # FastAPI application
    ├── requirements.txt
    └── README.md
Data, model checkpoints, and generated outputs are local resources, not bundled examples. Create the relevant directories as needed:
OrderIntelligence/
├── data/
│   ├── translations/
│   ├── orders/
│   ├── annotations/
│   └── processed/
├── checkpoints/
│   ├── translation/
│   ├── extraction/
│   └── matching/
└── outputs/
⚙️ Setup
Use a separate Python 3.11 Conda environment. The commands below describe the intended setup; installation and runtime compatibility have not been validated as part of this reference release.
git clone https://github.com/EfeA24/Order_Management_AiPart.git
cd Order_Management_AiPart/OrderIntelligence

conda create -n order-nlp python=3.11 pip -y
conda activate order-nlp
python -m pip install --upgrade pip
For an NVIDIA GPU, install the CUDA-enabled PyTorch build compatible with your driver, then install the project dependencies. For the CUDA 12.6 package index:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
The requirements file currently pins torch==2.11.0; installing it may change an existing Torch installation. Keep the selected PyTorch build consistent with that pin, or deliberately update the dependency specification when validating another version. Consult the official PyTorch installation selector for platform-specific instructions.
Verify the installed build:
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
In PyCharm, select the existing order-nlp Conda environment as the project interpreter. Run the following application commands from OrderIntelligence/ because the modules use local imports.
Initial model loading requires access to Hugging Face to download the configured model and tokenizer files. Subsequent loads may use cached files.
🚀 Usage
Read a spreadsheet without loading models
Place your own sample workbook at data/orders/sample.xlsx.
from reader import build_model_input, read_xlsx

cells = read_xlsx("data/orders/sample.xlsx")
text, cell_spans = build_model_input(cells, use_english=False)
print(text)
print(cell_spans)
Translate text
from translation import Translator

translator = Translator()
translations = translator.translate(["Paslanmaz çelik cıvata", "100"])
print(translations)
Run the full pipeline
For meaningful model-based extraction and matching, supply trained checkpoints. Without them, this remains an inspection of the code path and its heuristics.
from pipeline import OrderPipeline

pipeline = OrderPipeline()
result = pipeline.parse("data/orders/sample.xlsx", document_id="order-001")
print(result.model_dump_json(indent=2))
Start the API
python -m uvicorn api:app --host 127.0.0.1 --port 8000
Endpoint	Purpose
GET /health	Report pipeline initialization status
POST /orders/parse	Accept an .xlsx upload in multipart field file and return JSON
/docs	Interactive FastAPI documentation


The current upload limit is 10 MiB, checked after reading the upload. /health reports initialization, not model accuracy or the presence of task-trained heads.
A future .NET client can submit a workbook and consume the synchronous JSON response. No .NET client implementation is included.
📄 Illustrative JSON contract
The following example demonstrates the schema only. It is not a measured prediction or a reported model score.
{
  "document_id": "order-001",
  "products": [
    {
      "source_cell_id": "Sheet1!A2",
      "original_name": "Cıvata",
      "english_name": "Bolt",
      "fields": [
        {
          "header_original": "Adet",
          "header_english": "Quantity",
          "value_original": "100",
          "value_english": "100",
          "source_cell_id": "Sheet1!B2",
          "confidence": null,
          "needs_review": true
        }
      ]
    }
  ],
  "review_items": []
}
Original values remain separate from translated values. Numeric values remain strings; unit conversion and business validation are outside the current pipeline. Model softmax scores are not calibrated probabilities of correctness.
🏋️ Optional fine-tuning infrastructure
Training code is included for future experiments. Training has not been performed for this reference project, and no accuracy claims are made.
Translation data
data/translations/KDE4.en-tr.tr
data/translations/KDE4.en-tr.en
Corresponding line numbers form translation pairs. The preparation code checks line counts, removes blank and duplicate pairs, and groups identical Turkish source strings when creating splits.
KDE4 contains software localization text. It is not an order-domain dataset and contains no project-specific entity or relation annotations. It is not required for pretrained translation inference. Follow the corpus's source licensing and attribution requirements before using or redistributing it.
Extraction annotations
Store records in data/annotations/extraction.jsonl, with one JSON object per line. Character offsets must match the serialized text exactly, with an exclusive end offset:
{"document_id":"order-001","text":"[row] [cell] Product","entities":[{"type":"HEADER","start":13,"end":20,"text":"Product","cell_id":"Sheet1!A1"}],"cell_spans":[{"start":13,"end":20,"cell_id":"Sheet1!A1"}]}
Matching annotations
Store positive and negative pairs in data/annotations/matching.jsonl:
{"document_id":"order-001","header_text":"Quantity","candidate_text":"100","row_context":"Bolt | 100 | Stainless steel","column_distance":0,"is_match":true}
Use enough independent documents for distinct train, validation, and test sets. The current small-data split logic reuses records when only one or two document IDs exist; such splits are suitable only for smoke checks, not independent evaluation.
Training entry points
After preparing and validating the appropriate data:
python train.py --task translation --max-samples 512 --epochs 1
python train.py --task extraction --epochs 3
python train.py --task matching --epochs 3
Settings are in config.py. Training uses AdamW, validation-loss-based checkpoint selection, and optional CUDA mixed precision. Saved outputs include model/tokenizer files and training metadata under checkpoints/<task>/, with metric files under outputs/. Saved optimizer state is available, but automatic training resumption is not implemented by the CLI.
📊 Evaluation infrastructure
With the required datasets and checkpoints available:
python evaluate.py --task translation --pretrained --max-samples 200
python evaluate.py --task translation --checkpoint checkpoints/translation
python evaluate.py --task extraction --checkpoint checkpoints/extraction
python evaluate.py --task matching --checkpoint checkpoints/matching
python evaluate.py --task e2e
The code computes translation BLEU/chrF and code/number preservation, extraction precision/recall/F1, and matching classification metrics. The e2e command inspects a local sample workbook and reports counts and JSON; it does not measure end-to-end accuracy against ground truth.
The current extraction scorer aggregates entities across documents without document IDs in the comparison key. Its results require scorer validation before use as benchmark evidence.
🔎 Current limitations
- .xlsx input only; PDF, image, OCR, and legacy .xls support are not included.
- Product association assumes one product per row.
- Extraction input is truncated at the configured token limit; inference does not yet implement sliding windows for long workbooks.
- The structural extraction fallback is limited to one sheet and assumes the first nonempty row contains headers.
- Formatting-sensitive values may change during spreadsheet reading: cells are converted from their underlying values to strings, not reconstructed from Excel display formatting.
- Formula cells use cached values through data_only=True; formulas are not recalculated.
- Matching is evaluated sequentially per candidate; batching and throughput optimization remain future work.
- Authentication, durable job processing, and deployment hardening are outside this reference implementation.
- No trained project checkpoints, reproducible benchmark results, or verified production deployment are included.
🛠️ Next development steps
- Validate the pinned dependency environment and add focused integration checks.
- Create synthetic, redistributable order examples and expected JSON fixtures.
- Train and validate extraction and matching heads on annotated order documents.
- Require task-trained checkpoints or provide an explicit heuristic-only mode.
- Add document-aware evaluation and ground-truth end-to-end checks.
- Extend inference to long tables, multiple table regions, and multiline products.
- Add the .NET application when the integration scope is defined.
👤 Author
Efe Aydeskin
License
No project license file is currently included. Model weights and external datasets remain subject to their respective licenses.
