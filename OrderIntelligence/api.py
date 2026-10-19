from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from pipeline import OrderPipeline

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

_pipeline: OrderPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = OrderPipeline()
    yield
    _pipeline = None


app = FastAPI(title="OrderIntelligence", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    ready = _pipeline is not None and getattr(_pipeline, "ready", False)
    return {
        "status": "ok" if _pipeline is not None else "starting",
        "models_ready": ready,
    }


@app.post("/orders/parse")
async def parse_order(file: UploadFile = File(...)) -> JSONResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready.")
    filename = file.filename or ""
    if not filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx files are supported.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds size limit.")
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        result = _pipeline.parse(tmp_path, document_id=Path(filename).stem)
        return JSONResponse(content=result.model_dump())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
