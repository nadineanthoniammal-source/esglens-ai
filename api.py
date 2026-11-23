"""REST API so other tools (n8n, scripts, schedulers) can use ESGLens AI.

Run:  uvicorn api:app --port 8000
Test: curl -X POST -F "file=@report.pdf" -F "revenue_inr_crore=557163" http://localhost:8000/analyze
"""
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

load_dotenv()

from app.service import analyze_bytes, to_jsonable  # noqa: E402

app = FastAPI(title="ESGLens AI API", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(file: UploadFile = File(...), revenue_inr_crore: Optional[float] = Form(None)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    data = file.file.read()
    try:
        _, n_chunks, result = analyze_bytes(data, revenue_inr_crore)
    except ValueError as e:
        raise HTTPException(422, str(e))
    out = to_jsonable(result)
    out["chunks_indexed"] = n_chunks
    return out
