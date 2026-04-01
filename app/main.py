"""
FastAPI Backend — REST API for the Intelligent Document Processor.

Endpoints:
  POST /upload         — Upload a document (PDF/image), extract text, ingest into RAG
  POST /query          — Ask a question about uploaded documents
  GET  /documents      — List all ingested documents
  GET  /extract/{id}   — Get structured data extraction for a document
  DELETE /clear        — Clear all documents and reset
  GET  /health         — Health check
"""

import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import UPLOAD_DIR, API_HOST, API_PORT
from app.ocr import extract_text
from app.rag import RAGPipeline
from app.extraction import generate_extraction_summary

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──
app = FastAPI(
    title="RAG Document Intelligence",
    description="Production-grade RAG + OCR pipeline for document analysis. Upload PDFs or images, ask questions, extract structured data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ──
rag = RAGPipeline()
extraction_cache: dict[str, dict] = {}


# ── Models ──
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    method: str
    total_chars: int
    chunks: int
    message: str


# ── Endpoints ──

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_loaded": rag.document_count,
        "ready": rag.is_ready,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or image, extract text via OCR, and ingest into RAG pipeline."""

    # Validate file type
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(allowed)}"
        )

    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Uploaded: {file.filename} ({file_path.stat().st_size / 1024:.1f} KB)")

    try:
        # Step 1: Extract text (OCR if needed)
        extraction = extract_text(file_path)
        text = extraction["text"]

        if not text.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted from this document.")

        # Step 2: Ingest into RAG pipeline
        doc_id = rag.ingest(
            text=text,
            metadata={
                "filename": file.filename,
                "pages": extraction["pages"],
                "method": extraction["method"],
            },
        )

        # Step 3: Cache structured extraction
        extraction_cache[doc_id] = generate_extraction_summary(text)

        # Get document info
        docs = rag.list_documents()
        doc_info = next((d for d in docs if d["doc_id"] == doc_id), {})

        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            pages=extraction["pages"],
            method=extraction["method"],
            total_chars=extraction["total_chars"],
            chunks=doc_info.get("chunks", 0),
            message=f"Document processed successfully. {extraction['pages']} page(s) extracted via {extraction['method']}. Ready for questions.",
        )

    except Exception as e:
        logger.error(f"Processing failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question about the uploaded documents."""
    if not rag.is_ready:
        raise HTTPException(status_code=400, detail="No documents uploaded yet. Please upload a document first.")

    result = rag.query(request.question)
    return QueryResponse(**result)


@app.get("/documents")
async def list_documents():
    """List all ingested documents."""
    return {
        "documents": rag.list_documents(),
        "total": rag.document_count,
    }


@app.get("/extract/{doc_id}")
async def get_extraction(doc_id: str):
    """Get structured data extraction for a specific document."""
    if doc_id not in extraction_cache:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
    return extraction_cache[doc_id]


@app.delete("/clear")
async def clear_all():
    """Clear all documents and reset the pipeline."""
    rag.clear()
    extraction_cache.clear()
    # Clean upload directory
    for f in UPLOAD_DIR.iterdir():
        f.unlink(missing_ok=True)
    return {"message": "All documents cleared."}


# ── Run ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=True)
