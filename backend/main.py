"""
main.py
FastAPI application — exposes REST endpoints consumed by the frontend.
"""

import os
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from rag_pipeline import ingest_document, retrieve, list_documents, delete_document
from gemma_client import query_gemma
from evaluator import run_full_evaluation

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("legal-llm")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BarPal.ai — Legal Document Q&A",
    description="Gemma 4 + RAG for private legal document analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    doc_filter: Optional[str] = None   # filename to restrict search to


class QueryResponse(BaseModel):
    answer:  str
    sources: list[dict]
    chunks_retrieved: int


class DeleteResponse(BaseModel):
    deleted_chunks: int
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = _FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "BarPal.ai backend running. Frontend not found at expected path."}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "BarPal.ai Legal Document Q&A"}


@app.post("/upload", summary="Upload a legal document (PDF or DOCX)")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF or DOCX file into the RAG vector store.
    """
    filename = file.filename or "document"
    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only PDF and DOCX are supported."
        )

    # Save to a temp file then ingest
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        stats = ingest_document(tmp_path, filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Error ingesting document: %s", filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "success":  True,
        "filename": stats["filename"],
        "doc_id":   stats["doc_id"],
        "pages":    stats["pages"],
        "chunks":   stats["chunks"],
        "message":  f"Successfully indexed {stats['chunks']} chunks from {stats['pages']} pages.",
    }


@app.post("/query", response_model=QueryResponse, summary="Ask a question about your documents")
async def query_documents(req: QueryRequest):
    """
    Retrieve relevant context from stored documents and generate an answer via Gemma 4.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 1 — Retrieve
    try:
        chunks = retrieve(req.question, doc_filter=req.doc_filter)
    except Exception as e:
        logger.exception("Retrieval error")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    if not chunks:
        return QueryResponse(
            answer="No relevant content found in your documents. Please upload relevant documents first or rephrase your question.",
            sources=[],
            chunks_retrieved=0,
        )

    # Step 2 — Generate
    try:
        answer = await query_gemma(req.question, chunks)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e) + " Check backend/.env → NVIDIA_API_KEY")
    except Exception as e:
        logger.exception("Gemma API error")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    # Deduplicate sources for the response
    seen_sources = set()
    sources = []
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({
                "filename": c["source"],
                "page":     c["page"],
                "score":    c["score"],
            })

    return QueryResponse(
        answer=answer,
        sources=sources,
        chunks_retrieved=len(chunks),
    )


@app.post("/evaluate", summary="RAGAS evaluation — Faithfulness, Answer Relevance, Context Precision")
async def evaluate_rag(req: QueryRequest):
    """
    Run the full RAG pipeline for a question, then score the result
    using three RAGAS-style LLM-as-judge metrics run in parallel:
      - Faithfulness      : Does the answer hallucinate beyond the retrieved context?
      - Answer Relevance  : Does the answer directly address the question?
      - Context Precision : Were the retrieved chunks actually relevant?
    Overall = harmonic mean of all three (penalises any single weak metric).
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 1 — Retrieve
    try:
        chunks = retrieve(req.question, doc_filter=req.doc_filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No relevant content found. Upload documents before evaluating."
        )

    # Step 2 — Generate answer
    try:
        answer = await query_gemma(req.question, chunks)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Step 3 — Run all 3 RAGAS metrics concurrently
    try:
        report = await run_full_evaluation(req.question, answer, chunks)
    except Exception as e:
        logger.exception("Evaluation error")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

    return report


@app.get("/documents", summary="List all ingested documents")

async def get_documents():
    """Return a list of all documents stored in the vector database."""
    try:
        docs = list_documents()
    except Exception as e:
        logger.exception("Error listing documents")
        raise HTTPException(status_code=500, detail=str(e))
    return {"documents": docs, "total": len(docs)}


@app.delete("/documents/{doc_id}", response_model=DeleteResponse, summary="Delete a document")
async def remove_document(doc_id: str):
    """Remove all chunks for a document from the vector store."""
    try:
        deleted = delete_document(doc_id)
    except Exception as e:
        logger.exception("Error deleting document %s", doc_id)
        raise HTTPException(status_code=500, detail=str(e))

    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found.")

    return DeleteResponse(
        deleted_chunks=deleted,
        message=f"Deleted document and its {deleted} indexed chunks.",
    )


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
