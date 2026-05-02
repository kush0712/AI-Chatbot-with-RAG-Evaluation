"""
rag_pipeline.py
Handles document ingestion (PDF / DOCX) and semantic retrieval using
sentence-transformers + ChromaDB — all running 100% locally.
"""

import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Optional

import pdfplumber
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from docx import Document as DocxDocument
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K          = int(os.getenv("TOP_K_RESULTS", 5))

COLLECTION_NAME = "legal_docs"
EMBED_MODEL     = "all-MiniLM-L6-v2"  # Fast, 384-dim, runs locally

# ── ChromaDB client (singleton) ──────────────────────────────────────────────

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection    = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_pdf(file_path: str) -> list[dict]:
    """Returns list of {page: int, text: str}."""
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": i, "text": text})
    return pages


def _extract_docx(file_path: str) -> list[dict]:
    """Returns list of {page: int, text: str} (DOCX has no native pages, we group paragraphs)."""
    doc = DocxDocument(file_path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    # Treat the whole DOCX as page 1
    return [{"page": 1, "text": full_text}] if full_text else []


def _extract_text(file_path: str) -> list[dict]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return _extract_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Recursive character splitter: tries to split on paragraph breaks, then
    sentences, then words, before doing a hard character split.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []

    def _split(text: str, sep_idx: int):
        if len(text) <= chunk_size or sep_idx >= len(separators):
            if text.strip():
                chunks.append(text.strip())
            return
        sep = separators[sep_idx]
        parts = text.split(sep) if sep else list(text)
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # Start fresh, but keep overlap
                overlap_text = current[-overlap:] if overlap and current else ""
                current = overlap_text + (sep if overlap_text else "") + part
        if current.strip():
            chunks.append(current.strip())

    _split(text, 0)
    return chunks


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_document(file_path: str, filename: str) -> dict:
    """
    Extract → chunk → embed → store in ChromaDB.
    Returns stats dict: {doc_id, filename, pages, chunks}.
    """
    collection = _get_collection()

    # Create a stable doc_id from the filename
    doc_id = hashlib.md5(filename.encode()).hexdigest()

    # Delete any previous version of this file
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info("Deleted %d existing chunks for %s", len(existing["ids"]), filename)

    pages = _extract_text(file_path)
    if not pages:
        raise ValueError("No extractable text found in document.")

    all_chunks   = []
    all_ids      = []
    all_metadatas = []

    for page_data in pages:
        raw_chunks = _chunk_text(page_data["text"])
        for j, chunk in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_p{page_data['page']}_c{j}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "doc_id":   doc_id,
                "filename": filename,
                "page":     page_data["page"],
                "chunk_idx": j,
            })

    # Batch upsert (ChromaDB handles embedding internally via the embed function)
    BATCH = 100
    for i in range(0, len(all_chunks), BATCH):
        collection.upsert(
            ids=all_ids[i:i+BATCH],
            documents=all_chunks[i:i+BATCH],
            metadatas=all_metadatas[i:i+BATCH],
        )

    logger.info("Ingested '%s': %d pages, %d chunks", filename, len(pages), len(all_chunks))
    return {
        "doc_id":   doc_id,
        "filename": filename,
        "pages":    len(pages),
        "chunks":   len(all_chunks),
    }


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, doc_filter: Optional[str] = None, top_k: int = TOP_K) -> list[dict]:
    """
    Semantic similarity search.
    doc_filter: optional filename to restrict search to one document.
    Returns list of {text, source, page, score}.
    """
    collection = _get_collection()
    where_clause = {"filename": doc_filter} if doc_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count() or 1),
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs_list   = results.get("documents", [[]])[0]
    metas_list  = results.get("metadatas",  [[]])[0]
    dists_list  = results.get("distances",  [[]])[0]

    for doc_text, meta, dist in zip(docs_list, metas_list, dists_list):
        # Cosine distance → similarity score (0–1)
        score = round(1 - dist, 4)
        chunks.append({
            "text":   doc_text,
            "source": meta.get("filename", "unknown"),
            "page":   meta.get("page", 0),
            "score":  score,
        })

    # Filter out very low relevance chunks
    chunks = [c for c in chunks if c["score"] > 0.25]
    return chunks


# ── Document management ───────────────────────────────────────────────────────

def list_documents() -> list[dict]:
    """Return unique documents stored in the vector DB."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    seen = {}
    for meta in results["metadatas"]:
        doc_id   = meta.get("doc_id")
        filename = meta.get("filename", "unknown")
        if doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "filename": filename, "chunk_count": 0}
        seen[doc_id]["chunk_count"] += 1
    return list(seen.values())


def delete_document(doc_id: str) -> int:
    """Remove all chunks for a document. Returns number of chunks deleted."""
    collection = _get_collection()
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        return len(existing["ids"])
    return 0
