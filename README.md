# BarPal.ai — Legal Document Q&A for Startups & SMEs

> Private, AI-powered legal document analysis powered by **Gemma 4 (31B)** + **RAG**.  
> All documents stay on your machine. Only the top-K retrieved excerpts are sent to the model API.

---

## Features

- **Drag-and-drop upload** — PDF and DOCX (NDAs, contracts, bylaws, term sheets)
- **Local RAG pipeline** — `all-MiniLM-L6-v2` embeddings + ChromaDB vector store (fully offline)
- **Gemma 4 31B answers** — grounded in your documents, with source citations and page numbers
- **Risk flagging** — automatically highlights high-risk clauses (non-competes, IP assignment, unlimited liability)
- **Document library** — manage, filter, and delete documents from the sidebar
- **Per-document Q&A** — ask questions scoped to one specific contract
- **RAGAS evaluation** — built-in LLM-as-judge scoring (Faithfulness, Answer Relevance, Context Precision)

---

## Architecture

```
Browser (HTML / CSS / JS)
        │
        ▼
FastAPI Backend (Python)          ← main.py
        │
  ┌─────┴──────────┐
  │   RAG Pipeline  │             ← rag_pipeline.py
  │                 │
  │  pdfplumber     │  ← parse PDF pages
  │  python-docx    │  ← parse DOCX paragraphs
  │  Custom Chunker │  ← recursive split (800 chars, 100 overlap)
  │  MiniLM-L6-v2  │  ← embed locally (sentence-transformers, 384-dim)
  │  ChromaDB       │  ← store & retrieve vectors (HNSW, cosine, local disk)
  └─────┬──────────┘
        │  top-K chunks (default: 5, min score: 0.25)
        ▼
  NVIDIA Inference API            ← gemma_client.py
  Model: google/gemma-4-31b-it
  Endpoint: integrate.api.nvidia.com
  Mode: SSE streaming + thinking enabled
        │
        ▼ (evaluation only)
  RAGAS-style LLM-as-judge        ← evaluator.py
  3 metrics run concurrently via asyncio.gather
  Overall = harmonic mean
```

---

## Quick Start

### 1. Get an NVIDIA API key

Sign up at [build.nvidia.com](https://build.nvidia.com) and generate a free API key (`nvapi-...`).

### 2. Configure environment

```bash
cd "legal llm/backend"
cp .env.example .env
# Open .env and set:  NVIDIA_API_KEY=nvapi-<your-key-here>
```

### 3. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

> **Note:** First run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB). One-time download, runs fully locally after that.

### 4. Start the server

```bash
python main.py
# → Server running at http://localhost:8000
```

### 5. Open the app

Go to **http://localhost:8000** — the FastAPI server serves the frontend automatically.  
Interactive API docs: **http://localhost:8000/docs**

---

## Usage

1. **Upload** a legal document (PDF or DOCX) via drag-and-drop or the Browse button
2. **Wait** for indexing — you'll see chunk count in the toast notification
3. **Ask** a plain-English question, e.g.:
   - *"What is the non-compete duration and geographic scope?"*
   - *"Who owns the IP created during employment?"*
   - *"What are the termination conditions and notice periods?"*
4. **Review** the answer with source citations (filename + page number)
5. **Filter** by document using the dropdown to query a specific contract
6. **Evaluate** pipeline quality using the RAGAS tab — runs Faithfulness, Answer Relevance, and Context Precision in parallel

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | — | **Required.** Your NVIDIA Inference API key (`nvapi-...`) |
| `GEMMA_MODEL` | `google/gemma-4-31b-it` | Model ID (e.g. `google/gemma-4-27b-it` for smaller) |
| `NVIDIA_INVOKE_URL` | `https://integrate.api.nvidia.com/v1/chat/completions` | Inference endpoint |
| `CHROMA_DB_PATH` | `./chroma_db` | Local directory where ChromaDB persists vectors |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | `5` | Number of passages retrieved per query |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server health check |
| `POST` | `/upload` | Upload and ingest a PDF or DOCX file |
| `POST` | `/query` | Ask a question; returns answer + source citations |
| `POST` | `/evaluate` | Run full RAGAS evaluation on a question |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{doc_id}` | Remove a document and all its indexed chunks |

---

## RAGAS Evaluation

The `/evaluate` endpoint runs three LLM-as-judge metrics concurrently using Gemma 4 as the judge:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer contain only information from the retrieved passages? (hallucination check) |
| **Answer Relevance** | Does the answer directly address the question asked? |
| **Context Precision** | Were the retrieved document passages actually relevant to the question? |

**Overall score** = harmonic mean of all three. This penalises any single weak dimension — a pipeline that retrieves perfectly but hallucinates still scores low.

---

## Project Structure

```
legal llm/
├── backend/
│   ├── main.py           # FastAPI app — REST endpoints
│   ├── rag_pipeline.py   # PDF/DOCX ingestion, chunking, ChromaDB retrieval
│   ├── gemma_client.py   # NVIDIA API wrapper, SSE stream parser
│   ├── evaluator.py      # RAGAS-style LLM-as-judge evaluation
│   ├── requirements.txt
│   ├── .env.example
│   └── chroma_db/        # Local vector store (auto-created)
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

---

## Privacy & Security

- **Documents never leave your machine** — only the top-K retrieved text snippets are sent to the NVIDIA API
- **Local embeddings** — `all-MiniLM-L6-v2` runs entirely on your hardware
- **ChromaDB** stores vectors locally in `./chroma_db/` — nothing is uploaded to a cloud database
- **Secrets** are managed via `.env` — never committed to version control (`.env` is gitignored)

---

## Disclaimer

BarPal.ai provides information only, not legal advice. Always consult a qualified attorney for legal decisions.
