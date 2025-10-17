RAG (Retrieval-Augmented Generation) with LangChain-style architecture and local Ollama LLM
======================================================================================

Author: Ravi

Overview
--------
This project implements a production-ready RAG system over local PDF documents using:

- FastAPI for API endpoints and automatic Swagger (/docs)
- Ollama (local LLM) as the default local LLM client (switchable)
- sentence-transformers for embeddings (small MiniLM model)
- FAISS for the vector store (with an in-memory backend option for testing)

The project supports:
- Indexing multiple PDFs from the `data/` folder
- Uploading new PDFs via API and indexing them
- Querying the RAG system with a choice of local LLM (e.g., `ollama`, `echo`)
- Pluggable vector store (FAISS on disk, in-memory for tests)
- Unit tests that mock heavy ML dependencies for fast execution

Repository layout
-----------------
- app/
  - config.py        -- configuration constants
  - llms.py          -- local LLM wrappers (OllamaClient, EchoClient)
  - pdf_loader.py    -- PDF loading utilities
  - embeddings.py    -- embeddings helper (sentence-transformers, lazy-loaded)
  - vector_store.py  -- pluggable vector store (faiss/in-memory)
  - rag.py           -- RAG orchestration (indexing + answer flow)
  - main.py          -- FastAPI app with endpoints (upload, query, reindex)
- data/              -- sample PDFs (sample1.pdf, sample2.pdf)
- requirements.txt   -- recommended dependencies
- tests/             -- pytest-based unit tests
- README.md          -- this file

Quick start (macOS, zsh)
------------------------
1) Create and activate a virtual environment (Python 3.13.5):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies. Note: some packages (faiss-cpu, sentence-transformers) can be large; for a fast test run you can install a minimal subset first:

```bash
# Minimal deps for running the API & tests (without heavy ML libs)
pip install -r requirements.txt
# If you want to avoid heavy ML packages during development, you may edit requirements.txt
# to only include: fastapi, uvicorn, httpx, pytest, pytest-asyncio, pydantic, python-multipart, numpy
```

3) Start the API server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger UI at: http://127.0.0.1:8000/docs

API endpoints
-------------
- GET /health
  - Simple health check.

- POST /reindex
  - Rebuilds the index from all PDF files in the `data/` folder.
  - Response: {"indexed": <num_documents_indexed>}

- POST /query
  - Request JSON: {"query": "...", "llm_name": "ollama", "top_k": 4}
  - `llm_name` currently supports `ollama` and `echo` (debugging). Default: `ollama`.
  - Response: {"answer": "...", "sources": [...]}

- POST /upload
  - Multipart file upload with key `file` (must be a PDF).
  - The PDF is saved into the `data/` folder and newly discovered pages are indexed.
  - Response: {"added": n, "skipped": m, "filename": "..."}

Switching local LLMs
--------------------
The API accepts an `llm_name` parameter in the `/query` payload. Supported values:
- `ollama` (calls the local Ollama HTTP API at the URL in `app/config.py`)
- `echo` (a deterministic debug LLM that echoes the prompt)

To add or switch other local LLMs, extend `app/llms.py` and update the LLMFactory mapping.

Testing
-------
Run unit tests with pytest. Tests are designed to avoid heavy ML downloads by mocking/mutating the embedding and LLM functions.

```bash
pytest -q
```

Notes & production considerations
--------------------------------
- The FAISS index persists to `vector_store/faiss_index` and metadata to `vector_store/faiss_index.meta`.
- In production consider:
  - Background indexing (Celery/RQ) and incremental updates
  - Deduplication and document versioning
  - Authentication, rate limiting, and request size limits
  - Monitoring and health checks for Ollama
  - Using a managed vector database (Milvus, Pinecone, Weaviate) for externalization

Contact
-------
Author: Ravi

If you want, I can:
- Add environment-based configuration (e.g., toggle vector backend via ENV)
- Add Dockerfile and docker-compose with Ollama and the API
- Wire a proper persistence layer for documents and metadata

Enjoy!

