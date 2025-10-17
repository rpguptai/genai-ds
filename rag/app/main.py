"""
API server for the RAG system
Author: Ravi

Endpoints:
- GET /health
- POST /reindex  -> rebuild index from all PDFs
- POST /query    -> body: {"query": "...", "llm_name": "ollama"}
- POST /upload   -> multipart file upload; saves to data/ and indexes

Swagger UI (OpenAPI) is automatically provided by FastAPI at /docs
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import uuid
import logging
from typing import Optional

from .rag import get_rag_instance
from .pdf_loader import load_pdf
from .embeddings import embed_texts
from .config import DATA_DIR
from .llms import _resolve_client_and_model
from .llms import LLMError

logger = logging.getLogger("rag_app")

app = FastAPI(title="RAG API", version="0.1.0", description="RAG over local PDFs with Ollama LLM. Author: Ravi")


class QueryRequest(BaseModel):
    query: str
    llm_name: Optional[str] = "ollama"
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reindex")
async def reindex():
    rag = get_rag_instance()
    count = await rag.index_all()
    return {"indexed": count}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="query is required")
    # Validate llm_name early to return a clear 400 error for unknown names
    try:
        _resolve_client_and_model(req.llm_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    rag = get_rag_instance()
    top_k = req.top_k
    try:
        res = await rag.answer(req.query, llm_name=req.llm_name, top_k=top_k or None)
    except LLMError as e:
        # Bubble a controlled 502 to the client; log the error server-side
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=502, detail=f"LLM failed: {str(e)}")
    return JSONResponse(content=res)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a new PDF file. The file is saved to DATA_DIR and indexed immediately.
    Returns number of pages added.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_name = f"{uuid.uuid4().hex}-{file.filename}"
    dest = DATA_DIR / new_name
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    # Load pages and index them
    pages = load_pdf(dest)
    if not pages:
        return {"added": 0}
    texts = [p["text"] or "" for p in pages]
    embeddings = embed_texts(texts)
    ids = [p["id"] for p in pages]
    metas = [{**p["metadata"], "text": p["text"]} for p in pages]

    rag = get_rag_instance()
    # dedupe
    existing = set(rag.store.get_ids())
    to_add_idx = [i for i, id_ in enumerate(ids) if id_ not in existing]
    add_vectors = [embeddings[i] for i in to_add_idx]
    add_metas = [metas[i] for i in to_add_idx]
    add_ids = [ids[i] for i in to_add_idx]
    if add_ids:
        rag.store.add(add_vectors, add_metas, add_ids)

    return {"added": len(add_ids), "skipped": len(ids) - len(add_ids), "filename": new_name}
