"""
RAG orchestration
Author: Ravi

Builds/maintains a vector index from PDFs and answers queries by retrieving top-k passages
and calling a local LLM (e.g. Ollama).

Design notes:
- VectorStore is pluggable (faiss/in-memory) and persists FAISS on disk.
- Embeddings are provided by sentence-transformers helper.
- The RAG API provides `index_all()` to (re)build the index from the data folder and
  `answer(query, llm_name)` to perform a retrieve-and-generate flow.
"""
from typing import List, Dict, Optional
from .pdf_loader import load_all_pdfs
from .embeddings import embed_texts, get_embedding_dim
from .vector_store import VectorStore, SearchResult
from .llms import generate_text
from .config import DATA_DIR, TOP_K
from pathlib import Path
import asyncio


class RAG:
    def __init__(self, vector_backend: str = "faiss"):
        dim = get_embedding_dim()
        self.store = VectorStore(dim=dim, backend=vector_backend)
        self.lock = asyncio.Lock()

    async def index_all(self, folder: Optional[Path] = None) -> int:
        """Index all PDFs from `folder` or default data folder.
        Returns the number of documents indexed.
        """
        folder = folder or DATA_DIR
        docs = load_all_pdfs(folder)
        if not docs:
            return 0
        texts = [d["text"] or "" for d in docs]
        embeddings = embed_texts(texts)
        ids = [d["id"] for d in docs]
        # store text inside metadata so it can be used for building context later
        metas = [{**d["metadata"], "text": d["text"]} for d in docs]
        # Simple add - in production you may want to dedupe by id
        async with self.lock:
            self.store.add(embeddings, metas, ids)
        return len(ids)

    async def answer(self, query: str, llm_name: str = "ollama", top_k: int = TOP_K) -> Dict:
        """Retrieve top_k passages and call the LLM to generate an answer.

        Returns a dict with keys: answer (str), sources (list of metadata dicts)
        """
        q_emb = embed_texts([query])[0]
        hits: List[SearchResult] = self.store.search(q_emb, k=top_k)
        # Build context
        ctx_parts = []
        sources = []
        for h in hits:
            meta = h.metadata
            src = meta.get("source")
            page = meta.get("page_number")
            text = meta.get("text", "")
            ctx_parts.append(f"Source: {src} (page {page})\n{text}")
            sources.append({"id": h.id, "score": h.score, "metadata": meta})

        context = "\n\n".join([p for p in ctx_parts if p])
        prompt = f"Use the following context to answer the question.\n\n{context}\n\nQuestion: {query}\nAnswer:"
        # Call LLM
        answer = await generate_text(prompt, llm_name)
        return {"answer": answer, "sources": sources}


# Provide a module-level singleton RAG instance for the web app to reuse.
_rag_instance: Optional[RAG] = None


def get_rag_instance() -> RAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAG()
    return _rag_instance
