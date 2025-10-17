"""
API tests for the RAG FastAPI app
Author: Ravi

These tests mock heavy components (embeddings and LLM) so they run quickly without
large model downloads. They use FastAPI's TestClient (async via httpx) to call endpoints.
"""
import asyncio
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

import app.main as main_app
from app.rag import RAG
from app.vector_store import VectorStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def test_app(monkeypatch, tmp_path):
    # Create a fresh RAG with in-memory store and monkeypatch the module singleton
    rag = RAG(vector_backend="memory")
    # Monkeypatch get_rag_instance to return our rag
    monkeypatch.setattr(main_app, "get_rag_instance", lambda: rag)

    # Monkeypatch embed_texts to return simple embeddings (e.g., length and sum)
    def fake_embed(texts):
        return [[float(len(t)), float(sum(ord(c) for c in t) % 10)] for t in texts]

    monkeypatch.setattr("app.embeddings.embed_texts", fake_embed)

    # Monkeypatch generate_text to return a fixed answer
    async def fake_generate(prompt, llm_name="ollama", **kwargs):
        await asyncio.sleep(0)
        return "This is a fake answer"

    monkeypatch.setattr("app.llms.generate_text", fake_generate)

    # return the FastAPI app
    yield main_app.app


@pytest.mark.asyncio
async def test_health(test_app):
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        r = await ac.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_reindex_and_query(test_app):
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        # Reindex (data dir may be empty; this should succeed)
        r = await ac.post("/reindex")
        assert r.status_code == 200
        # Query with a sample prompt
        q = {"query": "What is in the documents?", "llm_name": "echo"}
        r2 = await ac.post("/query", json=q)
        assert r2.status_code == 200
        body = r2.json()
        assert "answer" in body
        assert body["answer"] == "This is a fake answer"


@pytest.mark.asyncio
async def test_upload_non_pdf(test_app):
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        files = {"file": ("test.txt", b"hello world", "text/plain")}
        r = await ac.post("/upload", files=files)
        assert r.status_code == 400


