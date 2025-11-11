"""
test_app.py - Unit tests for the FastAPI application.

Author: Ravi
Date: 2023-11-15

This file contains updated unit tests for the production-ready API endpoints,
using pytest and httpx. It correctly mocks the RAGService and background tasks.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure the app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, get_rag_service
from rag_service import RAGService

# This fixture creates a mock RAGService for each test
@pytest.fixture
def mock_rag_service():
    service = MagicMock(spec=RAGService)
    service.query.return_value = "Mocked RAG response"
    service.query_image.return_value = "Mocked image query response"
    service.add_documents = MagicMock()
    return service

# This fixture automatically overrides the dependency for every test
@pytest.fixture(autouse=True)
def override_dependency(mock_rag_service):
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    yield
    app.dependency_overrides = {}

@pytest.mark.asyncio
async def test_query_endpoint(mock_rag_service):
    """Tests the /query endpoint with a mocked RAG service."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/query", json={"query": "test query", "llm_name": "test_llm"})
    
    assert response.status_code == 200
    assert response.json() == {"response": "Mocked RAG response"}
    mock_rag_service.query.assert_called_once_with("test query", "test_llm")

@pytest.mark.asyncio
async def test_query_image_endpoint(mock_rag_service):
    """Tests the /query-image endpoint with a mocked RAG service."""
    dummy_image_content = b"fake image data"
    files = {'file': ('test.png', dummy_image_content, 'image/png')}
    data = {'text_query': 'test image query'}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/query-image", files=files, data=data)

    assert response.status_code == 200
    assert response.json() == {"response": "Mocked image query response"}
    # We can't easily check the file path argument since it's a temporary file,
    # so we check that the service was called with a path and the correct query.
    mock_rag_service.query_image.assert_called_once()
    args, _ = mock_rag_service.query_image.call_args
    assert isinstance(args[0], str)
    assert args[1] == 'test image query'


@pytest.mark.asyncio
async def test_upload_documents_endpoint():
    """Tests the /upload-documents endpoint to ensure it schedules a background task."""
    dummy_file_content = b"This is a test pdf."
    files = {'files': ('test.pdf', dummy_file_content, 'application/pdf')}

    with patch('app.process_documents_task') as mock_process_task:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/upload-documents", files=files)

        assert response.status_code == 200
        assert response.json()["message"] == "PDF documents accepted and scheduled for processing."
        assert "test.pdf" in response.json()["uploaded_files"]
        
        mock_process_task.assert_called_once()
        args, _ = mock_process_task.call_args
        assert isinstance(args[0], list)
        assert os.path.basename(args[0][0]) == 'test.pdf'

@pytest.mark.asyncio
async def test_upload_non_pdf_document():
    """Tests that uploading a non-PDF file returns a 400 error."""
    dummy_file_content = b"This is not a pdf."
    files = {'files': ('test.txt', dummy_file_content, 'text/plain')}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/upload-documents", files=files)

    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]
