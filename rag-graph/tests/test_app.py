import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os

# Mock the settings before importing the app
# This is crucial to prevent the app from trying to connect to real services on startup
from config import settings
settings.neo4j_uri = "bolt://mock-neo4j:7687"
settings.data_folder = "./test_data"
settings.vector_store_path = "./test_faiss_index"

from app import app, get_rag_service

# Pytest fixture to create a mock RAGService
@pytest.fixture
def mock_rag_service():
    mock = MagicMock()
    mock.query.return_value = "This is a mock response."
    mock.add_documents = MagicMock()
    return mock

# Pytest fixture to create a TestClient with the mocked service
@pytest.fixture
def client(mock_rag_service):
    # Use FastAPI's dependency overrides to replace the real RAGService with our mock
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    
    # Create a directory for test uploads
    os.makedirs(settings.data_folder, exist_ok=True)
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up the test directory and dependency override
    os.rmdir(settings.data_folder)
    app.dependency_overrides = {}


# --- Test Cases ---

def test_query_rag_model_success(client):
    """Test the /query endpoint for a successful response."""
    response = client.post("/query", json={"query": "What is RAG?"})
    assert response.status_code == 200
    assert response.json() == {"response": "This is a mock response."}

def test_query_rag_model_with_llm_name(client, mock_rag_service):
    """Test that the llm_name from the request is passed to the service."""
    client.post("/query", json={"query": "Test query", "llm_name": "test_model"})
    # Check that the service's query method was called with the correct arguments
    mock_rag_service.query.assert_called_with("Test query", "test_model")

def test_query_rag_model_service_error(client, mock_rag_service):
    """Test the /query endpoint when the RAG service raises an exception."""
    mock_rag_service.query.side_effect = Exception("Service failure")
    response = client.post("/query", json={"query": "This will fail"})
    assert response.status_code == 500
    assert "internal error" in response.json()["detail"]

def test_upload_documents_success(client):
    """Test successful upload of a PDF document."""
    # Create a dummy PDF file for upload
    dummy_pdf_content = b"%PDF-1.5..."
    files = {"files": ("test.pdf", dummy_pdf_content, "application/pdf")}
    
    response = client.post("/upload-documents", files=files)
    
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["message"] == "Documents accepted and scheduled for background processing."
    assert json_response["uploaded_files"] == ["test.pdf"]

def test_upload_documents_invalid_file_type(client):
    """Test uploading a non-PDF file, which should be rejected."""
    dummy_txt_content = b"this is not a pdf"
    files = {"files": ("test.txt", dummy_txt_content, "text/plain")}
    
    response = client.post("/upload-documents", files=files)
    
    assert response.status_code == 400
    assert "is not a PDF" in response.json()["detail"]

def test_root_path(client):
    """Test that the root path is not configured (as expected for an API)."""
    response = client.get("/")
    assert response.status_code == 404 # Or whatever is expected for the root path
