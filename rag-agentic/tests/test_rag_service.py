"""
test_rag_service.py - Unit tests for the 4-agent RAGService.

Author: Ravi
Date: 2023-10-27

This file contains unit tests for the RAGService, focusing on the
4-agent (Local Researcher, Web Researcher, Writer, Orchestrator) system.
"""

import pytest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure the app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_service import RAGService

@pytest.fixture
def mock_rag_service():
    """Fixture to create a mock RAGService for testing the multi-agent system."""
    with patch('rag_service.FAISS'), \
         patch('rag_service.HuggingFaceEmbeddings'), \
         patch('rag_service.load_and_split_pdfs'), \
         patch('rag_service.create_local_researcher') as mock_create_local, \
         patch('rag_service.create_web_researcher') as mock_create_web, \
         patch('rag_service.create_writer_agent') as mock_create_writer:

        # Mock Agent 1: Local Researcher
        mock_local_researcher = MagicMock()
        mock_local_researcher.invoke.return_value = {"output": "Local findings."}
        mock_create_local.return_value = mock_local_researcher

        # Mock Agent 2: Web Researcher
        mock_web_researcher = MagicMock()
        mock_web_researcher.invoke.return_value = {"output": "Web findings."}
        mock_create_web.return_value = mock_web_researcher

        # Mock Agent 4: Writer
        mock_writer = MagicMock()
        mock_writer.invoke.return_value = "Final synthesized answer."
        mock_create_writer.return_value = mock_writer

        # Instantiate the service (Orchestrator)
        service = RAGService()
        service.vector_store = MagicMock()
        
        yield service, mock_create_local, mock_create_web, mock_create_writer

def test_query_orchestration(mock_rag_service):
    """Tests that the query method correctly orchestrates the full agent workflow."""
    service, mock_create_local, mock_create_web, mock_create_writer = mock_rag_service
    
    result = service.query("test query", "test_llm")

    # 1. Assert the final response is from the writer
    assert result["response"] == "Final synthesized answer."

    # 2. Assert all agents were created correctly
    mock_create_local.assert_called_once_with("test_llm", service)
    mock_create_web.assert_called_once_with("test_llm")
    mock_create_writer.assert_called_once_with("test_llm")

    # 3. Assert the researcher agents were called with the initial query
    service.local_researchers["test_llm"].invoke.assert_called_once_with({"input": "test query"})
    service.web_researchers["test_llm"].invoke.assert_called_once_with({"input": "test query"})

    # 4. Assert the writer was called with the combined findings
    expected_context = "**Local Document Search Results:**\nLocal findings.\n\n**Web Search Results:**\nWeb findings."
    service.writers["test_llm"].invoke.assert_called_once_with({
        "question": "test query",
        "context": expected_context
    })

def test_add_documents_clears_all_caches(mock_rag_service):
    """Tests that adding documents clears all agent caches."""
    service, _, _, _ = mock_rag_service
    
    # Populate caches
    service.query("initial query", "test_llm")
    assert "test_llm" in service.local_researchers
    assert "test_llm" in service.web_researchers
    assert "test_llm" in service.writers

    # Add documents
    with patch('rag_service.FAISS.from_documents'), patch('rag_service.load_and_split_pdfs'):
        service.add_documents(["/path/to/new.pdf"])

    # Assert caches are cleared
    assert not service.local_researchers
    assert not service.web_researchers
    assert not service.writers
