"""
app.py - Main application file for the RAG model API.

Author: Ravi
Date: 2023-10-27 (Updated and Optimized)

This file sets up the FastAPI application, defines API endpoints for
querying the RAG model and uploading documents, and integrates with
the RAG service using proper dependency injection and lifespan events.
"""

import os
import uvicorn
import time
import threading
from functools import lru_cache
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request, Depends
from pydantic import BaseModel

from rag_service import RAGService
from config import settings
from logger import logger

# --- RAG Service Singleton ---
@lru_cache
def get_rag_service():
    return RAGService()

# --- Initial Document Processing Logic ---
def process_initial_documents():
    """Checks for and processes initial documents on startup."""
    try:
        rag_service = get_rag_service()
        vector_store_path = settings.vector_store_path
        # Run processing only if the vector store doesn't already exist.
        if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path):
            logger.info("Vector store not found or empty. Processing initial documents...")
            initial_docs_path = settings.data_folder
            if os.path.exists(initial_docs_path) and os.path.isdir(initial_docs_path):
                pdf_files = [os.path.join(initial_docs_path, f) for f in os.listdir(initial_docs_path) if f.endswith(".pdf")]
                if pdf_files:
                    logger.info(f"Found {len(pdf_files)} initial documents to process in the background.")
                    rag_service.add_documents(pdf_files)
                    logger.info("Initial document processing complete.")
                else:
                    logger.info("No initial PDF documents found in the data folder.")
            else:
                logger.warning(f"Data folder '{initial_docs_path}' not found. Creating it.")
                os.makedirs(initial_docs_path, exist_ok=True)
        else:
            logger.info("Existing vector store found. Skipping initial document processing.")
    except Exception as e:
        logger.error(f"An exception occurred during initial document processing: {e}", exc_info=True)

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup, run the initial document processing in a background thread.
    logger.info("Application startup: Scheduling initial document processing.")
    processing_thread = threading.Thread(target=process_initial_documents)
    processing_thread.start()
    yield
    # On shutdown
    logger.info("Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hybrid RAG Model API",
    description="API for a Retrieval-Augmented Generation model using a hybrid of vector search and a knowledge graph.",
    version="1.2.0",
    lifespan=lifespan
)

# --- Middleware for Logging ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(f"Response: {request.method} {request.url.path} | Status: {response.status_code} | Duration: {process_time:.2f}ms")
    return response

# --- API Models ---
class QueryRequest(BaseModel):
    query: str
    llm_name: str = None # Optional, will use default if not provided

class QueryResponse(BaseModel):
    response: str

class UploadResponse(BaseModel):
    message: str
    uploaded_files: List[str]

# --- Background Task for Document Upload ---
def process_uploaded_documents_task(file_paths: List[str]):
    try:
        logger.info(f"Background task started for {len(file_paths)} uploaded documents.")
        rag_service = get_rag_service()
        rag_service.add_documents(file_paths)
        logger.info(f"Background task finished for uploaded documents.")
    except Exception as e:
        logger.error(f"Background task for uploaded documents failed: {e}", exc_info=True)

# --- API Endpoints ---
@app.post("/query", response_model=QueryResponse, summary="Query the RAG model")
async def query_rag_model(request: QueryRequest, rag_service: RAGService = Depends(get_rag_service)):
    """
    Submits a query to the RAG model and retrieves a generated response.
    The model used can be specified in the request, otherwise the default is used.
    """
    logger.info(f"Query received for LLM '{request.llm_name or settings.default_llm}': '{request.query}'")
    try:
        response = rag_service.query(request.query, request.llm_name)
        return QueryResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the query.")

@app.post("/upload-documents", response_model=UploadResponse, summary="Upload new PDF documents")
async def upload_documents(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Uploads one or more PDF documents. The files are saved and then processed
    in the background to update the vector store and knowledge graph.
    """
    uploaded_file_paths = []
    os.makedirs(settings.data_folder, exist_ok=True)

    for file in files:
        if not file.filename.endswith(".pdf"):
            logger.warning(f"Invalid file type uploaded: {file.filename}. Only PDFs are supported.")
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a PDF.")
        
        file_path = os.path.join(settings.data_folder, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            uploaded_file_paths.append(file_path)
        except Exception as e:
            logger.error(f"Could not save uploaded file {file.filename}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save file {file.filename}.")
    
    # Use the standard FastAPI background tasks for uploads, as this is in a request-response cycle
    background_tasks.add_task(process_uploaded_documents_task, uploaded_file_paths)
    
    return UploadResponse(
        message="Documents accepted and scheduled for background processing.",
        uploaded_files=[os.path.basename(p) for p in uploaded_file_paths]
    )

# --- Main Entry Point for Development ---
if __name__ == "__main__":
    logger.info(f"Starting RAG API server on {settings.api_host}:{settings.api_port}")
    uvicorn.run("app:app", host=settings.api_host, port=settings.api_port, reload=True)
