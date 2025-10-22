"""
app.py - Main application file for the multi-agent RAG API.

Author: Ravi
Date: 2023-10-27

This file sets up the FastAPI application and defines API endpoints that return
a final answer and the sources used by the agentic system.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request, Depends
from pydantic import BaseModel
from typing import List
import os
import uvicorn
import time
from functools import lru_cache

from rag_service import RAGService
from config import settings
from logger import logger

# Use lru_cache to create a singleton instance of the RAGService.
@lru_cache
def get_rag_service():
    return RAGService()

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent RAG API (Researcher-Writer)",
    description="API for a multi-agent RAG model using a Researcher-Writer pattern.",
    version="4.0.0", # Final version
)

# Add a middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request received: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request finished: {request.method} {request.url.path} with status {response.status_code} in {process_time:.4f}s")
    return response

# --- API Models (Updated for Transparency) ---
class QueryRequest(BaseModel):
    query: str
    llm_name: str = settings.default_llm

class QueryResponse(BaseModel):
    response: str
    sources: str

class UploadResponse(BaseModel):
    message: str
    uploaded_files: List[str]

# --- Background Task Wrapper ---
def process_documents_task(file_paths: List[str]):
    try:
        logger.info(f"Background task started: Processing {len(file_paths)} documents.")
        rag_service = get_rag_service()
        rag_service.add_documents(file_paths)
        logger.info(f"Background task finished: Successfully processed documents.")
    except Exception as e:
        logger.error(f"Background task failed: {e}", exc_info=True)

# --- API Endpoints (Updated) ---

@app.post("/query", response_model=QueryResponse, summary="Query the RAG model")
async def query_rag_model(request: QueryRequest, rag_service: RAGService = Depends(get_rag_service)):
    """
    Submits a query to the multi-agent RAG system and retrieves a synthesized response and its sources.
    """
    logger.info(f"Query received: llm_name='{request.llm_name}', query='{request.query}'")
    try:
        # The service now returns a dictionary with the response and sources
        result = rag_service.query(request.query, request.llm_name)
        logger.info("Query successfully processed.")
        return QueryResponse(response=result["response"], sources=result["sources"])
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the query.")

@app.post("/upload-documents", response_model=UploadResponse, summary="Upload documents for background processing")
async def upload_documents(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Uploads PDF documents for background processing.
    """
    uploaded_file_paths = []
    os.makedirs(settings.data_folder, exist_ok=True)

    for file in files:
        if not file.filename.endswith(".pdf"):
            logger.warning(f"Invalid file type uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF.")
        
        file_path = os.path.join(settings.data_folder, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            uploaded_file_paths.append(file_path)
        except Exception as e:
            logger.error(f"Could not upload file {file.filename}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not upload file {file.filename}.")
    
    background_tasks.add_task(process_documents_task, uploaded_file_paths)
    logger.info(f"Scheduled background task to process {len(uploaded_file_paths)} files.")
    
    return UploadResponse(
        message="Documents accepted and scheduled for processing in the background.",
        uploaded_files=[os.path.basename(p) for p in uploaded_file_paths]
    )

# To run the application (for development)
if __name__ == "__main__":
    rag_service_instance = get_rag_service()
    if not os.path.exists(settings.vector_store_path) or not os.listdir(settings.vector_store_path):
        logger.info("Vector store not found or empty. Processing initial documents...")
        initial_docs_path = settings.data_folder
        if os.path.exists(initial_docs_path) and os.path.isdir(initial_docs_path):
            pdf_files = [os.path.join(initial_docs_path, f) for f in os.listdir(initial_docs_path) if f.endswith(".pdf")]
            if pdf_files:
                logger.info(f"Found {len(pdf_files)} initial documents to process.")
                rag_service_instance.add_documents(pdf_files)
                logger.info("Initial documents processed and added to the vector store.")
            else:
                logger.info("No initial PDF documents found in the data folder.")
        else:
            logger.info(f"Data folder '{initial_docs_path}' not found. Skipping initial document processing.")
            os.makedirs(initial_docs_path, exist_ok=True)

    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
