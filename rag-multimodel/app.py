"""
app.py - Main application file for the Multimodal RAG API.

Author: Ravi
Date: 2023-11-15

This file sets up the FastAPI application, defines API endpoints for
querying the RAG model and uploading documents, and integrates with
the RAG service using proper dependency injection.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request, Depends, Form
from pydantic import BaseModel
from typing import List, Optional
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
    title="Multimodal RAG API with LangChain and Ollama",
    description="API for a RAG model supporting text and image queries from PDFs.",
    version="3.0.0",
)

# --- Middleware for Logging ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request received: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request finished: {request.method} {request.url.path} with status {response.status_code} in {process_time:.4f}s")
    return response

# --- API Models ---
class QueryRequest(BaseModel):
    query: str
    llm_name: str = settings.default_llm

class QueryResponse(BaseModel):
    response: str

class UploadResponse(BaseModel):
    message: str
    uploaded_files: List[str]

# --- Background Task for Document Processing ---
def process_documents_task(file_paths: List[str]):
    try:
        logger.info(f"Background task started: Processing {len(file_paths)} documents.")
        rag_service = get_rag_service()
        rag_service.add_documents(file_paths)
        logger.info("Background task finished: Successfully processed documents.")
    except Exception as e:
        logger.error(f"Background task failed: {e}", exc_info=True)

# --- API Endpoints ---

@app.post("/query", response_model=QueryResponse, summary="Query with text")
async def query_rag_model(request: QueryRequest, rag_service: RAGService = Depends(get_rag_service)):
    logger.info(f"Text query received: llm_name='{request.llm_name}', query='{request.query}'")
    try:
        response = rag_service.query(request.query, request.llm_name)
        logger.info("Text query successfully processed.")
        return QueryResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing text query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.post("/query-image", response_model=QueryResponse, summary="Query with an image and optional text")
async def query_image_rag_model(
    file: UploadFile = File(...),
    text_query: Optional[str] = Form(None),
    rag_service: RAGService = Depends(get_rag_service)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are supported.")

    temp_file_path = os.path.join(settings.data_folder, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"Image query received: image='{file.filename}', text_query='{text_query}'")
        response = rag_service.query_image(temp_file_path, text_query)
        logger.info("Image query successfully processed.")
        return QueryResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing image query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/upload-documents", response_model=UploadResponse, summary="Upload PDF documents for processing")
async def upload_documents(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    uploaded_file_paths = []
    os.makedirs(settings.data_folder, exist_ok=True)

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            logger.warning(f"Invalid file type uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported for upload.")
        
        file_path = os.path.join(settings.data_folder, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            uploaded_file_paths.append(file_path)
        except Exception as e:
            logger.error(f"Could not upload file {file.filename}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not upload file {file.filename}.")
    
    background_tasks.add_task(process_documents_task, uploaded_file_paths)
    logger.info(f"Scheduled background task to process {len(uploaded_file_paths)} PDF files.")
    
    return UploadResponse(
        message="PDF documents accepted and scheduled for processing.",
        uploaded_files=[os.path.basename(p) for p in uploaded_file_paths]
    )

# --- Server Initialization ---
if __name__ == "__main__":
    # The RAGService is initialized on first dependency call, so no need to do anything here.
    # The service will load the vector store if it exists.
    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
