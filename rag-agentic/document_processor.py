"""
document_processor.py - Handles PDF parsing and text splitting.

Author: Ravi
Date: 2023-10-27

This module is responsible for loading PDF documents, extracting text,
and splitting the text into manageable chunks for embedding.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Updated import
from typing import List

def load_and_split_pdfs(file_paths: List[str]) -> List[str]:
    """
    Loads multiple PDF files, extracts their text, and splits it into chunks.

    Args:
        file_paths: A list of absolute paths to the PDF files.

    Returns:
        A list of text chunks (strings).
    """
    documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return [doc.page_content for doc in splits]
