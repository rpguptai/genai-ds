"""
rag_service.py - Core RAG service implementation using FAISS.

Author: Ravi
Date: 2023-10-27

This module provides the core RAG functionality using FAISS as the vector store
to avoid the dependency conflicts encountered with ChromaDB.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from document_processor import load_and_split_pdfs
from config import settings

class RAGService:
    def __init__(self):
        """
        Initializes the RAG service, loading the FAISS vector store if it exists.
        """
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        if os.path.exists(settings.vector_store_path):
            self.load_vector_store()

    def load_vector_store(self):
        """Loads the FAISS index from the configured path."""
        self.vector_store = FAISS.load_local(
            settings.vector_store_path,
            self.embedding_function,
            allow_dangerous_deserialization=True
        )

    def add_documents(self, file_paths: list):
        """
        Processes, adds new documents to the FAISS vector store, and saves it.
        """
        chunks = load_and_split_pdfs(file_paths)
        documents_to_add = [Document(page_content=chunk) for chunk in chunks]
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents_to_add, self.embedding_function)
        else:
            self.vector_store.add_documents(documents_to_add)
        
        self.vector_store.save_local(settings.vector_store_path)

    def query(self, query_text: str, llm_name: str = None) -> str:
        """
        Queries the RAG model with a given text and LLM.
        """
        if self.vector_store is None:
            return "Vector store is not initialized. Please upload documents first."

        # Use the provided llm_name or fall back to the default from settings
        final_llm_name = llm_name or settings.default_llm

        retriever = self.vector_store.as_retriever()
        llm = ChatOllama(model=final_llm_name)

        template = """
        Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)

        # Modernized RAG chain using RunnableParallel
        chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain.invoke(query_text)
