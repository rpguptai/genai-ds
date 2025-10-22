"""
rag_service.py - Orchestrates the 4-agent system (Local Researcher, Web Researcher, Writer).

Author: Ravi
Date: 2023-10-27

This module acts as the orchestrator, calling specialized agents to gather
information from local documents and the web, and then a writer agent to
synthesize the final answer.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from document_processor import load_and_split_pdfs
from config import settings
# Import the agent creation functions
from agents import create_local_researcher, create_web_researcher, create_writer_agent

class RAGService:
    def __init__(self):
        """
        Initializes the RAG service, preparing the FAISS vector store.
        """
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        if os.path.exists(settings.vector_store_path):
            self.load_vector_store()
        
        # Caches for our specialized agents
        self.local_researchers = {}
        self.web_researchers = {}
        self.writers = {}

    def load_vector_store(self):
        """
        Loads the FAISS vector store from the local path.
        """
        self.vector_store = FAISS.load_local(
            settings.vector_store_path,
            self.embedding_function,
            allow_dangerous_deserialization=True
        )

    def add_documents(self, file_paths: list):
        """
        Processes and adds new documents to the FAISS vector store and saves it.
        """
        chunks = load_and_split_pdfs(file_paths)
        documents_to_add = [Document(page_content=chunk) for chunk in chunks]
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents_to_add, self.embedding_function)
        else:
            self.vector_store.add_documents(documents_to_add)
        
        self.vector_store.save_local(settings.vector_store_path)
        # Clear agent caches since the data has changed
        self.local_researchers = {}
        self.web_researchers = {}
        self.writers = {}

    def query(self, query_text: str, llm_name: str = None) -> dict:
        """
        Orchestrates the multi-agent workflow to answer the query.
        """
        if self.vector_store is None:
            return {"response": "Vector store is not initialized. Please upload documents first.", "sources": ""}

        final_llm_name = llm_name or settings.default_llm

        # Get or create the Local Researcher Agent (Agent 1)
        if final_llm_name not in self.local_researchers:
            self.local_researchers[final_llm_name] = create_local_researcher(final_llm_name, self)
        local_researcher = self.local_researchers[final_llm_name]

        # Get or create the Web Researcher Agent (Agent 2)
        if final_llm_name not in self.web_researchers:
            self.web_researchers[final_llm_name] = create_web_researcher(final_llm_name)
        web_researcher = self.web_researchers[final_llm_name]

        # Get or create the Writer Agent (Agent 4)
        if final_llm_name not in self.writers:
            self.writers[final_llm_name] = create_writer_agent(final_llm_name)
        writer = self.writers[final_llm_name]

        # Step 1: Local Research
        local_findings_raw = local_researcher.invoke({"input": query_text})
        local_results = local_findings_raw.get("output", "No local research findings.")

        # Step 2: Web Research
        web_findings_raw = web_researcher.invoke({"input": query_text})
        web_results = web_findings_raw.get("output", "No web research findings.")

        # Step 3: Combine findings for the Writer
        combined_context = f"**Local Document Search Results:**\n{local_results}\n\n**Web Search Results:**\n{web_results}"

        # Step 4: Writer Agent synthesizes the final answer
        final_answer = writer.invoke({"question": query_text, "context": combined_context})
        
        return {"response": final_answer, "sources": combined_context}
