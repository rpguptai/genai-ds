"""
rag_service.py - Step 2: Full Multimodal RAG Pipeline (Corrected)

Author: Ravi
Date: 2023-11-10

This module orchestrates the complete multimodal RAG pipeline, ensuring that
document processing (extraction) is fully completed before summarization begins.
"""

import os
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_processor import extract_text_from_pdf, extract_visuals_from_pdf
from config import settings
from logger import logger

# --- AI Summarization (Worker Function) ---
def get_image_summary(image_path: str) -> Document:
    """
    Generates a summary for an image file using a multimodal LLM.
    This function is designed to be run in a separate process.
    """
    try:
        llm = ChatOllama(model=settings.multimodal_llm)
        
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "You are an expert data analyst. Describe the content of this chart, graph, or table in detail. What is its title? What data does it show? What are the key trends, patterns, and conclusions?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ]
            )
        ]
        
        summary = (llm | StrOutputParser()).invoke(prompt)
        logger.info(f"Generated summary for image {image_path}: {summary}")
        return Document(
            page_content=f"Summary of visual element: {summary}",
            metadata={"source_image": image_path}
        )
    except Exception as e:
        logger.error(f"Failed to generate summary for image {image_path}: {e}", exc_info=True)
        return None

# --- RAG Service Class ---
class RAGService:
    def __init__(self):
        """Initializes the RAG service and vector store."""
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        if os.path.exists(settings.vector_store_path):
            self.load_vector_store()

    def load_vector_store(self):
        """Loads the FAISS index from the configured path."""
        try:
            self.vector_store = FAISS.load_local(
                settings.vector_store_path,
                self.embedding_function,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS vector store loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FAISS vector store: {e}", exc_info=True)
            self.vector_store = None

    def add_documents(self, file_paths: list):
        """
        Orchestrates the full pipeline: text extraction, visual extraction,
        AI summarization, and indexing.
        """
        # --- Phase 1: Extraction ---
        logger.info("--- Phase 1: Starting Extraction ---")
        os.makedirs(settings.image_output_folder, exist_ok=True)
        
        all_text_docs = []
        all_image_paths = []

        for file_path in file_paths:
            # 1a. Extract text
            text_data = extract_text_from_pdf(file_path)
            for item in text_data:
                all_text_docs.append(Document(page_content=item["text"], metadata=item["metadata"]))
            
            # 1b. Extract and save visuals, getting their absolute paths
            image_paths = extract_visuals_from_pdf(file_path)
            all_image_paths.extend(image_paths)

        logger.info(f"Extraction complete. Found {len(all_text_docs)} text sections and {len(all_image_paths)} visual elements.")

        # --- Phase 2: Summarization ---
        logger.info(f"--- Phase 2: Starting AI Summarization for {len(all_image_paths)} images ---")
        image_summaries = []
        if all_image_paths:
            # Limit the number of parallel processes to avoid system overload
            max_workers = 5  # Adjust this value based on your system's capacity
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_image = {executor.submit(get_image_summary, path): path for path in all_image_paths}
                for future in as_completed(future_to_image):
                    result = future.result()
                    if result:
                        image_summaries.append(result)
        
        logger.info(f"AI summarization complete. Generated {len(image_summaries)} summaries.")

        # --- Phase 3: Indexing ---
        logger.info("--- Phase 3: Starting Indexing ---")
        all_docs_to_index = all_text_docs + image_summaries
        
        if not all_docs_to_index:
            logger.warning("No text or image summaries were generated. Nothing to index.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_docs_to_index)
        logger.info(f"Split all content into {len(chunks)} chunks for indexing.")

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embedding_function)
            logger.info("New FAISS vector store created.")
        else:
            self.vector_store.add_documents(chunks)
            logger.info(f"Added {len(chunks)} new chunks to the vector store.")
        
        self.vector_store.save_local(settings.vector_store_path)
        logger.info("FAISS vector store saved locally.")

    def query(self, query_text: str, llm_name: str = None) -> str:
        """Queries the RAG model with a given text and LLM."""
        if self.vector_store is None:
            return "Vector store is not initialized. Please upload documents first."

        final_llm_name = llm_name or settings.default_llm
        retriever = self.vector_store.as_retriever()
        llm = ChatOllama(model=final_llm_name)

        template = """
        You are an expert financial analyst. Answer the user's question based only on the following context, which may include text from a document and summaries of charts or graphs.

        Context:
        {context}

        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)

        chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain.invoke(query_text)

    def query_image(self, image_path: str, text_query: str = None) -> str:
        """
        Queries the multimodal LLM with an image and an optional text query.
        """
        llm = ChatOllama(model=settings.multimodal_llm)

        with open(image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

        prompt_messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": text_query or "Describe this image in detail."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ]
            )
        ]

        chain = llm | StrOutputParser()
        return chain.invoke(prompt_messages)
