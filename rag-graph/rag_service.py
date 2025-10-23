"""
rag_service.py - Core RAG service implementation using FAISS and Neo4j.

Author: Ravi
Date: 2023-10-27 (Updated and Optimized)

This module provides the core RAG functionality using FAISS as the vector store
and integrates Neo4j for graph-based context retrieval.
"""

import os
import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field, ValidationError

from document_processor import load_split_and_process_pdfs
from config import settings
from graph_db_manager import GraphDBManager

log = logging.getLogger(__name__)

# --- Pydantic Model for Query Entity Extraction ---
class QueryEntities(BaseModel):
    entities: List[str] = Field(description="A list of key entities mentioned in the query.")

class RAGService:
    def __init__(self):
        """
        Initializes the RAG service, loading the FAISS vector store if it exists,
        and setting up the Neo4j GraphDBManager.
        """
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        
        faiss_index_file = os.path.join(settings.vector_store_path, "index.faiss")
        if os.path.exists(faiss_index_file):
            try:
                self.load_vector_store()
                log.info("Successfully loaded existing FAISS vector store.")
            except RuntimeError as e:
                log.warning(f"Failed to load FAISS: {e}. Will re-process if triggered.")
                self.vector_store = None
        else:
            log.info(f"FAISS index file not found. Vector store will be initialized upon document processing.")

        self.db_manager = GraphDBManager(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )

        self.query_entity_llm = ChatOllama(model=settings.default_llm, format="json")
        self.query_entity_parser = JsonOutputParser(pydantic_object=QueryEntities)
        self.query_entity_prompt = PromptTemplate(
            template="Extract key entities from the following question. Format as a JSON object with a key 'entities' containing a list. Question: {question}",
            input_variables=["question"],
            partial_variables={"format_instructions": self.query_entity_parser.get_format_instructions()}
        )
        self.query_entity_chain = self.query_entity_prompt | self.query_entity_llm | self.query_entity_parser

        self._rag_llm_cache: Dict[str, ChatOllama] = {}
        self._get_rag_llm(settings.default_llm)

    def _get_rag_llm(self, llm_name: str) -> ChatOllama:
        if llm_name not in self._rag_llm_cache:
            self._rag_llm_cache[llm_name] = ChatOllama(model=llm_name)
        return self._rag_llm_cache[llm_name]

    def load_vector_store(self):
        self.vector_store = FAISS.load_local(
            settings.vector_store_path,
            self.embedding_function,
            allow_dangerous_deserialization=True
        )

    def add_documents(self, file_paths: list):
        chunks = load_split_and_process_pdfs(file_paths)
        documents_to_add = [Document(page_content=chunk) for chunk in chunks]
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents_to_add, self.embedding_function)
        else:
            self.vector_store.add_documents(documents_to_add)
        
        self.vector_store.save_local(settings.vector_store_path)
        log.info(f"FAISS vector store saved to {settings.vector_store_path}")

    def _get_graph_context(self, entities: List[str]) -> str:
        """
        Performs a flexible, case-insensitive search on the graph for the given entities.
        """
        if not entities:
            return ""

        context_statements = set()
        try:
            # Use a case-insensitive CONTAINS search for more flexible matching
            query = """
            UNWIND $entities AS entity_name
            MATCH (e)
            WHERE toLower(e.id) CONTAINS toLower(entity_name)
            OPTIONAL MATCH (e)-[r]-(n)
            RETURN e.id AS entity1_id, labels(e) AS entity1_labels, 
                   type(r) AS relationship_type, n.id AS entity2_id, labels(n) AS entity2_labels
            LIMIT 15
            """
            results = self.db_manager.run_query(query, parameters={"entities": entities})

            for record in results:
                entity1_id = record['entity1_id']
                entity1_labels = [label for label in record['entity1_labels'] if label != 'Node']
                
                if record['relationship_type'] and record['entity2_id']:
                    relationship_type = record['relationship_type'].replace('_', ' ').lower()
                    entity2_id = record['entity2_id']
                    entity2_labels = [label for label in record['entity2_labels'] if label != 'Node']
                    
                    stmt = f"{entity1_id} ({', '.join(entity1_labels)}) -> {relationship_type} -> {entity2_id} ({', '.join(entity2_labels)})"
                    context_statements.add(stmt)

        except Exception as e:
            log.error(f"Error querying Neo4j for graph context: {e}")

        if context_statements:
            return "\n--- Graph Context ---\n" + "\n".join(list(context_statements))
        return ""

    def query(self, query_text: str, llm_name: str = None) -> str:
        if self.vector_store is None:
            faiss_index_file = os.path.join(settings.vector_store_path, "index.faiss")
            if os.path.exists(faiss_index_file):
                try:
                    self.load_vector_store()
                    log.info("FAISS vector store loaded successfully after initial check.")
                except RuntimeError as e:
                    log.error(f"Failed to load FAISS vector store: {e}")
                    return "Vector store is not initialized and failed to load. Please ensure documents are processed correctly."
            else:
                return "Vector store not initialized. Please wait for initial processing to complete."

        rag_llm = self._get_rag_llm(llm_name or settings.default_llm)

        # 1. Extract entities from the query
        extracted_query_entities = []
        try:
            entities_output = self.query_entity_chain.invoke({"question": query_text})
            if isinstance(entities_output, dict) and "entities" in entities_output:
                extracted_query_entities = entities_output["entities"]
            elif isinstance(entities_output, QueryEntities):
                extracted_query_entities = entities_output.entities
            log.info(f"Extracted entities from query: {extracted_query_entities}")
        except Exception as e:
            log.error(f"Could not extract entities from query: {e}", exc_info=True)

        # 2. Get graph context
        graph_context = self._get_graph_context(extracted_query_entities)
        log.info(f"Retrieved graph context: {graph_context}")

        # 3. Get vector store context
        retriever = self.vector_store.as_retriever()
        vector_context_docs = retriever.invoke(query_text)
        vector_context = "\n".join([doc.page_content for doc in vector_context_docs])
        log.info(f"Retrieved vector context (first 200 chars): {vector_context[:200]}...")

        # 4. Combine contexts
        combined_context = f"""{vector_context}
{graph_context}""".strip()

        template = """
        Answer the question based only on the following context. 
        If the answer is not in the context, state that you don't know.

        Context:
        {context}

        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)

        chain = (
            RunnableParallel(context=RunnablePassthrough(), question=RunnablePassthrough())
            | prompt
            | rag_llm
            | StrOutputParser()
        )

        return chain.invoke({"context": combined_context, "question": query_text})
