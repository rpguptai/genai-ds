"""
document_processor.py - Handles PDF parsing, text splitting, and graph extraction.

Author: Ravi
Date: 2023-10-27 (Updated and Optimized)

This module is responsible for loading PDF documents, extracting text,
splitting it into chunks, and then extracting entities and relationships
to build a knowledge graph in Neo4j.
"""

import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError
from typing import List
from neo4j import Transaction

from graph_db_manager import GraphDBManager
from config import settings

log = logging.getLogger(__name__)

# --- Pydantic Models for Graph Extraction ---
class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node.")
    type: str = Field(description="The type or label of the node (e.g., Person, Organization, Concept).")

class Edge(BaseModel):
    source: str = Field(description="The ID of the source node.")
    target: str = Field(description="The ID of the target node.")
    type: str = Field(description="The type of the relationship between the source and target nodes.")

class Graph(BaseModel):
    nodes: List[Node] = Field(description="A list of all nodes in the graph.")
    edges: List[Edge] = Field(description="A list of all edges connecting the nodes.")

class DocumentGraphProcessor:
    def __init__(self):
        self.db_manager = GraphDBManager(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        self.llm = ChatOllama(model=settings.default_llm, format="json")
        self.parser = JsonOutputParser(pydantic_object=Graph)
        self.prompt = PromptTemplate(
            template="""
            From the text below, extract entities and their relationships.
            Format the output as a JSON object containing a list of nodes and a list of edges.
            A node should have a unique 'id' and a 'type'.
            An edge should have a 'source' node id, a 'target' node id, and a 'type'.
            The 'id' for a node should be the entity's name, normalized.
            
            Example Node: {{"id": "Paris", "type": "City"}}
            Example Edge: {{"source": "Eiffel Tower", "target": "Paris", "type": "LOCATED_IN"}}
            
            Text:
            {text}
            
            JSON Output:
            """,
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt | self.llm | self.parser

    def _create_graph_transaction(self, tx: Transaction, graph: Graph):
        """
        Internal method to execute graph creation within a Neo4j transaction.
        """
        node_types = {node.id: node.type for node in graph.nodes}

        # Create nodes
        for node in graph.nodes:
            tx.run(
                f"MERGE (n:`{node.type}` {{id: $id}})",
                id=node.id
            )

        # Create edges
        for edge in graph.edges:
            source_type = node_types.get(edge.source, "Entity")
            target_type = node_types.get(edge.target, "Entity")
            tx.run(
                f"""
                MATCH (a:`{source_type}` {{id: $source_id}})
                MATCH (b:`{target_type}` {{id: $target_id}})
                MERGE (a)-[r:`{edge.type}`]->(b)
                """,
                source_id=edge.source, target_id=edge.target
            )

    def load_split_and_process_pdfs(self, file_paths: List[str]) -> List[str]:
        """
        Loads PDFs, splits them, extracts a graph from each chunk,
        stores it in Neo4j, and returns the text chunks.
        """
        log.info("Loading and splitting documents...")
        documents = []
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        log.info(f"Generated {len(splits)} text chunks. Now extracting graph...")
        
        processed_chunks = []
        if self.db_manager.driver is None:
            log.error("Neo4j driver is not initialized. Skipping graph extraction.")
            return [doc.page_content for doc in splits]

        for i, doc in enumerate(splits):
            text_chunk = doc.page_content
            try:
                extracted_graph_dict = self.chain.invoke({"text": text_chunk})
                extracted_graph = Graph(**extracted_graph_dict)
                
                if extracted_graph.nodes or extracted_graph.edges:
                    # Use an explicit transaction block
                    with self.db_manager.driver.session() as session:
                        with session.begin_transaction() as tx:
                            self._create_graph_transaction(tx, extracted_graph)
                        log.info(f"Successfully stored graph from chunk {i+1}/{len(splits)}.")
                else:
                    log.info(f"No nodes or edges extracted from chunk {i+1}/{len(splits)}.")
            except ValidationError as e:
                log.error(f"Pydantic validation error for chunk {i+1}/{len(splits)}. LLM output may be malformed. Error: {e}")
            except Exception as e:
                log.error(f"Could not extract or store graph for chunk {i+1}/{len(splits)}. Error: {e}", exc_info=True)
            processed_chunks.append(text_chunk)

        log.info("Finished processing all chunks.")
        return processed_chunks

# Instantiate the processor for external use
document_graph_processor = DocumentGraphProcessor()

# Expose the main function for compatibility with rag_service.py
load_split_and_process_pdfs = document_graph_processor.load_split_and_process_pdfs
