# GenAI RAG Implementations

This repository contains a collection of Retrieval-Augmented Generation (RAG) projects, each demonstrating a different architectural pattern.

## Projects

### 1. Standard RAG (`/rag`)

A foundational RAG implementation that uses a local LLM and a vector database for document retrieval.

- **Core Technologies**: LangChain, Ollama, ChromaDB, FastAPI
- **Features**:
    - Indexes and retrieves information from multiple PDF documents.
    - Exposes a RESTful API for querying and document uploads.
    - Supports switching between different local LLMs at query time.

### 2. Hybrid RAG with Knowledge Graph (`/rag-graph`)

An advanced RAG model that combines vector-based search with a graph-based knowledge retrieval system.

- **Core Technologies**: LangChain, Ollama, FAISS, Neo4j, FastAPI
- **Features**:
    - **Hybrid Retrieval**: Leverages both semantic search (FAISS) and structured knowledge graph search (Neo4j) for richer context.
    - **Automated Knowledge Graph Creation**: Extracts entities and relationships from documents to build a Neo4j graph.
    - Asynchronous API with background processing for document ingestion.

### 3. Multi-Agent RAG (`/rag-agentic`)

A sophisticated, multi-agent RAG system that separates the tasks of research and writing for improved answer quality.

- **Core Technologies**: LangChain, Ollama, FAISS, FastAPI
- **Features**:
    - **4-Agent Architecture**:
        1.  **Local Document Researcher**: Searches the local vector store.
        2.  **Web Researcher**: Searches the web for real-time information.
        3.  **Orchestrator**: Manages the research agents and compiles a research brief.
        4.  **Writer**: Synthesizes the research brief into a final answer.
    - Provides transparency by including the sources used for each answer.

### 4. Multimodal RAG (`/rag-multimodel`)

A RAG implementation that supports both text and image-based queries.

- **Core Technologies**: LangChain, Ollama, FAISS, FastAPI
- **Features**:
    - **Multimodal Queries**: Can answer questions about text documents and images.
    - **AI-Powered Image Summarization**: Generates descriptive summaries for visual elements in documents, making them searchable.
    - Supports multimodal LLMs like LLaVA.
