# Hybrid RAG Model with Graph and Vector Search

**Author:** Ravi

This project implements an advanced Retrieval-Augmented Generation (RAG) model that combines traditional vector-based search with graph-based context retrieval. It uses LangChain, Ollama (for local LLMs), FAISS (for vector storage), and Neo4j (for the knowledge graph). The entire system is served via a robust FastAPI application.

## Features

*   **Hybrid RAG**: Leverages both semantic search (FAISS) and structured knowledge graph search (Neo4j) to provide richer, more accurate context to the LLM.
*   **Automated Knowledge Graph Creation**: Automatically extracts entities and relationships from PDF documents and populates a Neo4j graph database using transactional writes for data integrity.
*   **Local LLM Integration**: Utilizes Ollama to run local LLMs. The model for ingestion is set via an environment variable, while the query model can be specified at request time.
*   **Asynchronous & Optimized API**: A robust FastAPI application with non-blocking startup processing and background tasks for document uploads.
*   **Production-Ready Code**: Structured with proper dependency management, configuration (`.env`), logging, and optimized database interactions.
*   **Comprehensive Test Suite**: Includes a suite of `pytest` tests that mock external services, ensuring the API logic can be tested quickly and reliably.

## Project Structure

```
/rag-graph
|-- app.py                  # FastAPI application with lifespan events
|-- rag_service.py          # Core Hybrid RAG logic
|-- document_processor.py   # PDF processing and graph extraction
|-- graph_db_manager.py     # Manages connection and transactions with Neo4j
|-- config.py               # Centralized Pydantic configuration
|-- logger.py               # Application logger setup
|-- requirements.txt        # Python dependencies
|-- .env                    # Environment variables for configuration
|-- data/                   # Folder for your PDF documents
|   `-- sample.pdf
|-- tests/                  # Pytest test suite
|   `-- test_app.py
`-- README.md               # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rag-graph
    ```

2.  **Set up Neo4j:**
    The easiest way to run Neo4j is with Docker.
    ```bash
    docker run -d \
        --name neo4j-rag \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/password \
        neo4j:latest
    ```
    This starts a Neo4j instance with the username `neo4j` and password `password`. Access the Neo4j Browser at `http://localhost:7474`.

3.  **Set up Ollama and Pull Models:**
    *   Install Ollama from [ollama.ai](https://ollama.ai/).
    *   Pull the default model required for the background processing. This project is configured for `llama3`.
        ```bash
        ollama pull llama3
        ```

4.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

5.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Configure `.env` file:**
    Create a `.env` file in the root of the project. **Update the Neo4j password if you changed it.**
    ```env
    # --- Vector Store Path ---
    VECTOR_STORE_PATH=./faiss_index

    # --- LLM Configuration ---
    # This model MUST be available in your Ollama instance for the initial processing
    DEFAULT_LLM=llama3

    # --- Data Folder ---
    DATA_FOLDER=data

    # --- API Configuration ---
    API_HOST=0.0.0.0
    API_PORT=8000

    # --- Neo4j Connection Details ---
    NEO4J_URI="bolt://localhost:7687"
    NEO4J_USER="neo4j"
    NEO4J_PASSWORD="password"
    ```

7.  **Add PDF files:**
    Place your PDF files in the `data/` directory.

## Running the Application

First, ensure any old `faiss_index` directory is deleted to force processing on the first run.

Then, start the FastAPI application:
```bash
python app.py
```
On startup, the application will automatically begin processing any PDF files in the `data/` directory in a background thread. This populates both the FAISS vector store and the Neo4j knowledge graph. You should see logs indicating this process has started.

Access the API documentation at `http://localhost:8000/docs`.

## How the Hybrid RAG Works

1.  **Document Ingestion (on startup or upload)**: 
    *   PDFs are loaded and split into text chunks.
    *   For each chunk, a background task performs two actions:
        1.  **Vectorization**: An embedding is created and stored in a FAISS vector index for semantic search.
        2.  **Graph Extraction**: The LLM (`DEFAULT_LLM`) extracts key entities and relationships. These are saved to Neo4j in a single, atomic transaction to build the knowledge graph.

2.  **Query Time**: When a user submits a query to the `/query` endpoint:
    *   The LLM extracts key entities from the query text.
    *   These entities are used to query the Neo4j graph, retrieving relevant facts and relationships as structured "Graph Context".
    *   A semantic search is performed against the FAISS vector store to find relevant text chunks, which become the unstructured "Vector Context".
    *   Both contexts are combined into an enriched prompt and sent to the final LLM to generate a comprehensive and accurate answer.

## Running Tests

To run the test suite, which uses mocked services for speed and reliability, use `pytest`:
```bash
pytest
```
