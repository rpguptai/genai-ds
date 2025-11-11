# Multimodal RAG API with LangChain and Ollama

**Author:** Ravi
**Version:** 3.0.0

This project provides a production-ready REST API for a Retrieval-Augmented Generation (RAG) model that supports both text and image-based queries. It uses LangChain for orchestration and Ollama to run local Large Language Models (LLMs), offering a powerful and self-hosted solution for multimodal applications.

## Key Features

- **Multimodal Queries:** Supports text-based queries against a document corpus and direct queries with images.
- **Local LLM Integration:** Leverages Ollama to run local LLMs, including standard models like `llama2` and multimodal models like `llava`.
- **PDF Document Processing:** Extracts text and visual elements (charts, graphs) from PDF documents.
- **AI-Powered Image Summarization:** Generates descriptive summaries for visual elements, which are then indexed for retrieval.
- **Efficient Vector Store:** Uses FAISS for fast and scalable similarity searches.
- **Async & Parallel Processing:** Employs background tasks for document ingestion and a process pool for parallel image summarization to ensure the API remains responsive.
- **Production-Ready:** Features a modular structure, dependency injection, configuration management, and comprehensive logging.

## Project Structure

```
/rag-multimodel
|-- .env                    # Environment variables for configuration
|-- .idea/                  # IDE-specific settings
|-- data/                   # Default folder for storing uploaded PDF documents and extracted images
|-- tests/                  # Pytest unit tests
|   |-- __pycache__/
|   |-- conftest.py         # Pytest configuration and fixtures
|   `-- test_app.py         # Tests for API endpoints
|-- __pycache__/            # Python cache
|-- .pytest_cache/          # Pytest cache
|-- app.py                  # FastAPI application entry point and endpoints
|-- config.py               # Application configuration settings
|-- document_processor.py   # Handles PDF text and image extraction
|-- logger.py               # Centralized logging setup
|-- rag_service.py          # Core RAG logic for document processing and querying
|-- README.md               # This file
|-- requirements.txt        # Python dependencies
`-- pytest.ini              # Pytest configuration
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rag-multimodel
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Ollama:**
    - Install Ollama from [ollama.ai](https://ollama.ai/).
    - Pull the required LLMs. You'll need a standard model and a multimodal model.
        ```bash
        ollama pull llama2  # For text queries
        ollama pull llava   # For image summarization and queries
        ```

5.  **Create `.env` file:**
    Create a `.env` file in the root of the project to configure the application.
    ```
    DEFAULT_LLM=llama2
    MULTIMODAL_LLM=llava
    ```

6.  **Add initial documents (optional):**
    Place any initial PDF files you want to process in the `data` directory.

## Running the Application

To start the FastAPI application, run:

```bash
python app.py
```

The server will start, and on the first run, it will automatically process any PDF documents found in the `data` directory, creating a FAISS vector store.

Access the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`.

## API Endpoints

- **`POST /query`**
    - **Summary:** Query the RAG model with a text-based question.
    - **Request Body:**
        ```json
        {
          "query": "Your question here",
          "llm_name": "llama2" // Optional, defaults to DEFAULT_LLM
        }
        ```
    - **Response:**
        ```json
        {
          "response": "The model's answer based on the indexed documents."
        }
        ```

- **`POST /query-image`**
    - **Summary:** Query with an image and an optional text prompt.
    - **Request Body:** `multipart/form-data` with an image file and an optional text field.
        - `file`: The image file (`.png`, `.jpg`, etc.).
        - `text_query` (optional): A text question to ask about the image.
    - **Response:**
        ```json
        {
          "response": "The model's description or answer about the image."
        }
        ```

- **`POST /upload-documents`**
    - **Summary:** Upload one or more PDF documents for processing and indexing.
    - **Request Body:** `multipart/form-data` with one or more PDF files.
    - **Response:**
        ```json
        {
          "message": "PDF documents accepted and scheduled for processing.",
          "uploaded_files": ["file1.pdf"]
        }
        ```

## Running Tests

To run the unit tests, use `pytest`:

```bash
pytest
```
