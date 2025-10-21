# RAG Model with LangChain and Ollama

**Author:** Ravi

This project implements a Retrieval-Augmented Generation (RAG) model using LangChain and Ollama for leveraging local Large Language Models (LLMs). It provides a FastAPI-based API for querying the model and uploading new PDF documents.

## Features

*   **Local LLM Integration:** Utilizes Ollama to run local LLMs, with the ability to switch between models via an API parameter.
*   **Multiple PDF Support:** Can process and index multiple PDF documents.
*   **Vector Store:** Uses ChromaDB as a vector store, which can be externalized.
*   **API with Swagger:** FastAPI provides an interactive API documentation (via Swagger UI) for easy testing of endpoints.
*   **Document Upload:** API endpoint to upload new PDF documents to the knowledge base.
*   **Production-Ready Code:** Structured with proper packages, configuration management, and documentation.
*   **Unit Tests:** Includes unit tests for the API endpoints.

## Project Structure

```
/rag
|-- app.py                  # FastAPI application
|-- rag_service.py          # Core RAG logic
|-- document_processor.py   # PDF processing
|-- config.py               # Configuration settings
|-- requirements.txt        # Python dependencies
|-- .env                    # Environment variables
|-- data/                   # Folder for PDF documents
|   |-- sample1.pdf
|   `-- sample2.pdf
|-- tests/                  # Unit tests
|   `-- test_app.py
`-- README.md               # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rag
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
    *   Install Ollama from [ollama.ai](https://ollama.ai/).
    *   Pull the desired local LLMs. For example, to get `llama2`:
        ```bash
        ollama pull llama2
        ```

5.  **Create `.env` file:**
    Create a `.env` file in the root of the project with the following content:
    ```
    VECTOR_STORE_PATH=./chroma_db
    DEFAULT_LLM=llama2
    DATA_FOLDER=data
    API_HOST=0.0.0.0
    API_PORT=8000
    ```

6.  **Add PDF files:**
    Place your PDF files in the `data` directory.

## Running the Application

To start the FastAPI application, run:

```bash
python app.py
```

The application will start, and on the first run, it will process any PDF files in the `data` directory and create the ChromaDB vector store.

Access the API documentation at `http://localhost:8000/docs`.

## API Endpoints

*   **POST /query**
    *   **Summary:** Query the RAG model.
    *   **Request Body:**
        ```json
        {
          "query": "Your question here",
          "llm_name": "llama2" 
        }
        ```
    *   **Response:**
        ```json
        {
          "response": "The model's answer."
        }
        ```

*   **POST /upload-documents**
    *   **Summary:** Upload new PDF documents.
    *   **Request Body:** `multipart/form-data` with one or more files.
    *   **Response:**
        ```json
        {
          "message": "Documents uploaded and processed successfully.",
          "uploaded_files": ["file1.pdf", "file2.pdf"]
        }
        ```

## Running Tests

To run the unit tests, use `pytest`:

```bash
pytest
```
