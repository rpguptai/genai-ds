# Advanced Multi-Agent RAG System

**Author:** Ravi

This project implements a sophisticated, multi-agent Retrieval-Augmented Generation (RAG) system based on your design. It uses a 4-part architecture to ensure robust, high-quality answers by separating the concerns of research and writing.

## Core Architecture: 4-Agent Orchestrated System

This project uses a clear and powerful multi-agent pattern to process user queries:

1.  **Agent 1: Local Document Researcher**
    *   A specialized agent whose only job is to search the local `FAISS` vector store for information relevant to the user's query.

2.  **Agent 2: Web Researcher**
    *   A second specialized agent whose only job is to search the web (simulated as `google.com` and `bing.com`) for real-time information and general context.

3.  **Agent 3: The Orchestrator (`RAGService`)**
    *   This is not an LLM-based agent, but a reliable, code-based manager. It receives the initial query and orchestrates the entire workflow:
        1.  Calls the **Local Researcher**.
        2.  Calls the **Web Researcher**.
        3.  Combines the findings from both into a comprehensive research brief.
        4.  Passes the brief and the original question to the **Writer Agent**.

4.  **Agent 4: The Writer**
    *   A final, specialized agent with no tools. Its only purpose is to receive the comprehensive research brief from the Orchestrator and synthesize it into a high-quality, well-written final answer.

This design ensures that both local and web searches are always performed and that the final answer is based on a complete set of information.

## Features

*   **Advanced Multi-Agent System:** Implements a robust 4-agent architecture (2 Researchers, 1 Orchestrator, 1 Writer).
*   **Local LLM Integration:** Utilizes Ollama to run local LLMs (e.g., `llama3.2`), with the ability to switch models via an API parameter.
*   **FAISS Vector Store:** Uses `FAISS` for efficient, local vector storage.
*   **RESTful API with Swagger:** Built with FastAPI, providing interactive API documentation and transparent outputs, including the sources used for the answer.
*   **Production-Ready Code:** Features a clean project structure, centralized configuration, structured logging, and comprehensive unit tests that verify the multi-agent workflow.

## Project Structure

```
/rag-agentic
|-- app.py                  # FastAPI application entry point
|-- rag_service.py          # The Orchestrator (Agent 3)
|-- agents.py               # Definitions for Researcher and Writer agents
|-- document_processor.py   # Handles PDF loading and text splitting
|-- config.py               # Centralized Pydantic configuration
|-- requirements.txt        # All Python dependencies
|-- .env                    # Environment variables
|-- data/                   # Folder for your PDF documents
|-- tests/                  # Unit tests for the API and service layer
`-- README.md               # This file
```

## Final Setup and Running

1.  **Install a Fortran Compiler (if on macOS):**
    ```bash
    brew install gfortran
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    *   First, ensure no other service is running on port 8000.
    *   Then, start the app:
    ```bash
    python app.py
    ```

4.  **Use the API:**
    *   Open your browser to `http://localhost:8000/docs`.
    *   Use the `/query` endpoint. The response will now include the final `response` and the `sources` used by the Writer agent, so you can verify that both local and web searches were performed.

This project is now complete and correctly implements the advanced architecture you designed. Thank you for your patience and guidance.
