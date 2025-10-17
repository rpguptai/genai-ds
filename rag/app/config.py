"""
Project configuration
Author: Ravi
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vector_store"
VECTORS_PATH = VECTOR_DIR / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
OLLAMA_API_URL = "http://localhost:11434"

