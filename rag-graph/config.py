"""
config.py - Centralized and validated configuration settings.

Author: Ravi
Date: 2023-10-27

This file uses Pydantic's BaseSettings to define and validate all
configuration variables for the application, loading them from environment
variables and a .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    # Use model_config instead of the deprecated inner Config class
    model_config = ConfigDict(env_file=".env", extra='ignore')

    # --- Vector Store Configuration ---
    vector_store_path: str = "./faiss_index" # Standardized name

    # --- LLM Configuration ---
    default_llm: str = "llama3"

    # --- Data Configuration ---
    data_folder: str = "data"

    # --- API Configuration ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Neo4j Configuration ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

# Create a single, validated settings instance
settings = Settings()
