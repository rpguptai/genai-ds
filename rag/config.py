"""
config.py - Centralized and validated configuration settings.

Author: Ravi
Date: 2023-10-27

This file uses Pydantic's BaseSettings to define and validate all
configuration variables for the application, loading them from environment
variables and a .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict # Corrected import

class Settings(BaseSettings):
    # Use model_config instead of the deprecated inner Config class
    model_config = ConfigDict(env_file=".env", extra='ignore')

    # --- Vector Store Configuration ---
    vector_store_path: str = "./vector_store"

    # --- LLM Configuration ---
    default_llm: str = "llama2"

    # --- Data Configuration ---
    data_folder: str = "data"

    # --- API Configuration ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

# Create a single, validated settings instance
settings = Settings()
