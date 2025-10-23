"""
conftest.py - Pytest configuration file for manual environment setup.

Author: Ravi
Date: 2023-10-27

This file programmatically sets the required environment variables before any
tests run. This is the most robust method to ensure the test environment is
configured correctly, bypassing issues with automatic .env loading.
"""

import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    A session-wide fixture to manually set environment variables for all tests.
    This runs once at the beginning of the test session.
    """
    os.environ["VECTOR_STORE_PATH"] = "./test_vector_store"
    os.environ["DEFAULT_LLM"] = "test_llm"
    os.environ["DATA_FOLDER"] = "./test_data"
    os.environ["API_HOST"] = "0.0.0.0"
    os.environ["API_PORT"] = "8001"
