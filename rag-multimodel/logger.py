"""
logger.py - Centralized logging configuration.

Author: Ravi
Date: 2023-10-27

This module sets up a standardized logger for the application to ensure
consistent, structured logging, which is essential for production environments.
"""

import logging
import sys

# Create a logger instance
logger = logging.getLogger("rag_app")
logger.setLevel(logging.INFO)

# Create a handler to output to stdout
handler = logging.StreamHandler(sys.stdout)

# Create a formatter for structured logging (e.g., JSON or a specific format)
# For simplicity, we'll use a detailed text format. For a real production
# system, you might use python-json-logger.
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers:
    logger.addHandler(handler)
