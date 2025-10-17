"""
Embeddings helper using sentence-transformers
Author: Ravi

Provides a thin async-compatible wrapper to compute embeddings for a list of texts.
"""
from typing import List
import numpy as np

# Model name
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model():
    """Lazily import and return the SentenceTransformer model."""
    global _model
    if _model is None:
        # Import here to avoid requiring the package on initial import of this module
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts as plain python lists."""
    model = _get_model()
    arr = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # Ensure 2D
    if arr.ndim == 1:
        arr = np.expand_dims(arr, 0)
    return [list(map(float, row)) for row in arr]


def get_embedding_dim() -> int:
    """Return the model embedding dimension."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()
