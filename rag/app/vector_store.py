"""
Vector store wrapper
Author: Ravi

Provides a thin FAISS-backed vector store with metadata management.
Supports two backends: "faiss" (default) and "memory" (Python list).
The store persists vectors (faiss index) and metadata (pickle) under VECTOR_DIR.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import pickle
from .config import VECTOR_DIR, VECTORS_PATH, TOP_K

VECTOR_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: Dict


class FaissStore:
    def __init__(self, dim: int, index_path: Path = VECTORS_PATH):
        self.dim = dim
        self.index_path = index_path
        # Index is created lazily
        self.index = None
        self.metadatas: List[Dict] = []
        self.ids: List[str] = []
        self._load()

    def _load(self):
        try:
            import faiss
        except Exception:
            # faiss not installed; start empty
            self.index = None
            self.ids = []
            self.metadatas = []
            return

        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                meta_path = str(self.index_path) + ".meta"
                with open(meta_path, "rb") as f:
                    obj = pickle.load(f)
                    self.ids = obj.get("ids", [])
                    self.metadatas = obj.get("metadatas", [])
            except Exception:
                # Create fresh
                self.index = None
                self.ids = []
                self.metadatas = []
        else:
            self.index = None
            self.ids = []
            self.metadatas = []

    def _ensure_index(self):
        try:
            import faiss
        except Exception:
            raise RuntimeError("faiss is required for FaissStore")
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)

    def add(self, vectors: List[List[float]], metadatas: List[Dict], ids: List[str]):
        import numpy as np
        arr = np.array(vectors, dtype="float32")
        # normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        self._ensure_index()
        self.index.add(arr)
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)
        self._persist()

    def _persist(self):
        if self.index is None:
            return
        import faiss
        faiss.write_index(self.index, str(self.index_path))
        meta_path = str(self.index_path) + ".meta"
        with open(meta_path, "wb") as f:
            pickle.dump({"ids": self.ids, "metadatas": self.metadatas}, f)

    def search(self, vector: List[float], k: int = TOP_K) -> List[SearchResult]:
        import numpy as np
        if self.index is None or getattr(self.index, "ntotal", 0) == 0:
            return []
        arr = np.array([vector], dtype="float32")
        # normalize
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        arr = arr / norm
        D, I = self.index.search(arr, k)
        results: List[SearchResult] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            results.append(SearchResult(id=self.ids[idx], score=float(score), metadata=self.metadatas[idx]))
        return results


class InMemoryStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: List[List[float]] = []
        self.metadatas: List[Dict] = []
        self.ids: List[str] = []

    def add(self, vectors: List[List[float]], metadatas: List[Dict], ids: List[str]):
        self.vectors.extend(vectors)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def search(self, vector: List[float], k: int = TOP_K) -> List[SearchResult]:
        import numpy as np
        if not self.vectors:
            return []
        arr = np.array(self.vectors, dtype="float32")
        q = np.array(vector, dtype="float32")
        # cosine sim
        a_norm = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        scores = a_norm.dot(q_norm)
        idxs = np.argsort(-scores)[:k]
        return [SearchResult(id=self.ids[i], score=float(scores[i]), metadata=self.metadatas[i]) for i in idxs]


class VectorStore:
    """Facade to choose backend. dim must be provided for initialization."""

    def __init__(self, dim: int, backend: str = "faiss"):
        self.dim = dim
        self.backend = backend
        if backend == "faiss":
            self.store = FaissStore(dim)
        else:
            self.store = InMemoryStore(dim)

    def add(self, vectors: List[List[float]], metadatas: List[Dict], ids: List[str]):
        self.store.add(vectors, metadatas, ids)

    def search(self, vector: List[float], k: int = TOP_K) -> List[SearchResult]:
        return self.store.search(vector, k)

    def is_empty(self) -> bool:
        # crude emptiness check
        if isinstance(self.store, FaissStore):
            return self.store.index is None or getattr(self.store.index, "ntotal", 0) == 0
        return len(self.store.ids) == 0

    def get_ids(self) -> List[str]:
        """Return the list of ids stored in the vector store."""
        return list(self.store.ids)

    def clear(self):
        """Clear the store in-memory and persist empty state (useful for tests)."""
        if isinstance(self.store, FaissStore):
            # recreate index
            self.store.index = None
            self.store.ids = []
            self.store.metadatas = []
            # remove files if exist
            try:
                if self.store.index_path.exists():
                    self.store.index_path.unlink()
                meta_path = str(self.store.index_path) + ".meta"
                p = Path(meta_path)
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        else:
            self.store.ids = []
            self.store.metadatas = []
            self.store.vectors = []
