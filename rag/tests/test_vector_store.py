"""
Unit tests for vector store
Author: Ravi
"""
from app.vector_store import VectorStore


def test_inmemory_add_and_search():
    vs = VectorStore(dim=3, backend="memory")
    # two orthogonal vectors
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    metas = [{"source": "a", "page_number": 1}, {"source": "b", "page_number": 2}]
    ids = ["a::p1", "b::p2"]
    vs.add(vectors, metas, ids)

    # search near first vector
    q = [1.0, 0.0, 0.0]
    res = vs.search(q, k=2)
    assert len(res) >= 1
    assert res[0].id == "a::p1"
    assert res[0].score >= 0.9


def test_get_ids_and_clear():
    vs = VectorStore(dim=2, backend="memory")
    vs.add([[1, 0]], [{"m": 1}], ["id1"])
    assert "id1" in vs.get_ids()
    vs.clear()
    assert vs.get_ids() == []

