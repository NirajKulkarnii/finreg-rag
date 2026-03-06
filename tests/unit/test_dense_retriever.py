"""
Unit tests for retrieval.dense_retriever
"""

from tests.conftest import FakeEmbedder, FakeVectorStore, make_chunk

from retrieval.dense_retriever import DenseRetriever


def _make_retriever(texts=None):
    chunks = [
        make_chunk(doc_id=f"doc-{i}", chunk_index=0, text=t)
        for i, t in enumerate(texts or ["default text"])
    ]
    store = FakeVectorStore(chunks=chunks)
    return DenseRetriever(FakeEmbedder(), store), store


class TestDenseRetriever:
    def test_returns_list_of_dicts(self):
        retriever, _ = _make_retriever(["Consumer Duty obligations."])
        results = retriever.retrieve("Consumer Duty")
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_result_has_required_keys(self):
        retriever, _ = _make_retriever(["text"])
        results = retriever.retrieve("query")
        for r in results:
            assert {"chunk_id", "text", "score", "metadata"} <= r.keys()

    def test_embed_called_once_per_query(self):
        from unittest.mock import MagicMock
        embedder = FakeEmbedder()
        embedder.embed = MagicMock(return_value=[[0.0] * 1024])
        store = FakeVectorStore(chunks=[make_chunk()])
        retriever = DenseRetriever(embedder, store)
        retriever.retrieve("test query")
        embedder.embed.assert_called_once_with(["test query"])

    def test_where_filter_forwarded_to_store(self):
        from unittest.mock import MagicMock
        store = FakeVectorStore(chunks=[make_chunk()])
        store.query = MagicMock(return_value=[])
        retriever = DenseRetriever(FakeEmbedder(), store)
        where = {"source": {"$eq": "fca"}}
        retriever.retrieve("query", where=where)
        store.query.assert_called_once()
        _, kwargs = store.query.call_args
        assert kwargs.get("where") == where

    def test_empty_store_returns_empty(self):
        store = FakeVectorStore(chunks=[])
        store.query = lambda *a, **kw: []
        retriever = DenseRetriever(FakeEmbedder(), store)
        results = retriever.retrieve("anything")
        assert results == []

    def test_top_k_passed_to_store(self):
        from unittest.mock import MagicMock
        store = FakeVectorStore(chunks=[make_chunk()])
        store.query = MagicMock(return_value=[])
        retriever = DenseRetriever(FakeEmbedder(), store)
        retriever.retrieve("query", top_k=7)
        _, kwargs = store.query.call_args
        assert kwargs.get("top_k") == 7
