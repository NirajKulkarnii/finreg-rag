"""
Unit tests for ingestion.vector_store
"""

from unittest.mock import MagicMock, call, patch

import pytest

from ingestion.vector_store import UPSERT_BATCH, VectorStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mock_collection(count: int = 0):
    col = MagicMock()
    col.count.return_value = count
    col.query.return_value = {
        "ids": [["chunk-0", "chunk-1"]],
        "documents": [["text zero", "text one"]],
        "metadatas": [[{"source": "fca"}, {"source": "fca"}]],
        "distances": [[0.1, 0.25]],
    }
    return col


def make_mock_chroma_client(collection):
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client


def make_fake_chunk(i: int):
    chunk = MagicMock()
    chunk.chunk_id = f"doc::chunk{i}"
    chunk.text = f"text {i}"
    chunk.metadata = {"source": "fca", "chunk_index": i}
    return chunk


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestVectorStore:
    def _make_store(self, count=0):
        col = make_mock_collection(count)
        client = make_mock_chroma_client(col)
        with patch("ingestion.vector_store.chromadb") as mock_chroma:
            mock_chroma.PersistentClient.return_value = client
            store = VectorStore(persistent_path="/tmp/test_chroma")
        store._collection = col
        store._client = client
        return store, col, client

    def test_upsert_calls_collection_with_correct_args(self):
        store, col, _ = self._make_store()
        chunks = [make_fake_chunk(i) for i in range(3)]
        embeddings = [[0.1] * 1024 for _ in chunks]
        store.upsert(chunks, embeddings)
        col.upsert.assert_called_once()
        call_kwargs = col.upsert.call_args[1]
        assert call_kwargs["ids"] == [c.chunk_id for c in chunks]
        assert call_kwargs["documents"] == [c.text for c in chunks]

    def test_upsert_returns_count(self):
        store, col, _ = self._make_store()
        chunks = [make_fake_chunk(i) for i in range(5)]
        embeddings = [[0.0] * 1024 for _ in chunks]
        result = store.upsert(chunks, embeddings)
        assert result == 5

    def test_upsert_batches_large_inputs(self):
        store, col, _ = self._make_store()
        n = UPSERT_BATCH + 10
        chunks = [make_fake_chunk(i) for i in range(n)]
        embeddings = [[0.0] * 1024 for _ in chunks]
        store.upsert(chunks, embeddings)
        assert col.upsert.call_count == 2  # two batches

    def test_upsert_mismatched_lengths_raises(self):
        store, _, _ = self._make_store()
        chunks = [make_fake_chunk(0)]
        embeddings = []  # wrong length
        with pytest.raises(AssertionError):
            store.upsert(chunks, embeddings)

    def test_query_returns_list_of_dicts(self):
        store, col, _ = self._make_store()
        results = store.query([0.0] * 1024, top_k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        assert "chunk_id" in results[0]
        assert "text" in results[0]
        assert "score" in results[0]

    def test_query_converts_distance_to_similarity(self):
        """score = 1 - distance, so distance=0.1 → score=0.9"""
        store, col, _ = self._make_store()
        results = store.query([0.0] * 1024, top_k=2)
        assert abs(results[0]["score"] - 0.9) < 1e-6
        assert abs(results[1]["score"] - 0.75) < 1e-6

    def test_query_passes_where_filter(self):
        store, col, _ = self._make_store()
        where = {"source": {"$eq": "fca"}}
        store.query([0.0] * 1024, top_k=5, where=where)
        call_kwargs = col.query.call_args[1]
        assert call_kwargs["where"] == where

    def test_query_no_where_filter_by_default(self):
        store, col, _ = self._make_store()
        store.query([0.0] * 1024, top_k=5)
        call_kwargs = col.query.call_args[1]
        assert "where" not in call_kwargs

    def test_count_delegates_to_collection(self):
        store, col, _ = self._make_store(count=42)
        assert store.count() == 42

    def test_delete_collection_resets(self):
        store, col, client = self._make_store()
        store.delete_collection()
        client.delete_collection.assert_called_once()
