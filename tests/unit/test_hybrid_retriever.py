"""
Unit tests for retrieval.retriever (HybridRetriever)
"""

from unittest.mock import MagicMock, patch

import pytest

from retrieval.models import RetrievalResult
from retrieval.retriever import HybridRetriever
from tests.conftest import make_retrieval_result


def _make_hybrid(dense_results=None, bm25_results=None, reranked=None):
    """Build a HybridRetriever with fully mocked sub-components."""
    dummy_store = MagicMock()
    dummy_emb = MagicMock()

    with patch("retrieval.retriever.DenseRetriever") as MockDense, \
         patch("retrieval.retriever.BM25Retriever") as MockBM25, \
         patch("retrieval.retriever.CrossEncoderReranker") as MockReranker:

        MockDense.return_value.retrieve.return_value = dense_results or []
        MockBM25.return_value.retrieve.return_value = bm25_results or []
        MockReranker.return_value.rerank.return_value = reranked or []

        retriever = HybridRetriever(
            vector_store=dummy_store,
            embedder=dummy_emb,
        )
        retriever._dense = MockDense.return_value
        retriever._bm25 = MockBM25.return_value
        retriever._reranker = MockReranker.return_value

    return retriever


def _candidate(chunk_id: str, source: str = "fca",
               jurisdiction: str = "UK") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"text for {chunk_id}",
        "score": 0.5,
        "metadata": {
            "source": source,
            "jurisdiction": jurisdiction,
        },
    }


class TestHybridRetrieverMerge:
    """Tests for the _merge static method directly."""

    def test_deduplicates_by_chunk_id(self):
        dense = [_candidate("a"), _candidate("b")]
        bm25 = [_candidate("b"), _candidate("c")]
        merged = HybridRetriever._merge(dense, bm25, None, None)
        ids = [m["chunk_id"] for m in merged]
        assert len(ids) == len(set(ids))

    def test_dense_and_bm25_scores_stored(self):
        dense = [_candidate("a")]
        bm25 = [_candidate("a")]
        merged = HybridRetriever._merge(dense, bm25, None, None)
        item = next(m for m in merged if m["chunk_id"] == "a")
        assert "dense_score" in item
        assert "bm25_score" in item

    def test_bm25_only_item_has_dense_score_zero(self):
        merged = HybridRetriever._merge(
            [], [_candidate("bm25-only")], None, None
        )
        assert merged[0]["dense_score"] == 0.0

    def test_dense_only_item_has_bm25_score_zero(self):
        merged = HybridRetriever._merge(
            [_candidate("dense-only")], [], None, None
        )
        assert merged[0]["bm25_score"] == 0.0

    def test_source_filter_applied_to_bm25_only(self):
        bm25 = [
            _candidate("keep", source="fca"),
            _candidate("drop", source="eurlex"),
        ]
        merged = HybridRetriever._merge([], bm25, source_filter="fca",
                                        jurisdiction_filter=None)
        ids = [m["chunk_id"] for m in merged]
        assert "keep" in ids
        assert "drop" not in ids

    def test_jurisdiction_filter_applied_to_bm25_only(self):
        bm25 = [
            _candidate("keep", jurisdiction="UK"),
            _candidate("drop", jurisdiction="EU"),
        ]
        merged = HybridRetriever._merge([], bm25, source_filter=None,
                                        jurisdiction_filter="UK")
        ids = [m["chunk_id"] for m in merged]
        assert "keep" in ids
        assert "drop" not in ids

    def test_empty_inputs_returns_empty(self):
        assert HybridRetriever._merge([], [], None, None) == []


class TestBuildWhere:
    def test_no_filters_returns_none(self):
        assert HybridRetriever._build_where(None, None) is None

    def test_source_only(self):
        w = HybridRetriever._build_where("fca", None)
        assert w == {"source": {"$eq": "fca"}}

    def test_jurisdiction_only(self):
        w = HybridRetriever._build_where(None, "UK")
        assert w == {"jurisdiction": {"$eq": "UK"}}

    def test_both_filters_uses_and(self):
        w = HybridRetriever._build_where("fca", "UK")
        assert "$and" in w
        assert len(w["$and"]) == 2


class TestHybridRetrieverRetrieve:
    def test_returns_retrieval_results(self):
        reranked = [make_retrieval_result()]
        retriever = _make_hybrid(
            dense_results=[_candidate("a")],
            bm25_results=[_candidate("b")],
            reranked=reranked,
        )
        results = retriever.retrieve("test query")
        assert results == reranked

    def test_empty_candidates_returns_empty(self):
        retriever = _make_hybrid(dense_results=[], bm25_results=[])
        results = retriever.retrieve("test")
        assert results == []

    def test_reranker_called_with_merged_candidates(self):
        dense = [_candidate("a")]
        bm25 = [_candidate("b")]
        retriever = _make_hybrid(
            dense_results=dense, bm25_results=bm25
        )
        retriever.retrieve("query")
        retriever._reranker.rerank.assert_called_once()
        _, candidates, _ = retriever._reranker.rerank.call_args[0]
        ids = {c["chunk_id"] for c in candidates}
        assert "a" in ids
        assert "b" in ids
