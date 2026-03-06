"""
Unit tests for retrieval.reranker (CrossEncoderReranker)
"""

from unittest.mock import MagicMock, patch

import pytest

from retrieval.models import RetrievalResult
from retrieval.reranker import CrossEncoderReranker


def make_candidate(chunk_id: str = "doc::chunk0", text: str = "text",
                   score: float = 0.5) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
        "dense_score": score,
        "bm25_score": 0.0,
        "metadata": {
            "doc_id": "doc-001",
            "source": "fca",
            "title": "Test Doc",
            "url": "https://example.com",
            "jurisdiction": "UK",
            "category": "test",
            "date_published": "2024-01-01",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    }


def _make_reranker(scores: list[float]) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker.device = "cpu"
    mock_model = MagicMock()
    mock_model.predict = MagicMock(return_value=scores)
    reranker._model = mock_model
    return reranker


class TestCrossEncoderReranker:
    def test_returns_retrieval_result_objects(self):
        reranker = _make_reranker([0.8])
        results = reranker.rerank("query", [make_candidate()], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)

    def test_sorted_by_score_descending(self):
        candidates = [
            make_candidate("doc::chunk0", "text A", 0.3),
            make_candidate("doc::chunk1", "text B", 0.7),
            make_candidate("doc::chunk2", "text C", 0.1),
        ]
        reranker = _make_reranker([0.3, 0.9, 0.1])
        results = reranker.rerank("query", candidates, top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_truncation(self):
        candidates = [make_candidate(f"doc::chunk{i}") for i in range(5)]
        reranker = _make_reranker([0.9, 0.8, 0.7, 0.6, 0.5])
        results = reranker.rerank("query", candidates, top_k=3)
        assert len(results) == 3

    def test_empty_candidates_returns_empty(self):
        reranker = _make_reranker([])
        results = reranker.rerank("query", [], top_k=5)
        assert results == []

    def test_metadata_mapped_to_result_fields(self):
        candidate = make_candidate()
        reranker = _make_reranker([0.75])
        result = reranker.rerank("query", [candidate], top_k=1)[0]
        assert result.source == "fca"
        assert result.jurisdiction == "UK"
        assert result.title == "Test Doc"
        assert result.chunk_index == 0

    def test_cross_encoder_score_stored(self):
        candidate = make_candidate()
        reranker = _make_reranker([0.92])
        result = reranker.rerank("query", [candidate], top_k=1)[0]
        assert abs(result.score - 0.92) < 1e-6

    def test_load_model_raises_on_missing_dep(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        reranker.device = "cpu"
        reranker._model = None
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                reranker._load_model()
