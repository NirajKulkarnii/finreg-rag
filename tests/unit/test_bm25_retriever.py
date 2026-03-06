"""
Unit tests for retrieval.bm25_retriever
"""

import pytest

from retrieval.bm25_retriever import BM25Retriever, _tokenize
from tests.conftest import FakeVectorStore, make_chunk


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_lowercases(self):
        assert "fca" in _tokenize("FCA")

    def test_removes_punctuation(self):
        tokens = _tokenize("anti-money laundering!")
        assert "!" not in " ".join(tokens)
        assert "-" not in " ".join(tokens)

    def test_removes_stopwords(self):
        tokens = _tokenize("the regulation of financial markets")
        assert "the" not in tokens
        assert "of" not in tokens

    def test_keeps_meaningful_terms(self):
        tokens = _tokenize("capital requirements directive AML")
        assert "capital" in tokens
        assert "requirements" in tokens
        assert "aml" in tokens

    def test_empty_string_returns_empty(self):
        assert _tokenize("") == []

    def test_only_stopwords_returns_empty(self):
        assert _tokenize("the and or but") == []


# ── BM25Retriever ─────────────────────────────────────────────────────────────

def _make_retriever_with_docs(texts: list[str]) -> BM25Retriever:
    chunks = [
        make_chunk(
            doc_id=f"doc-{i:03d}",
            chunk_index=0,
            text=text,
        )
        for i, text in enumerate(texts)
    ]
    store = FakeVectorStore(chunks=chunks)
    return BM25Retriever(store)


class TestBM25Retriever:
    CORPUS = [
        "Consumer Duty requires firms to deliver good outcomes for retail customers.",
        "Anti-money laundering AML obligations apply to all regulated firms.",
        "Capital Requirements Regulation CRR sets minimum capital ratios.",
        "MiFID II governs investment services and investor protection in the EU.",
        "The Prudential Regulation Authority supervises banks and insurers.",
    ]

    def test_index_size_matches_corpus(self):
        retriever = _make_retriever_with_docs(self.CORPUS)
        assert retriever.index_size() == len(self.CORPUS)

    def test_relevant_doc_ranks_first(self):
        retriever = _make_retriever_with_docs(self.CORPUS)
        results = retriever.retrieve("AML anti-money laundering requirements")
        assert len(results) > 0
        # The AML doc should be top-ranked
        assert "laundering" in results[0]["text"].lower()

    def test_returns_at_most_top_k(self):
        retriever = _make_retriever_with_docs(self.CORPUS)
        results = retriever.retrieve("regulation", top_k=2)
        assert len(results) <= 2

    def test_scores_normalised_to_0_1(self):
        retriever = _make_retriever_with_docs(self.CORPUS)
        results = retriever.retrieve("capital requirements", top_k=5)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_top_result_has_score_1(self):
        retriever = _make_retriever_with_docs(self.CORPUS)
        results = retriever.retrieve("consumer duty outcomes")
        if results:
            assert abs(results[0]["score"] - 1.0) < 1e-6

    def test_empty_query_returns_empty(self):
        retriever = _make_retriever_with_docs(self.CORPUS)
        results = retriever.retrieve("")
        assert results == []

    def test_stopwords_only_query_returns_empty(self):
        retriever = _make_retriever_with_docs(self.CORPUS)
        results = retriever.retrieve("the and or but")
        assert results == []

    def test_result_dict_has_required_keys(self):
        retriever = _make_retriever_with_docs(self.CORPUS)
        results = retriever.retrieve("capital")
        assert all(
            {"chunk_id", "text", "score", "metadata"} <= r.keys()
            for r in results
        )

    def test_empty_store_returns_empty(self):
        store = FakeVectorStore(chunks=[])
        retriever = BM25Retriever(store)
        results = retriever.retrieve("anything")
        assert results == []

    def test_scores_descending(self):
        retriever = _make_retriever_with_docs(self.CORPUS)
        results = retriever.retrieve("regulation capital requirements")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
