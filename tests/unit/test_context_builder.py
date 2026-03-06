"""
Unit tests for generation.context_builder
"""

import pytest

from generation.context_builder import (
    _truncate_to_words,
    build_context,
    retrieval_confidence,
)
from generation.models import Intent
from tests.conftest import make_retrieval_result


def _results(n: int, score_start: float = 0.9, step: float = 0.05,
             same_doc: bool = False) -> list:
    return [
        make_retrieval_result(
            chunk_id=f"doc-{i if not same_doc else 0}::chunk{i}",
            doc_id=f"doc-{i if not same_doc else 0}",
            score=score_start - i * step,
        )
        for i in range(n)
    ]


# ── _truncate_to_words ────────────────────────────────────────────────────────

class TestTruncateToWords:
    def test_short_text_unchanged(self):
        text = "Hello world"
        assert _truncate_to_words(text, 100) == text

    def test_long_text_truncated(self):
        text = " ".join(f"word{i}" for i in range(50))
        result = _truncate_to_words(text, 10)
        assert len(result.split()) <= 11  # 10 words + ellipsis token
        assert result.endswith("…")

    def test_exact_length_not_truncated(self):
        words = ["w"] * 10
        text = " ".join(words)
        result = _truncate_to_words(text, 10)
        assert result == text
        assert not result.endswith("…")


# ── build_context ─────────────────────────────────────────────────────────────

class TestBuildContext:
    def test_out_of_scope_returns_empty(self):
        ctx, sources = build_context(_results(3), Intent.OUT_OF_SCOPE)
        assert ctx == ""
        assert sources == []

    def test_below_threshold_chunks_excluded(self):
        low_score = [make_retrieval_result(score=0.10)]
        ctx, sources = build_context(low_score, Intent.REGULATORY_QUERY)
        assert ctx == ""
        assert sources == []

    def test_above_threshold_included(self):
        ctx, sources = build_context(_results(1), Intent.REGULATORY_QUERY)
        assert ctx != ""
        assert len(sources) == 1

    def test_max_chunks_respected_regulatory(self):
        # Budget: 5 chunks for regulatory
        ctx, sources = build_context(
            _results(10), Intent.REGULATORY_QUERY
        )
        assert ctx.count("[Source") <= 5

    def test_max_chunks_respected_general(self):
        # Budget: 3 chunks for general
        ctx, sources = build_context(_results(10), Intent.GENERAL_FINREG)
        assert ctx.count("[Source") <= 3

    def test_source_numbering_starts_at_1(self):
        ctx, _ = build_context(_results(2), Intent.REGULATORY_QUERY)
        assert "[Source 1]" in ctx
        assert "[Source 2]" in ctx

    def test_citation_deduplication_by_doc_id(self):
        """Multiple chunks from the same doc → only one citation."""
        results = _results(3, same_doc=True)
        _, sources = build_context(results, Intent.REGULATORY_QUERY)
        assert len(sources) == 1

    def test_sources_ordered_by_relevance(self):
        results = _results(3)
        _, sources = build_context(results, Intent.REGULATORY_QUERY)
        scores = [s.relevance_score for s in sources]
        assert scores == sorted(scores, reverse=True)

    def test_text_truncated_per_budget(self):
        """Each chunk's text must not exceed the per-intent word budget."""
        long_text = " ".join(f"w{i}" for i in range(1000))
        results = [
            make_retrieval_result(
                chunk_id=f"doc-{i}::chunk0",
                doc_id=f"doc-{i}",
                text=long_text,
                score=0.9,
            )
            for i in range(3)
        ]
        # GENERAL_FINREG budget = 200 words per chunk
        ctx, _ = build_context(results, Intent.GENERAL_FINREG)
        for block in ctx.split("[Source"):
            if not block.strip():
                continue
            word_count = len(block.split())
            assert word_count < 250  # budget + header overhead

    def test_source_fields_populated(self):
        results = _results(1)
        _, sources = build_context(results, Intent.REGULATORY_QUERY)
        s = sources[0]
        assert s.title
        assert s.url
        assert s.source
        assert s.jurisdiction
        assert 0.0 <= s.relevance_score <= 1.0


# ── retrieval_confidence ──────────────────────────────────────────────────────

class TestRetrievalConfidence:
    def test_empty_results_returns_zero(self):
        assert retrieval_confidence([]) == 0.0

    def test_returns_top_score(self):
        results = _results(3, score_start=0.9, step=0.1)
        assert retrieval_confidence(results) == pytest.approx(0.9)
