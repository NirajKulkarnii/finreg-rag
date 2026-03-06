"""
Unit tests for generation.web_search
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from generation.web_search import (
    build_web_search_query,
    should_use_web_search,
    web_search,
)


# ── should_use_web_search ─────────────────────────────────────────────────────

class TestShouldUseWebSearch:
    def test_no_results_triggers_search(self):
        assert should_use_web_search(0.0, "regulatory_query") is True

    def test_low_score_triggers_search(self):
        assert should_use_web_search(0.1, "general_finreg") is True

    def test_above_threshold_no_search(self):
        assert should_use_web_search(0.9, "regulatory_query") is False

    def test_at_threshold_no_search(self):
        assert should_use_web_search(0.25, "regulatory_query") is False

    def test_out_of_scope_never_triggers(self):
        # Never search for out-of-scope — just decline politely
        assert should_use_web_search(0.0, "out_of_scope") is False

    def test_custom_threshold(self):
        assert should_use_web_search(0.4, "general_finreg",
                                     score_threshold=0.5) is True
        assert should_use_web_search(0.6, "general_finreg",
                                     score_threshold=0.5) is False


# ── build_web_search_query ────────────────────────────────────────────────────

class TestBuildWebSearchQuery:
    def test_regulatory_query_prefix(self):
        q = build_web_search_query("Consumer Duty", "regulatory_query")
        assert "financial regulation" in q.lower()
        assert "Consumer Duty" in q

    def test_financial_data_prefix(self):
        q = build_web_search_query("CET1 ratio", "financial_data")
        assert "capital" in q.lower()
        assert "CET1 ratio" in q

    def test_general_prefix(self):
        q = build_web_search_query("Basel III", "general_finreg")
        assert "financial regulation" in q.lower()

    def test_unknown_intent_falls_back(self):
        q = build_web_search_query("anything", "unknown_intent")
        assert "anything" in q


# ── web_search ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestWebSearch:
    async def test_returns_formatted_string_and_raw(self):
        fake_results = [
            {
                "title": "FCA Consumer Duty",
                "href": "https://www.fca.org.uk",
                "body": "The Consumer Duty sets higher standards.",
            }
        ]
        mock_ddgs = AsyncMock()
        mock_ddgs.__aenter__ = AsyncMock(return_value=mock_ddgs)
        mock_ddgs.__aexit__ = AsyncMock(return_value=None)
        mock_ddgs.atext = AsyncMock(return_value=fake_results)

        with patch("generation.web_search.AsyncDDGS", return_value=mock_ddgs):
            formatted, raw = await web_search("Consumer Duty FCA")

        assert "FCA Consumer Duty" in formatted
        assert "https://www.fca.org.uk" in formatted
        assert len(raw) == 1

    async def test_timeout_returns_empty(self):
        import asyncio

        mock_ddgs = AsyncMock()
        mock_ddgs.__aenter__ = AsyncMock(return_value=mock_ddgs)
        mock_ddgs.__aexit__ = AsyncMock(return_value=None)
        mock_ddgs.atext = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("generation.web_search.AsyncDDGS", return_value=mock_ddgs):
            formatted, raw = await web_search("query", timeout=0.001)

        assert formatted == ""
        assert raw == []

    async def test_exception_returns_empty(self):
        mock_ddgs = AsyncMock()
        mock_ddgs.__aenter__ = AsyncMock(return_value=mock_ddgs)
        mock_ddgs.__aexit__ = AsyncMock(return_value=None)
        mock_ddgs.atext = AsyncMock(side_effect=RuntimeError("network error"))

        with patch("generation.web_search.AsyncDDGS", return_value=mock_ddgs):
            formatted, raw = await web_search("query")

        assert formatted == ""
        assert raw == []

    async def test_no_results_returns_empty(self):
        mock_ddgs = AsyncMock()
        mock_ddgs.__aenter__ = AsyncMock(return_value=mock_ddgs)
        mock_ddgs.__aexit__ = AsyncMock(return_value=None)
        mock_ddgs.atext = AsyncMock(return_value=[])

        with patch("generation.web_search.AsyncDDGS", return_value=mock_ddgs):
            formatted, raw = await web_search("query")

        assert formatted == ""
        assert raw == []

    async def test_missing_duckduckgo_returns_empty(self, caplog):
        import logging
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            with caplog.at_level(logging.ERROR):
                # Force reimport with patched module
                import importlib
                import generation.web_search as ws
                ws_mod = importlib.import_module("generation.web_search")
                # Simulate ImportError path
                formatted, raw = await ws_mod.web_search("query")
        assert formatted == ""
        assert raw == []
