"""
RAG Pipeline Integration Tests
--------------------------------
Tests the full generate() path with mocked LLM and retriever.

Coverage:
  1. Diverse query scenarios (regulatory, financial data, general, out-of-scope)
  2. Intent routing correctness
  3. Context injection / retrieval behaviour
  4. Source attribution integrity
  5. Follow-up question quality
  6. Confidence scoring logic
  7. Web search fallback trigger
  8. Edge cases (empty query, very long query, special characters)
  9. Prompt version switching
  10. Filter propagation
"""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from generation.generator import RAGGenerator
from generation.models import (
    Confidence,
    GenerationRequest,
    GenerationResponse,
    Intent,
)
from tests.conftest import (
    FakeRetriever,
    make_llm_response,
    make_retrieval_result,
)


# ── Scenario helpers ──────────────────────────────────────────────────────────

def _make_generator(
    results=None,
    intent: Intent = Intent.REGULATORY_QUERY,
    answer: str = "Test answer.",
    confidence: str = "high",
    follow_ups: list[str] | None = None,
):
    retriever = FakeRetriever(results=results or [make_retrieval_result()])
    llm_resp = make_llm_response(
        answer=answer,
        confidence=confidence,
        follow_up_questions=follow_ups or [
            "What are the obligations?",
            "When did this come into force?",
            "Who does this apply to?",
        ],
    )
    with patch("generation.generator.AsyncOpenAI"):
        gen = RAGGenerator(retriever=retriever)
    gen._client = AsyncMock()
    gen._client.chat.completions.create = AsyncMock(return_value=llm_resp)
    gen._intent_classifier = AsyncMock()
    gen._intent_classifier.classify = AsyncMock(return_value=intent)
    return gen


# ── Scenario 1: Regulatory queries ───────────────────────────────────────────

@pytest.mark.asyncio
class TestRegulatoryQueryScenarios:
    _REGULATORY_QUERIES = [
        "What are the FCA Consumer Duty obligations for retail firms?",
        "Explain the MiFID II best execution requirements.",
        "What does PS23/3 say about vulnerable customers?",
        "What are the AML/KYC obligations under the Money Laundering Regulations?",
        "How does the FCA supervise SMCR compliance?",
        "What are the PRA's capital buffer requirements for systemic banks?",
        "Explain the EMIR reporting obligations for OTC derivatives.",
        "What is the FCA's approach to product governance under COBS?",
    ]

    @pytest.mark.parametrize("query", _REGULATORY_QUERIES)
    async def test_regulatory_intent_assigned(self, query):
        gen = _make_generator(intent=Intent.REGULATORY_QUERY)
        response = await gen.generate(GenerationRequest(query=query))
        assert response.intent == Intent.REGULATORY_QUERY

    async def test_regulatory_response_has_sources(self):
        results = [make_retrieval_result(score=0.9)]
        gen = _make_generator(results=results, intent=Intent.REGULATORY_QUERY)
        response = await gen.generate(
            GenerationRequest(query="What is Consumer Duty?")
        )
        assert len(response.sources) >= 1

    async def test_regulatory_response_high_confidence_with_context(self):
        results = [make_retrieval_result(score=0.9)]
        gen = _make_generator(
            results=results,
            intent=Intent.REGULATORY_QUERY,
            confidence="high",
        )
        response = await gen.generate(
            GenerationRequest(query="What is Consumer Duty?")
        )
        assert response.confidence == Confidence.HIGH

    async def test_regulatory_source_has_url(self):
        gen = _make_generator(intent=Intent.REGULATORY_QUERY)
        response = await gen.generate(
            GenerationRequest(query="Consumer Duty obligations?")
        )
        for src in response.sources:
            assert src.url.startswith("http")

    async def test_regulatory_source_has_jurisdiction(self):
        gen = _make_generator(intent=Intent.REGULATORY_QUERY)
        response = await gen.generate(
            GenerationRequest(query="FCA Consumer Duty?")
        )
        for src in response.sources:
            assert src.jurisdiction  # non-empty


# ── Scenario 2: Financial data queries ───────────────────────────────────────

@pytest.mark.asyncio
class TestFinancialDataScenarios:
    _FINANCIAL_QUERIES = [
        "What is the minimum CET1 ratio under CRR?",
        "How is the Liquidity Coverage Ratio (LCR) calculated?",
        "What are the leverage ratio requirements under Basel III?",
        "What is the Net Stable Funding Ratio (NSFR)?",
        "How does IFRS 9 affect loan loss provisioning?",
    ]

    @pytest.mark.parametrize("query", _FINANCIAL_QUERIES)
    async def test_financial_intent_assigned(self, query):
        gen = _make_generator(intent=Intent.FINANCIAL_DATA)
        response = await gen.generate(GenerationRequest(query=query))
        assert response.intent == Intent.FINANCIAL_DATA

    async def test_financial_response_has_follow_ups(self):
        gen = _make_generator(
            intent=Intent.FINANCIAL_DATA,
            follow_ups=["What is Tier 2 capital?",
                        "How is RWA calculated?",
                        "What triggers a SREP?"],
        )
        response = await gen.generate(
            GenerationRequest(query="What is CET1?")
        )
        assert len(response.follow_up_questions) >= 1


# ── Scenario 3: General finreg queries ───────────────────────────────────────

@pytest.mark.asyncio
class TestGeneralFinregScenarios:
    _GENERAL_QUERIES = [
        "What is financial regulation?",
        "How does prudential regulation differ from conduct regulation?",
        "What is the role of central banks in financial stability?",
        "Explain the twin peaks regulatory model.",
        "What is macroprudential policy?",
    ]

    @pytest.mark.parametrize("query", _GENERAL_QUERIES)
    async def test_general_intent_assigned(self, query):
        gen = _make_generator(intent=Intent.GENERAL_FINREG)
        response = await gen.generate(GenerationRequest(query=query))
        assert response.intent == Intent.GENERAL_FINREG

    async def test_general_response_is_complete(self):
        gen = _make_generator(
            intent=Intent.GENERAL_FINREG,
            answer="Financial regulation encompasses prudential and conduct rules.",
        )
        response = await gen.generate(
            GenerationRequest(query="What is financial regulation?")
        )
        assert response.answer
        assert response.confidence
        assert isinstance(response.follow_up_questions, list)


# ── Scenario 4: Out-of-scope queries ─────────────────────────────────────────

@pytest.mark.asyncio
class TestOutOfScopeScenarios:
    _OOS_QUERIES = [
        "What's the best recipe for chocolate cake?",
        "Who won the football match last night?",
        "Recommend a sci-fi movie.",
        "Tell me about quantum computing.",
        "What is the capital of France?",
    ]

    @pytest.mark.parametrize("query", _OOS_QUERIES)
    async def test_out_of_scope_intent(self, query):
        gen = _make_generator(intent=Intent.OUT_OF_SCOPE, results=[])
        response = await gen.generate(GenerationRequest(query=query))
        assert response.intent == Intent.OUT_OF_SCOPE

    async def test_out_of_scope_no_web_search(self):
        gen = _make_generator(intent=Intent.OUT_OF_SCOPE, results=[])
        with patch(
            "generation.generator.web_search",
            new=AsyncMock(return_value=("", []))
        ) as mock_ws:
            response = await gen.generate(
                GenerationRequest(query="Tell me a recipe.")
            )
        assert response.used_web_search is False

    async def test_out_of_scope_no_sources(self):
        gen = _make_generator(intent=Intent.OUT_OF_SCOPE, results=[])
        response = await gen.generate(
            GenerationRequest(query="Tell me a joke.")
        )
        # No retrieval context → no sources (out-of-scope budget = 0)
        assert response.sources == []


# ── Scenario 5: Web search fallback ──────────────────────────────────────────

@pytest.mark.asyncio
class TestWebSearchFallback:
    async def test_web_search_triggered_when_no_retrieval(self):
        gen = _make_generator(
            results=[make_retrieval_result(score=0.05)],
            intent=Intent.REGULATORY_QUERY,
        )
        with patch(
            "generation.generator.web_search",
            new=AsyncMock(
                return_value=("Web result about Consumer Duty.", [{"r": 1}])
            )
        ):
            response = await gen.generate(
                GenerationRequest(query="Very specific obscure question?")
            )
        assert response.used_web_search is True
        assert response.web_search_query is not None

    async def test_web_search_not_triggered_above_threshold(self):
        gen = _make_generator(
            results=[make_retrieval_result(score=0.95)],
            intent=Intent.REGULATORY_QUERY,
        )
        with patch(
            "generation.generator.web_search",
            new=AsyncMock(return_value=("", []))
        ) as mock_ws:
            await gen.generate(GenerationRequest(query="Consumer Duty?"))
        mock_ws.assert_not_called()

    async def test_web_search_query_contains_domain_context(self):
        gen = _make_generator(
            results=[make_retrieval_result(score=0.0)],
            intent=Intent.REGULATORY_QUERY,
        )
        captured_query = {}

        async def capture_search(query, **kwargs):
            captured_query["q"] = query
            return ("result", [{"r": 1}])

        with patch("generation.generator.web_search", new=capture_search):
            await gen.generate(
                GenerationRequest(query="Consumer Duty obligations")
            )
        assert "Consumer Duty" in captured_query.get("q", "")


# ── Scenario 6: Filter propagation ───────────────────────────────────────────

@pytest.mark.asyncio
class TestFilterPropagation:
    async def test_source_filter_propagated(self):
        tracking = {}

        class TrackingRetriever:
            def retrieve(self, query, **kwargs):
                tracking.update(kwargs)
                return [make_retrieval_result(score=0.9)]

        with patch("generation.generator.AsyncOpenAI"):
            gen = RAGGenerator(retriever=TrackingRetriever())
        gen._client = AsyncMock()
        gen._client.chat.completions.create = AsyncMock(
            return_value=make_llm_response()
        )
        gen._intent_classifier = AsyncMock()
        gen._intent_classifier.classify = AsyncMock(
            return_value=Intent.REGULATORY_QUERY
        )
        await gen.generate(
            GenerationRequest(
                query="Q?", source_filter="fca", jurisdiction_filter="UK"
            )
        )
        assert tracking["source_filter"] == "fca"
        assert tracking["jurisdiction_filter"] == "UK"


# ── Scenario 7: Prompt version switching ─────────────────────────────────────

@pytest.mark.asyncio
class TestPromptVersionSwitching:
    async def test_v1_and_v2_produce_different_system_prompts(self):
        captured = {}

        gen = _make_generator(intent=Intent.REGULATORY_QUERY)

        async def capturing_create(*args, **kwargs):
            captured["messages"] = kwargs["messages"]
            return make_llm_response()

        gen._client.chat.completions.create = capturing_create

        from config import settings as cfg
        import generation.prompts as prompts

        # Capture v1
        cfg.settings.prompt_version = "v1"
        await gen.generate(GenerationRequest(query="Consumer Duty?"))
        sys_v1 = captured["messages"][0]["content"]

        # Capture v2
        cfg.settings.prompt_version = "v2"
        await gen.generate(GenerationRequest(query="Consumer Duty?"))
        sys_v2 = captured["messages"][0]["content"]

        # Reset
        cfg.settings.prompt_version = "v1"

        assert sys_v1 != sys_v2, (
            "v1 and v2 system prompts should differ"
        )


# ── Scenario 8: Edge cases ────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestEdgeCases:
    async def test_very_long_query(self):
        long_query = "What is the FCA Consumer Duty? " * 100
        gen = _make_generator(intent=Intent.REGULATORY_QUERY)
        response = await gen.generate(GenerationRequest(query=long_query))
        assert isinstance(response, GenerationResponse)

    async def test_query_with_special_characters(self):
        gen = _make_generator(intent=Intent.GENERAL_FINREG)
        response = await gen.generate(
            GenerationRequest(query="What does §3(2)(a) mean in MiFID II?")
        )
        assert isinstance(response, GenerationResponse)

    async def test_unicode_query(self):
        gen = _make_generator(intent=Intent.GENERAL_FINREG)
        response = await gen.generate(
            GenerationRequest(query="Qu'est-ce que la réglementation financière?")
        )
        assert isinstance(response, GenerationResponse)

    async def test_response_always_has_intent(self):
        for intent in list(Intent):
            gen = _make_generator(intent=intent, results=[])
            response = await gen.generate(GenerationRequest(query="test?"))
            assert response.intent == intent

    async def test_response_confidence_is_valid_enum(self):
        gen = _make_generator(confidence="medium")
        response = await gen.generate(GenerationRequest(query="test?"))
        assert response.confidence in list(Confidence)

    async def test_broken_llm_json_still_returns_response(self):
        gen = _make_generator()
        gen._intent_classifier.classify = AsyncMock(
            return_value=Intent.GENERAL_FINREG
        )
        broken = MagicMock()
        broken.choices[0].message.content = "NOT VALID JSON AT ALL"
        gen._client.chat.completions.create = AsyncMock(return_value=broken)

        response = await gen.generate(GenerationRequest(query="test?"))
        assert isinstance(response, GenerationResponse)
        # Answer falls back to raw text
        assert "NOT VALID JSON AT ALL" in response.answer
        assert response.confidence == Confidence.LOW


# ── Scenario 9: Response schema completeness ──────────────────────────────────

@pytest.mark.asyncio
class TestResponseSchemaCompleteness:
    async def test_all_response_fields_present(self):
        gen = _make_generator()
        gen._intent_classifier.classify = AsyncMock(
            return_value=Intent.REGULATORY_QUERY
        )
        response = await gen.generate(GenerationRequest(query="Q?"))
        assert hasattr(response, "intent")
        assert hasattr(response, "answer")
        assert hasattr(response, "sources")
        assert hasattr(response, "follow_up_questions")
        assert hasattr(response, "confidence")
        assert hasattr(response, "used_web_search")
        assert hasattr(response, "retrieval_used")

    async def test_source_fields_complete(self):
        results = [make_retrieval_result(score=0.9)]
        gen = _make_generator(results=results)
        gen._intent_classifier.classify = AsyncMock(
            return_value=Intent.REGULATORY_QUERY
        )
        response = await gen.generate(GenerationRequest(query="Q?"))
        for src in response.sources:
            assert hasattr(src, "title")
            assert hasattr(src, "url")
            assert hasattr(src, "source")
            assert hasattr(src, "jurisdiction")
            assert hasattr(src, "relevance_score")
            assert 0.0 <= src.relevance_score <= 1.0
