"""
Unit tests for generation.generator (RAGGenerator)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from generation.generator import RAGGenerator, _parse_llm_output
from generation.models import (
    Confidence,
    GenerationRequest,
    GenerationResponse,
    Intent,
)
from tests.conftest import FakeRetriever, make_llm_response, make_retrieval_result


# ── _parse_llm_output ─────────────────────────────────────────────────────────

class TestParseLlmOutput:
    def test_valid_json_parsed(self):
        raw = json.dumps({
            "answer": "Consumer Duty answer.",
            "confidence": "high",
            "follow_up_questions": ["Q1?", "Q2?", "Q3?"],
        })
        answer, confidence, follow_ups = _parse_llm_output(raw)
        assert answer == "Consumer Duty answer."
        assert confidence == Confidence.HIGH
        assert follow_ups == ["Q1?", "Q2?", "Q3?"]

    def test_follow_ups_capped_at_3(self):
        raw = json.dumps({
            "answer": "ans",
            "confidence": "medium",
            "follow_up_questions": ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
        })
        _, _, follow_ups = _parse_llm_output(raw)
        assert len(follow_ups) == 3

    def test_markdown_fences_stripped(self):
        raw = "```json\n" + json.dumps({
            "answer": "ans",
            "confidence": "low",
            "follow_up_questions": [],
        }) + "\n```"
        answer, _, _ = _parse_llm_output(raw)
        assert answer == "ans"

    def test_invalid_json_returns_raw_text(self):
        raw = "This is not JSON."
        answer, confidence, follow_ups = _parse_llm_output(raw)
        assert answer == raw
        assert confidence == Confidence.LOW
        assert follow_ups == []

    def test_missing_fields_use_defaults(self):
        raw = json.dumps({"answer": "only answer"})
        answer, confidence, follow_ups = _parse_llm_output(raw)
        assert answer == "only answer"
        assert confidence == Confidence.MEDIUM
        assert follow_ups == []


# ── RAGGenerator.generate ─────────────────────────────────────────────────────

def _make_generator(retrieval_results=None, llm_answer="Test answer."):
    """Build a RAGGenerator with mocked retriever and LLM client."""
    retriever = FakeRetriever(results=retrieval_results)
    llm_resp = make_llm_response(answer=llm_answer)

    with patch("generation.generator.AsyncOpenAI"):
        generator = RAGGenerator(retriever=retriever)

    generator._client = AsyncMock()
    generator._client.chat.completions.create = AsyncMock(
        return_value=llm_resp
    )
    generator._intent_classifier = AsyncMock()
    return generator


@pytest.mark.asyncio
class TestRAGGeneratorGenerate:
    async def test_returns_generation_response(self):
        generator = _make_generator()
        generator._intent_classifier.classify = AsyncMock(
            return_value=Intent.REGULATORY_QUERY
        )
        request = GenerationRequest(query="What is Consumer Duty?")
        response = await generator.generate(request)
        assert isinstance(response, GenerationResponse)

    async def test_intent_set_correctly(self):
        generator = _make_generator()
        generator._intent_classifier.classify = AsyncMock(
            return_value=Intent.FINANCIAL_DATA
        )
        request = GenerationRequest(query="What is the CET1 ratio?")
        response = await generator.generate(request)
        assert response.intent == Intent.FINANCIAL_DATA

    async def test_retrieval_used_when_results_above_threshold(self):
        results = [make_retrieval_result(score=0.9)]
        generator = _make_generator(retrieval_results=results)
        generator._intent_classifier.classify = AsyncMock(
            return_value=Intent.REGULATORY_QUERY
        )
        request = GenerationRequest(query="Consumer Duty obligations?")
        response = await generator.generate(request)
        assert response.retrieval_used is True

    async def test_web_search_triggered_on_low_score(self):
        results = [make_retrieval_result(score=0.05)]
        generator = _make_generator(retrieval_results=results)
        generator._intent_classifier.classify = AsyncMock(
            return_value=Intent.REGULATORY_QUERY
        )
        with patch(
            "generation.generator.web_search",
            new=AsyncMock(return_value=("web content here", [{"r": 1}]))
        ):
            request = GenerationRequest(query="Something obscure")
            response = await generator.generate(request)
        assert response.used_web_search is True

    async def test_web_search_not_triggered_on_out_of_scope(self):
        """out_of_scope intent must never trigger web search."""
        generator = _make_generator(retrieval_results=[])
        generator._intent_classifier.classify = AsyncMock(
            return_value=Intent.OUT_OF_SCOPE
        )
        with patch(
            "generation.generator.web_search",
            new=AsyncMock(return_value=("", []))
        ) as mock_ws:
            request = GenerationRequest(query="Tell me a recipe")
            response = await generator.generate(request)
        assert response.used_web_search is False

    async def test_sources_populated_from_retrieval(self):
        results = [make_retrieval_result(score=0.9)]
        generator = _make_generator(retrieval_results=results)
        generator._intent_classifier.classify = AsyncMock(
            return_value=Intent.REGULATORY_QUERY
        )
        request = GenerationRequest(query="Consumer Duty?")
        response = await generator.generate(request)
        assert len(response.sources) >= 1
        assert response.sources[0].url

    async def test_answer_from_llm(self):
        generator = _make_generator(llm_answer="The answer is 42.")
        generator._intent_classifier.classify = AsyncMock(
            return_value=Intent.GENERAL_FINREG
        )
        request = GenerationRequest(query="What is Basel?")
        response = await generator.generate(request)
        assert "42" in response.answer

    async def test_follow_up_questions_present(self):
        generator = _make_generator()
        generator._intent_classifier.classify = AsyncMock(
            return_value=Intent.REGULATORY_QUERY
        )
        request = GenerationRequest(query="Consumer Duty?")
        response = await generator.generate(request)
        assert isinstance(response.follow_up_questions, list)
        assert len(response.follow_up_questions) <= 3

    async def test_source_and_jurisdiction_filters_forwarded(self):
        """Filters on request must reach the retriever."""
        called_kwargs = {}

        class TrackingRetriever:
            def retrieve(self, query, **kwargs):
                called_kwargs.update(kwargs)
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

        request = GenerationRequest(
            query="Q?",
            source_filter="fca",
            jurisdiction_filter="UK",
        )
        await gen.generate(request)
        assert called_kwargs.get("source_filter") == "fca"
        assert called_kwargs.get("jurisdiction_filter") == "UK"

    async def test_llm_json_mode_requested(self):
        generator = _make_generator()
        generator._intent_classifier.classify = AsyncMock(
            return_value=Intent.GENERAL_FINREG
        )
        await generator.generate(GenerationRequest(query="test"))
        call_kwargs = generator._client.chat.completions.create.call_args[1]
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    async def test_parallel_intent_retrieval(self):
        """intent classification and retrieval run concurrently."""
        import asyncio

        classify_started = asyncio.Event()
        retrieve_started = asyncio.Event()

        async def slow_classify(query):
            classify_started.set()
            await asyncio.sleep(0.05)
            return Intent.GENERAL_FINREG

        original_retrieve = FakeRetriever().retrieve

        async def slow_retrieve(q, sf, jf):
            retrieve_started.set()
            await asyncio.sleep(0.05)
            return [make_retrieval_result(score=0.9)]

        generator = _make_generator()
        generator._intent_classifier.classify = slow_classify
        generator._retrieve = slow_retrieve

        import time
        t0 = time.monotonic()
        await generator.generate(GenerationRequest(query="test"))
        elapsed = time.monotonic() - t0

        # If parallel: ~0.05s; if sequential: ~0.10s
        assert elapsed < 0.12, "Intent + retrieval should run in parallel"


# ── generate_stream ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRAGGeneratorStream:
    async def test_stream_yields_text_chunks(self):
        generator = _make_generator()
        generator._intent_classifier.classify = AsyncMock(
            return_value=Intent.GENERAL_FINREG
        )

        # Build a fake async stream
        tokens = ["Consumer", " Duty", " is", " important."]

        async def fake_stream_gen():
            for token in tokens:
                chunk = MagicMock()
                chunk.choices[0].delta.content = token
                yield chunk

        generator._client.chat.completions.create = AsyncMock(return_value=fake_stream_gen())

        collected = []
        async for chunk in generator.generate_stream(
            GenerationRequest(query="What is Consumer Duty?")
        ):
            collected.append(chunk)

        full = "".join(collected)
        assert "Consumer" in full
        # Meta block must be appended
        meta_parts = [c for c in collected if c.startswith("\n__META__:")]
        assert len(meta_parts) == 1

    async def test_stream_meta_contains_intent(self):
        generator = _make_generator()
        generator._intent_classifier.classify = AsyncMock(
            return_value=Intent.REGULATORY_QUERY
        )

        async def fake_stream_gen():
            chunk = MagicMock()
            chunk.choices[0].delta.content = "answer"
            yield chunk

        generator._client.chat.completions.create = AsyncMock(return_value=fake_stream_gen())

        meta_json = None
        async for chunk in generator.generate_stream(
            GenerationRequest(query="Q?")
        ):
            if chunk.startswith("\n__META__:"):
                meta_json = json.loads(chunk[len("\n__META__:"):])

        assert meta_json is not None
        assert meta_json["intent"] == "regulatory_query"
