"""
Unit tests for generation.intent_classifier
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from generation.intent_classifier import IntentClassifier, _heuristic_classify
from generation.models import Intent


# ── Heuristic tests ───────────────────────────────────────────────────────────

class TestHeuristicClassify:
    @pytest.mark.parametrize("query,expected", [
        ("What are the FCA requirements for Consumer Duty?",
         Intent.REGULATORY_QUERY),
        ("Explain the MiFID II compliance obligations",
         Intent.REGULATORY_QUERY),
        ("What does PS23/3 say about vulnerable customers?",
         Intent.REGULATORY_QUERY),
        ("What is the CET1 ratio for HSBC?", Intent.FINANCIAL_DATA),
        ("Explain the LCR liquidity coverage ratio", Intent.FINANCIAL_DATA),
        ("What is the net income on the balance sheet?", Intent.FINANCIAL_DATA),
        ("What is the best recipe for pasta?", Intent.OUT_OF_SCOPE),
        ("Tell me about the weather today", Intent.OUT_OF_SCOPE),
        ("Recommend a movie to watch", Intent.OUT_OF_SCOPE),
    ])
    def test_classification(self, query, expected):
        assert _heuristic_classify(query) == expected

    def test_ambiguous_returns_none(self):
        # Both regulatory and financial — defer to LLM
        result = _heuristic_classify(
            "What is the CET1 capital ratio requirement under CRR?"
        )
        assert result is None

    def test_generic_finreg_returns_none(self):
        # No clear keyword match — defer to LLM
        result = _heuristic_classify("How does financial regulation work?")
        assert result is None

    def test_case_insensitive(self):
        assert _heuristic_classify("fca handbook guidance") == Intent.REGULATORY_QUERY
        assert _heuristic_classify("FCA HANDBOOK GUIDANCE") == Intent.REGULATORY_QUERY


# ── IntentClassifier (LLM path) ───────────────────────────────────────────────

def _build_llm_response(intent: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = json.dumps({"intent": intent})
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio
class TestIntentClassifierLLM:
    async def test_heuristic_bypass_skips_llm(self):
        """When heuristics fire, the LLM must NOT be called."""
        client = AsyncMock()
        classifier = IntentClassifier(client, model="test-model")
        result = await classifier.classify(
            "What are the FCA Consumer Duty obligations?"
        )
        assert result == Intent.REGULATORY_QUERY
        client.chat.completions.create.assert_not_called()

    async def test_llm_called_for_ambiguous_query(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _build_llm_response(
            "general_finreg"
        )
        classifier = IntentClassifier(client, model="test-model")
        result = await classifier.classify(
            "How does financial regulation work?"
        )
        assert result == Intent.GENERAL_FINREG
        client.chat.completions.create.assert_called_once()

    async def test_llm_low_tokens_used(self):
        """Verify the classifier uses max_tokens=20 for speed."""
        client = AsyncMock()
        client.chat.completions.create.return_value = _build_llm_response(
            "general_finreg"
        )
        classifier = IntentClassifier(client, model="test-model")
        await classifier.classify("What is fintech?")
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 20

    async def test_llm_temperature_zero(self):
        """Temperature must be 0 for deterministic classification."""
        client = AsyncMock()
        client.chat.completions.create.return_value = _build_llm_response(
            "general_finreg"
        )
        classifier = IntentClassifier(client, model="test-model")
        await classifier.classify("What is fintech?")
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    async def test_llm_failure_defaults_to_general_finreg(self):
        client = AsyncMock()
        client.chat.completions.create.side_effect = Exception("timeout")
        classifier = IntentClassifier(client, model="test-model")
        result = await classifier.classify("What is fintech?")
        assert result == Intent.GENERAL_FINREG

    async def test_invalid_llm_label_raises_value_error_caught(self):
        """Invalid intent label from LLM should fall back to GENERAL_FINREG."""
        client = AsyncMock()
        client.chat.completions.create.return_value = _build_llm_response(
            "not_a_valid_intent"
        )
        classifier = IntentClassifier(client, model="test-model")
        result = await classifier.classify("What is fintech?")
        assert result == Intent.GENERAL_FINREG

    async def test_all_valid_intents_accepted(self):
        valid = [
            "regulatory_query", "financial_data",
            "general_finreg", "out_of_scope"
        ]
        for label in valid:
            client = AsyncMock()
            client.chat.completions.create.return_value = (
                _build_llm_response(label)
            )
            classifier = IntentClassifier(client, model="test-model")
            result = await classifier.classify("What is fintech regulation?")
            assert result.value == label
