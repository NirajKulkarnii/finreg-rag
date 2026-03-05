"""
Intent Classifier
-----------------
Classifies user queries into one of four intent types using a fast,
low-token LLM call to vLLM (Qwen2.5-7B).

Latency optimisation:
  - max_tokens=20  — only the JSON label is needed
  - temperature=0  — deterministic, no sampling overhead
  - JSON mode      — structured output, no post-processing regex needed

Fallback:
  - Keyword heuristics are applied first (< 1 ms, no network call).
    The LLM is only invoked when heuristics are inconclusive.
"""

from __future__ import annotations

import json
import logging
import re

from openai import AsyncOpenAI

from config.settings import settings
from .models import Intent
from .prompts import get_prompt, active_version

logger = logging.getLogger(__name__)

# ── Keyword heuristics ────────────────────────────────────────────────────────
# Applied before the LLM call to save latency on obvious cases.

_REGULATORY_KEYWORDS = re.compile(
    r"\b(regulation|directive|rule|article|obligation|compliance|FCA|PRA|MiFID|CRR|"
    r"GDPR|AML|KYC|EMIR|SFDR|AIFMD|UCITS|Basel|Solvency|consumer duty|PS\d{2}|"
    r"CP\d{2}|TR\d{2}|policy statement|guidance|handbook|supervisory)\b",
    re.IGNORECASE,
)

_FINANCIAL_DATA_KEYWORDS = re.compile(
    r"\b(ratio|capital|liquidity|LCR|NSFR|CET1|leverage|RWA|P&L|revenue|earnings|"
    r"EPS|ROE|ROA|net income|balance sheet|tier 1|tier 2|IFRS|GAAP|provision|"
    r"impairment|NPL|coverage ratio|solvency ratio)\b",
    re.IGNORECASE,
)

_OUT_OF_SCOPE_KEYWORDS = re.compile(
    r"\b(recipe|weather|sport|movie|music|travel|covid|health|medical|fashion|"
    r"celebrity|politics|game|gaming)\b",
    re.IGNORECASE,
)


def _heuristic_classify(query: str) -> Intent | None:
    """
    Fast keyword-based pre-classification. Returns None when inconclusive
    so the LLM call is used instead.
    """
    if _OUT_OF_SCOPE_KEYWORDS.search(query):
        return Intent.OUT_OF_SCOPE
    reg = bool(_REGULATORY_KEYWORDS.search(query))
    fin = bool(_FINANCIAL_DATA_KEYWORDS.search(query))
    if reg and not fin:
        return Intent.REGULATORY_QUERY
    if fin and not reg:
        return Intent.FINANCIAL_DATA
    # Both or neither — defer to LLM
    return None


# ── LLM-based classifier ──────────────────────────────────────────────────────

class IntentClassifier:
    """
    Classifies user queries via a fast, low-token LLM call.

    Parameters
    ----------
    client : AsyncOpenAI
        Pre-initialised async client pointed at vLLM.
    model : str
        vLLM model name (default from settings).
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = settings.vllm_model,
    ) -> None:
        self._client = client
        self._model = model

    async def classify(self, query: str) -> Intent:
        """
        Return the Intent for the given query.

        Tries heuristics first; falls back to LLM on ambiguous cases.
        """
        # ── Fast path: heuristics ─────────────────────────────────────────
        heuristic = _heuristic_classify(query)
        if heuristic is not None:
            logger.debug(f"Intent (heuristic): {heuristic} for '{query[:60]}'")
            return heuristic

        # ── Slow path: LLM classification ────────────────────────────────
        try:
            pv = active_version()
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": get_prompt("intent_system", pv)},
                    {"role": "user", "content": get_prompt("intent_user", pv).format(query=query)},
                ],
                max_tokens=20,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            parsed = json.loads(raw)
            intent_str = parsed.get("intent", "general_finreg")
            intent = Intent(intent_str)
            logger.debug(f"Intent (LLM): {intent} for '{query[:60]}'")
            return intent

        except Exception as exc:
            logger.warning(f"Intent classification failed ({exc}); defaulting to general_finreg")
            return Intent.GENERAL_FINREG
