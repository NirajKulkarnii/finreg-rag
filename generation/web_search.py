"""
Web Search Tool
---------------
Provides a lightweight async fallback when retrieval context is absent
or below the confidence threshold.

Backend: DuckDuckGo (no API key required, rate-limit friendly for dev).
         Swap for SerpAPI / Bing by changing _search_backend().

Returned snippets are formatted identically to retrieval context blocks
so the generator prompt is consistent whether the answer comes from the
vector store or the web.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS  # type: ignore[no-redef]
    except ImportError:
        DDGS = None  # type: ignore[assignment,misc]

# Maximum web results to fetch and inject into prompt
_MAX_WEB_RESULTS = 5
# Maximum words per web snippet to inject
_MAX_WORDS_PER_SNIPPET = 150


def _format_web_block(n: int, title: str, url: str, body: str, max_words: int) -> str:
    words = body.split()
    if len(words) > max_words:
        body = " ".join(words[:max_words]) + " …"
    return f"[Web {n}] {title}\nURL: {url}\n{body}\n"


async def web_search(
    query: str,
    max_results: int = _MAX_WEB_RESULTS,
    region: str = "wt-wt",          # worldwide; use "uk-en" for UK-biased results
    timeout: float = 5.0,
) -> tuple[str, list[dict]]:
    """
    Perform an async DuckDuckGo text search.

    Parameters
    ----------
    query       : Search query string.
    max_results : Maximum number of snippets to return.
    region      : DuckDuckGo region code.
    timeout     : Maximum seconds to wait for results.

    Returns
    -------
    formatted_str : Ready-to-inject prompt block (empty string on failure).
    raw_results   : List of dicts with keys: title, url, body.
    """
    if DDGS is None:
        logger.error(
            "duckduckgo-search is not installed. "
            "Add it to requirements.txt: duckduckgo-search>=8.0.0"
        )
        return "", []

    def _sync_search() -> list[dict]:
        with DDGS() as ddgs:
            return ddgs.text(query, region=region, max_results=max_results)

    try:
        raw: list[dict] = await asyncio.wait_for(
            asyncio.to_thread(_sync_search),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Web search timed out for query: '{query[:80]}'")
        return "", []
    except Exception as exc:
        logger.warning(f"Web search failed: {exc}")
        return "", []

    if not raw:
        return "", []

    blocks: list[str] = []
    for n, item in enumerate(raw, start=1):
        title = item.get("title", "")
        url   = item.get("href", item.get("url", ""))
        body  = item.get("body", "")
        blocks.append(_format_web_block(n, title, url, body, _MAX_WORDS_PER_SNIPPET))

    formatted = "\n".join(blocks)
    return formatted, raw


def should_use_web_search(
    retrieval_score: float,
    intent: str,
    score_threshold: float = 0.25,
) -> bool:
    """
    Decide whether to fall back to web search.

    Triggers when:
      - No retrieval results (score == 0.0)
      - Top retrieval score is below the threshold
      - Intent is out_of_scope (retrieval is skipped entirely)

    Parameters
    ----------
    retrieval_score : Top cross-encoder score from retrieval (0.0 if none).
    intent          : Classified intent string.
    score_threshold : Minimum acceptable retrieval score.
    """
    if intent == "out_of_scope":
        return False   # out-of-scope: don't search — just decline politely
    return retrieval_score < score_threshold


def build_web_search_query(query: str, intent: str) -> str:
    """
    Augment the raw user query with domain context for better web results.
    """
    prefix_map = {
        "regulatory_query": "financial regulation compliance",
        "financial_data":   "financial regulatory capital",
        "general_finreg":   "financial regulation",
    }
    prefix = prefix_map.get(intent, "financial regulation")
    return f"{prefix} {query}"
