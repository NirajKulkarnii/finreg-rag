"""
Context Builder
---------------
Assembles a token-efficient context string from retrieval results.

Design goals:
  1. Enforce a per-intent token budget so the LLM context window is not
     flooded with low-value text (wasted tokens → higher latency + cost).
  2. Truncate individual chunks at the word level rather than cutting
     arbitrarily mid-sentence.
  3. Surface the chunks that earned the highest cross-encoder scores first
     so the LLM attends to the most relevant material early.
  4. Return both the formatted string *and* a deduplicated Source list for
     the response citations.

Token budget (approximate words ≈ tokens * 0.75 — conservative estimate):
  - REGULATORY_QUERY : 5 chunks × 300 words  = 1 500 words  ≈ 2 000 tokens
  - FINANCIAL_DATA   : 5 chunks × 400 words  = 2 000 words  ≈ 2 700 tokens
  - GENERAL_FINREG   : 3 chunks × 200 words  =   600 words  ≈   800 tokens
  - OUT_OF_SCOPE     : 0 chunks (no context injected)
"""

from __future__ import annotations

from retrieval.models import RetrievalResult
from .models import Intent, Source
from .prompts import get_prompt, active_version

# ── Per-intent budget config ──────────────────────────────────────────────────

_BUDGET: dict[str, tuple[int, int]] = {
    # intent                 max_chunks  max_words_per_chunk
    Intent.REGULATORY_QUERY: (5, 300),
    Intent.FINANCIAL_DATA:   (5, 400),
    Intent.GENERAL_FINREG:   (3, 200),
    Intent.OUT_OF_SCOPE:     (0, 0),
}

# Minimum cross-encoder score to include a chunk in context.
# Below this the chunk is likely off-topic; better not to inject noise.
_MIN_SCORE_THRESHOLD = 0.15


def _truncate_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def build_context(
    results: list[RetrievalResult],
    intent: Intent,
) -> tuple[str, list[Source]]:
    """
    Format retrieval results into a prompt-ready context string and source list.

    Parameters
    ----------
    results : list[RetrievalResult]
        Ranked retrieval results (highest cross-encoder score first).
    intent : Intent
        Classified intent — controls the token budget.

    Returns
    -------
    context_str : str
        Numbered context blocks ready to inject into the user prompt.
        Empty string when no context should be injected (OUT_OF_SCOPE or
        all results below threshold).
    sources : list[Source]
        Deduplicated citations for the response, ordered by relevance.
    """
    max_chunks, max_words = _BUDGET.get(intent, (3, 200))

    if max_chunks == 0:
        return "", []

    # Filter by score threshold, already sorted best-first by reranker
    eligible = [r for r in results if r.score >= _MIN_SCORE_THRESHOLD]

    # Cap at budget
    selected = eligible[:max_chunks]

    if not selected:
        return "", []

    blocks: list[str] = []
    sources: list[Source] = []
    seen_docs: set[str] = set()

    context_block_tpl = get_prompt("context_block", active_version())

    for n, result in enumerate(selected, start=1):
        text_excerpt = _truncate_to_words(result.text, max_words)

        block = context_block_tpl.format(
            n=n,
            title=result.title,
            source=result.source,
            jurisdiction=result.jurisdiction,
            date=result.date_published or "n/a",
            url=result.url,
            text=text_excerpt,
            score=result.score,   # used by v2+; silently ignored by v1
        )
        blocks.append(block)

        # Deduplicate citations by doc_id (multiple chunks from same doc → one citation)
        if result.doc_id not in seen_docs:
            seen_docs.add(result.doc_id)
            sources.append(
                Source(
                    title=result.title,
                    url=result.url,
                    source=result.source,
                    jurisdiction=result.jurisdiction,
                    date_published=result.date_published or "",
                    relevance_score=round(result.score, 4),
                )
            )

    context_str = "\n".join(blocks)
    return context_str, sources


def retrieval_confidence(results: list[RetrievalResult]) -> float:
    """
    Return the top cross-encoder score from retrieved results.
    Used by the generator to decide whether to trigger web search fallback.
    """
    if not results:
        return 0.0
    return results[0].score
