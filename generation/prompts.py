"""
Versioned Prompt Registry
--------------------------
All prompts live here, organised by version string (e.g. "v1", "v2").
To experiment with a new prompt set:
  1. Copy an existing version block and change its key (e.g. "v2").
  2. Edit any fields you want to change.
  3. Set PROMPT_VERSION=v2 in your .env (or override at runtime).
  4. Compare results — the old version is untouched.

Callers use the accessor functions at the bottom of this file:

    from generation.prompts import get_prompt, get_system_prompt, active_version

Every version must define ALL keys listed in _REQUIRED_KEYS.
A KeyError at startup (via validate_version) catches missing keys early.

Prompt keys
-----------
  System prompts (one per intent):
    "sys_regulatory"   — regulatory / compliance questions
    "sys_financial"    — financial data / metrics
    "sys_general"      — broad finreg topic
    "sys_out_of_scope" — query outside domain

  User prompt templates (format-string, use .format(**kwargs)):
    "user_with_context"  — kwargs: context_blocks, query
    "user_with_web"      — kwargs: web_query, web_results, query
    "user_no_context"    — kwargs: query

  Intent classifier prompts:
    "intent_system"     — system message for the classifier LLM
    "intent_user"       — user message template; kwargs: query

  Shared building blocks (composed into system prompts by the version):
    "context_block"     — single retrieval chunk template;
                          kwargs: n, title, source, jurisdiction, date, url, text

_SYSTEM_INTENT_MAP maps Intent enum values → system-prompt key so
generator.py stays generic across versions.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Keys every version must supply ───────────────────────────────────────────

_REQUIRED_KEYS = {
    "sys_regulatory",
    "sys_financial",
    "sys_general",
    "sys_out_of_scope",
    "user_with_context",
    "user_with_web",
    "user_no_context",
    "intent_system",
    "intent_user",
    "context_block",
}

# Maps Intent.value → system-prompt key (version-independent)
_SYSTEM_INTENT_MAP = {
    "regulatory_query": "sys_regulatory",
    "financial_data":   "sys_financial",
    "general_finreg":   "sys_general",
    "out_of_scope":     "sys_out_of_scope",
}

# ══════════════════════════════════════════════════════════════════════════════
# VERSION v1  — initial production prompts
# ══════════════════════════════════════════════════════════════════════════════

_JSON_SCHEMA_v1 = """
Respond ONLY with a valid JSON object — no markdown fences, no prose outside JSON.

Schema:
{
  "answer": "<string — clear, well-structured markdown answer>",
  "confidence": "<'high' | 'medium' | 'low'>",
  "follow_up_questions": ["<string>", "<string>", "<string>"]
}

Rules:
- answer: Write in structured markdown (headings, bullet points as needed).
  Cite inline using [Source N] where N matches the context block number.
  If context is unavailable or insufficient, state that clearly and use general knowledge.
- confidence: 'high' if directly supported by context/sources;
  'medium' if partially supported or inferred; 'low' if speculative.
- follow_up_questions: Exactly 3 concise questions the user might ask next.
"""

_v1: dict[str, str] = {
    # ── System prompts ────────────────────────────────────────────────────────
    "sys_regulatory": (
        "You are RegIQ, an expert AI assistant specialising in financial regulation. "
        "Your audience is compliance officers, legal counsel, and regulatory analysts. "
        "Answer with precision: cite the specific rule, article, or guidance section. "
        "Distinguish between binding requirements and supervisory expectations. "
        "Flag jurisdiction-specific differences where relevant. "
        "Do NOT provide legal advice — recommend consulting qualified counsel for firm-specific decisions."
        + _JSON_SCHEMA_v1
    ),
    "sys_financial": (
        "You are RegIQ, an expert AI assistant specialising in financial data and metrics. "
        "Interpret the question in the context of regulatory capital, liquidity, or financial reporting. "
        "Be precise with numbers and formulas. Distinguish between regulatory and accounting figures. "
        "State clearly when data is approximate, outdated, or jurisdiction-specific."
        + _JSON_SCHEMA_v1
    ),
    "sys_general": (
        "You are RegIQ, an accessible AI guide to financial regulation. "
        "Explain concepts clearly for a professional audience that may not be deeply specialised. "
        "Use plain language, illustrate with examples where helpful. "
        "Ground the answer in provided context; if context is absent, use general knowledge and "
        "note that the user should verify against official sources."
        + _JSON_SCHEMA_v1
    ),
    "sys_out_of_scope": (
        "You are RegIQ, an AI assistant focused on financial regulation. "
        "The query appears to be outside your domain. "
        "Politely acknowledge this, briefly explain your scope, and suggest how the user might "
        "rephrase their question if it has a financial-regulation angle."
        + _JSON_SCHEMA_v1
    ),

    # ── User prompt templates ─────────────────────────────────────────────────
    "user_with_context": (
        "Retrieved context (use these as primary references):\n"
        "{context_blocks}\n"
        "---\n"
        "Question: {query}"
    ),
    "user_with_web": (
        "Web search results for '{web_query}':\n"
        "{web_results}\n"
        "---\n"
        "Question: {query}"
    ),
    "user_no_context": "Question: {query}",

    # ── Intent classifier ─────────────────────────────────────────────────────
    "intent_system": (
        "You are an intent classifier for a financial-regulation assistant. "
        "Classify the user query into exactly one of these intents:\n"
        "  regulatory_query  — asks about a specific rule, regulation, law, obligation, or compliance requirement\n"
        "  financial_data    — asks about financial numbers, ratios, metrics, capital requirements, or reporting figures\n"
        "  general_finreg    — broad financial-regulation question or educational topic\n"
        "  out_of_scope      — unrelated to financial regulation\n\n"
        'Respond with ONLY a JSON object: {"intent": "<label>"}'
    ),
    "intent_user": "Query: {query}",

    # ── Context block template ────────────────────────────────────────────────
    "context_block": (
        "[Source {n}] {title} ({source}, {jurisdiction}, {date})\n"
        "URL: {url}\n"
        "{text}\n"
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# VERSION v2  — terser system prompts, stricter JSON schema, chain-of-thought
#               reasoning step before the final answer
# ══════════════════════════════════════════════════════════════════════════════

_JSON_SCHEMA_v2 = """
You MUST respond with a single valid JSON object. No text outside the JSON.

{
  "reasoning": "<1–3 sentences of internal reasoning before answering>",
  "answer": "<markdown answer — use ## headings and bullet points>",
  "confidence": "<'high' | 'medium' | 'low'>",
  "follow_up_questions": ["<string>", "<string>", "<string>"]
}

Cite retrieved sources inline as [N]. Confidence is 'high' only when every
key claim is directly supported by a [Source N] block.
"""

_v2: dict[str, str] = {
    # ── System prompts ────────────────────────────────────────────────────────
    "sys_regulatory": (
        "You are RegIQ, a precise financial-regulation AI for compliance professionals. "
        "Always ground answers in the numbered source blocks provided. "
        "State the exact article, rule, or section. "
        "Distinguish mandatory requirements from guidance. "
        "Note cross-jurisdiction differences. "
        "Never give legal advice."
        + _JSON_SCHEMA_v2
    ),
    "sys_financial": (
        "You are RegIQ, a financial-data AI for regulatory analysts. "
        "Ground answers in source blocks. "
        "Be exact with ratios, thresholds, and formulas. "
        "Distinguish regulatory from accounting definitions. "
        "Flag data that may be stale."
        + _JSON_SCHEMA_v2
    ),
    "sys_general": (
        "You are RegIQ, a financial-regulation guide for professionals. "
        "Use plain language and examples. "
        "Ground answers in source blocks when available; "
        "otherwise use general knowledge and flag that verification is needed."
        + _JSON_SCHEMA_v2
    ),
    "sys_out_of_scope": (
        "You are RegIQ, focused on financial regulation. "
        "This query is outside your domain. "
        "Politely decline, state your scope, and suggest a finreg angle if one exists."
        + _JSON_SCHEMA_v2
    ),

    # ── User prompt templates ─────────────────────────────────────────────────
    "user_with_context": (
        "### Retrieved Sources\n"
        "{context_blocks}\n"
        "---\n"
        "### Question\n"
        "{query}"
    ),
    "user_with_web": (
        "### Web Search Results (query: '{web_query}')\n"
        "{web_results}\n"
        "---\n"
        "### Question\n"
        "{query}"
    ),
    "user_no_context": "### Question\n{query}",

    # ── Intent classifier (same logic, tighter wording) ───────────────────────
    "intent_system": (
        "Classify the query into one label:\n"
        "  regulatory_query — specific rule, law, or compliance requirement\n"
        "  financial_data   — numbers, ratios, capital, reporting figures\n"
        "  general_finreg   — broad finreg topic or education\n"
        "  out_of_scope     — unrelated to financial regulation\n"
        'Output ONLY: {"intent": "<label>"}'
    ),
    "intent_user": "Query: {query}",

    # ── Context block template (adds score for transparency) ──────────────────
    "context_block": (
        "[Source {n}] {title}\n"
        "Source: {source} | Jurisdiction: {jurisdiction} | Date: {date} | Score: {score:.3f}\n"
        "URL: {url}\n"
        "{text}\n"
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# Registry — add new versions here
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_REGISTRY: dict[str, dict[str, str]] = {
    "v1": _v1,
    "v2": _v2,
}

# ── Validation ────────────────────────────────────────────────────────────────

def validate_version(version: str) -> None:
    """Raise ValueError if `version` is missing or has missing keys."""
    if version not in PROMPT_REGISTRY:
        available = ", ".join(sorted(PROMPT_REGISTRY))
        raise ValueError(
            f"Prompt version '{version}' not found. Available: {available}"
        )
    missing = _REQUIRED_KEYS - PROMPT_REGISTRY[version].keys()
    if missing:
        raise ValueError(
            f"Prompt version '{version}' is missing keys: {sorted(missing)}"
        )


def list_versions() -> list[str]:
    """Return all registered prompt version identifiers."""
    return sorted(PROMPT_REGISTRY.keys())


# ── Accessors ─────────────────────────────────────────────────────────────────

def get_prompt(key: str, version: str | None = None) -> str:
    """
    Retrieve a single prompt string from the registry.

    Parameters
    ----------
    key     : Prompt key (e.g. "sys_regulatory", "user_with_context").
    version : Version string. If None, the active version from settings is used.
    """
    from config.settings import settings          # late import — avoids circular at module load
    v = version or settings.prompt_version
    try:
        return PROMPT_REGISTRY[v][key]
    except KeyError:
        available = ", ".join(sorted(PROMPT_REGISTRY))
        raise KeyError(f"Prompt key '{key}' not found in version '{v}'. Available versions: {available}")


def get_system_prompt(intent: str, version: str | None = None) -> str:
    """
    Return the system prompt for a given intent value.

    Parameters
    ----------
    intent  : Intent.value string (e.g. "regulatory_query").
    version : Prompt version. Defaults to settings.prompt_version.
    """
    key = _SYSTEM_INTENT_MAP.get(intent, "sys_general")
    return get_prompt(key, version)


def active_version() -> str:
    """Return the currently configured prompt version."""
    from config.settings import settings
    return settings.prompt_version
