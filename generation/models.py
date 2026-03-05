"""
Generation output models.
--------------------------
All generation responses conform to GenerationResponse — a structured
schema that carries the answer, citations, follow-up questions, and
metadata needed to surface a complete UI card.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Intent(str, Enum):
    """Classified intent of the user query."""
    REGULATORY_QUERY    = "regulatory_query"      # specific regulation / compliance question
    FINANCIAL_DATA      = "financial_data"         # financial metrics, ratios, numbers
    GENERAL_FINREG      = "general_finreg"         # broad finreg topic, educational
    OUT_OF_SCOPE        = "out_of_scope"           # unrelated to financial regulation


class Confidence(str, Enum):
    HIGH   = "high"    # strong grounding in retrieved / web context
    MEDIUM = "medium"  # partial grounding, some inference
    LOW    = "low"     # mostly inference, no direct evidence found


class Source(BaseModel):
    """A single citation linked to the answer."""
    title: str
    url: str
    source: str          # "fca" | "eurlex" | "financebench"
    jurisdiction: str
    date_published: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class GenerationResponse(BaseModel):
    """
    Complete structured response returned by the generation pipeline.

    Fields
    ------
    intent              : Classified intent of the user query.
    answer              : The main answer text (markdown-safe).
    sources             : Ordered list of cited documents (highest relevance first).
    follow_up_questions : 2–3 suggested follow-up questions.
    confidence          : Self-assessed answer confidence.
    used_web_search     : True when the answer relied on live web results.
    web_search_query    : The query sent to web search (if used).
    retrieval_used      : True when retrieved context was injected.
    """
    intent: Intent
    answer: str
    sources: list[Source] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM
    used_web_search: bool = False
    web_search_query: Optional[str] = None
    retrieval_used: bool = False


class GenerationRequest(BaseModel):
    """Input to the generation pipeline."""
    query: str
    source_filter: Optional[str] = None          # "fca" | "eurlex" | "financebench"
    jurisdiction_filter: Optional[str] = None    # "UK" | "EU" | ...
    stream: bool = False                         # reserved for streaming endpoint
