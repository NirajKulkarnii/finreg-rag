"""
Shared data models for regiq document loaders.
All loaders normalise their output to the RegulatoryDocument schema.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class DataSource(str, Enum):
    FCA = "fca"
    EURLEX = "eurlex"
    FINANCEBENCH = "financebench"


@dataclass
class RegulatoryDocument:
    """
    Canonical representation of a regulatory document across all sources.

    Every loader must produce instances of this class so the ingestion
    pipeline can treat all sources uniformly.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    doc_id: str                          # Unique ID within the source (e.g. celex_id, FCA ref)
    source: DataSource                   # Which dataset this came from
    title: str                           # Human-readable document title
    text: str                            # Full document text (pre-cleaning)

    # ── Provenance ────────────────────────────────────────────────────────────
    url: Optional[str] = None            # Source URL for citation
    jurisdiction: Optional[str] = None  # e.g. "UK", "EU", "US"
    language: str = "en"                 # ISO 639-1 language code
    category: Optional[str] = None      # e.g. "banking", "consumer_credit", "AML"
    date_published: Optional[str] = None

    # ── For evaluation (FinanceBench only) ────────────────────────────────────
    eval_question: Optional[str] = None
    eval_answer: Optional[str] = None

    # ── Computed at ingestion time ────────────────────────────────────────────
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        """Returns flat metadata dict for ChromaDB storage alongside embeddings."""
        return {
            "doc_id": self.doc_id,
            "source": self.source.value,
            "title": self.title,
            "url": self.url or "",
            "jurisdiction": self.jurisdiction or "",
            "language": self.language,
            "category": self.category or "",
            "date_published": self.date_published or "",
            **self.metadata,
        }