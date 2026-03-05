"""
Retrieval result models.
"""

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """
    Canonical output of the retrieval pipeline.

    Carries the chunk text, all provenance metadata, and both the
    original retrieval scores and the final cross-encoder score.
    """
    chunk_id: str
    text: str
    score: float          # final cross-encoder score (higher = more relevant)
    dense_score: float    # cosine similarity from ChromaDB (0–1)
    bm25_score: float     # normalised BM25 score (0–1), 0.0 if not in BM25 results

    # Document provenance (from RegulatoryDocument metadata)
    doc_id: str
    source: str           # "fca" | "eurlex" | "financebench"
    title: str
    url: str
    jurisdiction: str
    category: str
    date_published: str
    chunk_index: int
    total_chunks: int
