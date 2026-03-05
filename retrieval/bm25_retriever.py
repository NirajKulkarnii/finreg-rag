"""
BM25 Retriever
--------------
Builds an in-memory BM25 index over all chunks stored in ChromaDB and
provides lexical keyword retrieval as a complement to dense vector search.

The index is built once at construction time by pulling all chunk texts
from ChromaDB. This is fast enough for the dataset sizes used here
(tens of thousands of chunks) and avoids maintaining a separate index file.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "that", "this", "these", "those",
    "it", "its", "as", "not", "no", "nor", "so", "yet", "both", "either",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace, strip stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w and w not in _STOPWORDS]


class BM25Retriever:
    """
    Lexical retriever using BM25Okapi over all ChromaDB chunks.

    The full chunk corpus is loaded from ChromaDB at init time and tokenised
    once. Retrieval is O(n_chunks) per query but is fast enough in practice.

    Usage:
        retriever = BM25Retriever(vector_store)
        candidates = retriever.retrieve("money laundering AML directive", top_k=20)
    """

    def __init__(self, vector_store):
        """
        Args:
            vector_store: VectorStore instance — used to pull all stored chunks.
        """
        self.vector_store = vector_store
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._chunk_metadatas: list[dict] = []
        self._bm25 = None
        self._build_index()

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Return top_k chunks by BM25 score.

        Args:
            query:  Plain-text query string.
            top_k:  Number of candidates to return.

        Returns:
            List of dicts: {chunk_id, text, score, metadata}
            score is normalised BM25 in [0, 1].
        """
        if self._bm25 is None or not self._chunk_ids:
            logger.warning("BM25Retriever: index is empty, returning no results.")
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        raw_scores = self._bm25.get_scores(tokens)

        # Get indices sorted by score descending
        top_indices = sorted(
            range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True
        )[:top_k]

        # Normalise scores to [0, 1] relative to the maximum in this query
        max_score = raw_scores[top_indices[0]] if top_indices else 1.0
        if max_score == 0:
            return []

        results = []
        for idx in top_indices:
            score = raw_scores[idx]
            if score <= 0:
                break
            results.append({
                "chunk_id": self._chunk_ids[idx],
                "text": self._chunk_texts[idx],
                "score": float(score / max_score),
                "metadata": self._chunk_metadatas[idx],
            })

        logger.debug(f"BM25Retriever: {len(results)} candidates for query '{query[:60]}'")
        return results

    def index_size(self) -> int:
        return len(self._chunk_ids)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_index(self):
        """Fetch all chunks from ChromaDB and build BM25Okapi index."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 not installed. Run: pip install rank-bm25")

        logger.info("BM25Retriever: loading all chunks from ChromaDB...")
        collection = self.vector_store._collection

        total = collection.count()
        if total == 0:
            logger.warning("BM25Retriever: ChromaDB collection is empty — index not built.")
            return

        # ChromaDB get() returns all stored chunks (no embedding needed)
        result = collection.get(include=["documents", "metadatas"])
        self._chunk_ids = result["ids"]
        self._chunk_texts = result["documents"]
        self._chunk_metadatas = result["metadatas"]

        tokenized = [_tokenize(t) for t in self._chunk_texts]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25Retriever: index built over {len(self._chunk_ids)} chunks.")
