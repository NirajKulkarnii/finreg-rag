"""
Cross-Encoder Reranker
----------------------
Reranks a list of retrieved candidates using a cross-encoder model
(query, passage) → relevance score, providing more accurate ranking
than the initial retrieval scores.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Fast, lightweight, strong on passage retrieval tasks
  - Can be swapped for a larger model via settings.reranker_model
"""

import logging
from typing import Optional

from .models import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieval candidates with a cross-encoder scoring model.

    The cross-encoder jointly encodes (query, passage) pairs and produces
    a relevance score, which is more accurate than cosine similarity alone
    but requires a forward pass per (query, candidate) pair.

    Usage:
        reranker = CrossEncoderReranker()
        results = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Score and rerank candidates, returning the top_k most relevant.

        Args:
            query:      Plain-text query string.
            candidates: List of dicts from DenseRetriever / BM25Retriever merge.
                        Each dict must have keys: chunk_id, text, score, metadata,
                        and optionally dense_score, bm25_score.
            top_k:      Number of results to return.

        Returns:
            List of RetrievalResult sorted by cross-encoder score descending.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]

        logger.debug(
            f"CrossEncoderReranker: scoring {len(pairs)} candidates "
            f"with {self.model_name}..."
        )
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        # Pair scores with candidates and sort
        scored = sorted(
            zip(raw_scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        results = []
        for ce_score, cand in scored:
            meta = cand.get("metadata", {})
            results.append(RetrievalResult(
                chunk_id=cand["chunk_id"],
                text=cand["text"],
                score=float(ce_score),
                dense_score=float(cand.get("dense_score", 0.0)),
                bm25_score=float(cand.get("bm25_score", 0.0)),
                doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""),
                title=meta.get("title", ""),
                url=meta.get("url", ""),
                jurisdiction=meta.get("jurisdiction", ""),
                category=meta.get("category", ""),
                date_published=meta.get("date_published", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                total_chunks=int(meta.get("total_chunks", 0)),
            ))

        logger.debug(f"CrossEncoderReranker: returning {len(results)} results.")
        return results

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        logger.info(f"Loading reranker {self.model_name} on {self.device}...")
        model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Reranker loaded.")
        return model
