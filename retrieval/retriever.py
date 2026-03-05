"""
Hybrid Retriever
----------------
Orchestrates dense vector retrieval, BM25 lexical retrieval, and
cross-encoder reranking into a single, unified retrieval pipeline.

Pipeline:
  1. Dense retrieval  — BGE-M3 query embedding → ChromaDB ANN → top-K candidates
  2. BM25 retrieval   — tokenised keyword search over all stored chunks → top-K candidates
  3. Merge & dedup    — union of both result sets, keyed by chunk_id
  4. Cross-encoder    — (query, chunk) scoring → sorted, truncated to rerank_top_k

Usage:
    from ingestion import Embedder, VectorStore
    from retrieval import HybridRetriever

    store    = VectorStore(persistent_path="chroma_db")
    embedder = Embedder(device="cuda")
    retriever = HybridRetriever(store, embedder)

    results = retriever.retrieve("Consumer Duty obligations for retail firms")
    for r in results:
        print(r.score, r.source, r.title)
        print(r.text[:300])
"""

import logging
from typing import Optional

from config.settings import settings
from .dense_retriever import DenseRetriever
from .bm25_retriever import BM25Retriever
from .reranker import CrossEncoderReranker
from .models import RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Two-stage hybrid retriever with cross-encoder reranking.

    Args:
        vector_store:      VectorStore connected to ChromaDB.
        embedder:          Embedder instance (BAAI/bge-m3).
        reranker_model:    HuggingFace model ID for the cross-encoder.
        device:            Torch device for embedder and reranker ("cpu" / "cuda").
        retrieval_top_k:   Candidates fetched from each retriever before reranking.
        rerank_top_k:      Final results returned after reranking.
    """

    def __init__(
        self,
        vector_store,
        embedder,
        reranker_model: str = settings.reranker_model,
        device: str = settings.embedding_device,
        retrieval_top_k: int = settings.retrieval_top_k,
        rerank_top_k: int = settings.rerank_top_k,
    ):
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

        self._dense = DenseRetriever(embedder, vector_store)
        self._bm25 = BM25Retriever(vector_store)
        self._reranker = CrossEncoderReranker(model_name=reranker_model, device=device)

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        jurisdiction_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query:               Plain-text question or search string.
            top_k:               Number of final results (overrides rerank_top_k).
            source_filter:       Restrict to one source: "fca", "eurlex", or "financebench".
            jurisdiction_filter: Restrict to jurisdiction: "UK", "EU", etc.

        Returns:
            List of RetrievalResult sorted by cross-encoder score descending.
        """
        final_k = top_k if top_k is not None else self.rerank_top_k

        # Build ChromaDB where filter for dense retrieval
        where = self._build_where(source_filter, jurisdiction_filter)

        # ── Stage 1: Dense retrieval ───────────────────────────────────────
        dense_results = self._dense.retrieve(query, top_k=self.retrieval_top_k, where=where)

        # ── Stage 2: BM25 retrieval ────────────────────────────────────────
        bm25_results = self._bm25.retrieve(query, top_k=self.retrieval_top_k)

        # ── Stage 3: Merge & deduplicate ──────────────────────────────────
        candidates = self._merge(
            dense_results,
            bm25_results,
            source_filter=source_filter,
            jurisdiction_filter=jurisdiction_filter,
        )

        if not candidates:
            logger.warning(f"HybridRetriever: no candidates found for '{query[:60]}'")
            return []

        logger.info(
            f"HybridRetriever: {len(dense_results)} dense + {len(bm25_results)} BM25 "
            f"→ {len(candidates)} merged candidates → reranking to top {final_k}."
        )

        # ── Stage 4: Cross-encoder reranking ──────────────────────────────
        return self._reranker.rerank(query, candidates, top_k=final_k)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_where(
        source_filter: Optional[str],
        jurisdiction_filter: Optional[str],
    ) -> Optional[dict]:
        """Build ChromaDB $and/$eq filter from optional source and jurisdiction."""
        conditions = []
        if source_filter:
            conditions.append({"source": {"$eq": source_filter}})
        if jurisdiction_filter:
            conditions.append({"jurisdiction": {"$eq": jurisdiction_filter}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _merge(
        dense: list[dict],
        bm25: list[dict],
        source_filter: Optional[str],
        jurisdiction_filter: Optional[str],
    ) -> list[dict]:
        """
        Merge dense and BM25 results into a deduplicated candidate list.

        For chunks appearing in both lists, keep the best score from each and
        store both as dense_score / bm25_score. BM25-only results are
        post-filtered by source/jurisdiction (dense results already filtered
        by the ChromaDB where clause).
        """
        merged: dict[str, dict] = {}

        for item in dense:
            cid = item["chunk_id"]
            merged[cid] = {
                **item,
                "dense_score": item["score"],
                "bm25_score": 0.0,
            }

        for item in bm25:
            cid = item["chunk_id"]
            meta = item.get("metadata", {})

            # Apply source/jurisdiction filter to BM25-only results
            if source_filter and meta.get("source") != source_filter:
                continue
            if jurisdiction_filter and meta.get("jurisdiction") != jurisdiction_filter:
                continue

            if cid in merged:
                merged[cid]["bm25_score"] = item["score"]
            else:
                merged[cid] = {
                    **item,
                    "dense_score": 0.0,
                    "bm25_score": item["score"],
                }

        return list(merged.values())
