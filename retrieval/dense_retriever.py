"""
Dense Retriever
---------------
Embeds a query with BGE-M3 and retrieves nearest-neighbour chunks from ChromaDB.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Retrieves document chunks via approximate nearest-neighbour search in ChromaDB.

    Dependencies are injected so the same Embedder and VectorStore instances
    can be shared across the pipeline without reloading the model.

    Usage:
        retriever = DenseRetriever(embedder, vector_store)
        candidates = retriever.retrieve("Consumer Duty obligations", top_k=20)
    """

    def __init__(self, embedder, vector_store):
        """
        Args:
            embedder:     Embedder instance (BAAI/bge-m3).
            vector_store: VectorStore instance connected to ChromaDB.
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Embed query and return the top_k most similar chunks.

        Args:
            query:  Plain-text query string.
            top_k:  Number of candidates to return.
            where:  Optional ChromaDB metadata filter (e.g. {"source": "fca"}).

        Returns:
            List of dicts: {chunk_id, text, score, metadata}
            score = cosine similarity in [0, 1].
        """
        query_vector = self.embedder.embed_one(query)
        results = self.vector_store.query(query_vector, top_k=top_k, where=where)
        logger.debug(f"DenseRetriever: {len(results)} candidates for query '{query[:60]}'")
        return results
