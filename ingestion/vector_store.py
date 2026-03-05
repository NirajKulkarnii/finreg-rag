"""
ChromaDB Vector Store
---------------------
Wraps chromadb to upsert and query document chunks.

Collection schema:
  - ids:        chunk_id strings
  - embeddings: 1024-dim BGE-M3 float vectors
  - documents:  raw chunk text (stored for retrieval)
  - metadatas:  flat dict from RegulatoryDocument.to_metadata() + chunk fields
"""

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Batch size for ChromaDB upserts (avoids gRPC message size limits)
UPSERT_BATCH = 100


class VectorStore:
    """
    Manages a ChromaDB collection for regulatory document chunks.

    Connects to a running ChromaDB server (HTTP client) or an
    in-process persistent client depending on config.

    Usage:
        store = VectorStore(host="localhost", port=8001)
        store.upsert(chunks, embeddings)
        results = store.query(query_vector, top_k=10)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8001,
        collection_name: str = "regiq_docs",
        persistent_path: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self._client = self._connect(host, port, persistent_path)
        self._collection = self._get_or_create_collection()

    # ── Public API ────────────────────────────────────────────────────────────

    def upsert(self, chunks, embeddings: list[list[float]]) -> int:
        """
        Upsert chunks with their embeddings into ChromaDB.

        Args:
            chunks:     List of Chunk instances.
            embeddings: Parallel list of embedding vectors.

        Returns:
            Number of chunks upserted.
        """
        assert len(chunks) == len(embeddings), "chunks and embeddings must be same length"

        total = 0
        for i in range(0, len(chunks), UPSERT_BATCH):
            batch_chunks = chunks[i:i + UPSERT_BATCH]
            batch_embeds = embeddings[i:i + UPSERT_BATCH]

            self._collection.upsert(
                ids=[c.chunk_id for c in batch_chunks],
                embeddings=batch_embeds,
                documents=[c.text for c in batch_chunks],
                metadatas=[c.metadata for c in batch_chunks],
            )
            total += len(batch_chunks)
            logger.debug(f"  Upserted batch {i // UPSERT_BATCH + 1}: {total} chunks so far")

        return total

    def query(
        self,
        query_vector: list[float],
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Nearest-neighbour search against the collection.

        Args:
            query_vector: Embedded query vector.
            top_k:        Number of results to return.
            where:        Optional ChromaDB metadata filter dict.

        Returns:
            List of dicts with keys: chunk_id, text, score, metadata.
        """
        kwargs = dict(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            output.append({
                "chunk_id": chunk_id,
                "text": text,
                "score": 1 - dist,   # cosine similarity from distance
                "metadata": meta,
            })

        return output

    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._collection.count()

    def delete_collection(self):
        """Drop and recreate the collection (full re-index)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create_collection()
        logger.info(f"Collection '{self.collection_name}' reset.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self, host: str, port: int, persistent_path: Optional[str]):
        if persistent_path:
            logger.info(f"ChromaDB: using persistent local store at {persistent_path}")
            return chromadb.PersistentClient(path=persistent_path)
        logger.info(f"ChromaDB: connecting to {host}:{port}")
        return chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(anonymized_telemetry=False),
        )

    def _get_or_create_collection(self):
        collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{self.collection_name}' ready "
            f"({collection.count()} chunks already indexed)."
        )
        return collection
