"""
Ingestion Pipeline
------------------
Reads cached JSONL files from data/raw/, chunks each document,
embeds chunks with BGE-M3, and upserts into ChromaDB.

Usage:
    python -m ingestion.ingest                   # ingest all sources
    python -m ingestion.ingest --source fca      # one source only
    python -m ingestion.ingest --reset           # wipe collection first
    python -m ingestion.ingest --local           # use local ChromaDB (no server)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from config.settings import settings
from data.models import RegulatoryDocument, DataSource
from ingestion.chunker import chunk_document
from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Map source name → JSONL file in data/raw/
SOURCE_FILES = {
    "fca":          Path("data/raw/fca.jsonl"),
    "eurlex":       Path("data/raw/eurlex.jsonl"),
    "financebench": Path("data/raw/financebench.jsonl"),
}


def load_jsonl(path: Path) -> list[RegulatoryDocument]:
    """Deserialise a JSONL cache file into RegulatoryDocument instances."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            docs.append(RegulatoryDocument(
                doc_id=data["doc_id"],
                source=DataSource(data["source"]),
                title=data["title"],
                text=data["text"],
                url=data.get("url"),
                jurisdiction=data.get("jurisdiction"),
                language=data.get("language", "en"),
                category=data.get("category"),
                date_published=data.get("date_published"),
                eval_question=data.get("eval_question"),
                eval_answer=data.get("eval_answer"),
                metadata=data.get("metadata", {}),
            ))
    return docs


def ingest_source(
    source_name: str,
    path: Path,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Chunk, embed, and upsert all documents from one source. Returns chunk count."""
    if not path.exists():
        logger.warning(f"  {source_name}: {path} not found, skipping.")
        return 0

    logger.info(f"[{source_name.upper()}] Loading {path} ...")
    docs = load_jsonl(path)
    logger.info(f"[{source_name.upper()}] {len(docs)} documents loaded.")

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

    logger.info(f"[{source_name.upper()}] {len(all_chunks)} chunks created.")

    # Embed in one pass (batched internally by Embedder)
    logger.info(f"[{source_name.upper()}] Embedding {len(all_chunks)} chunks ...")
    t0 = time.time()
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed(texts)
    elapsed = time.time() - t0
    logger.info(
        f"[{source_name.upper()}] Embedding done in {elapsed:.1f}s "
        f"({len(all_chunks)/elapsed:.1f} chunks/s)."
    )

    # Upsert into ChromaDB
    logger.info(f"[{source_name.upper()}] Upserting into ChromaDB ...")
    n = store.upsert(all_chunks, embeddings)
    logger.info(f"[{source_name.upper()}] {n} chunks upserted.")
    return n


def main():
    parser = argparse.ArgumentParser(description="Ingest regulatory docs into ChromaDB")
    parser.add_argument(
        "--source",
        choices=list(SOURCE_FILES.keys()),
        default=None,
        help="Ingest a single source (default: all)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the ChromaDB collection before ingesting",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use a local persistent ChromaDB instead of a server",
    )
    parser.add_argument(
        "--local-path",
        default="chroma_db",
        help="Path for local persistent ChromaDB (default: chroma_db/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size (reduce on low-RAM machines)",
    )
    args = parser.parse_args()

    # ── Connect ────────────────────────────────────────────────────────────────
    store = VectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection,
        persistent_path=args.local_path if args.local else None,
    )

    if args.reset:
        logger.info("--reset: wiping existing collection...")
        store.delete_collection()

    # ── Load embedder ──────────────────────────────────────────────────────────
    embedder = Embedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=args.batch_size,
    )

    # ── Ingest ─────────────────────────────────────────────────────────────────
    sources = (
        {args.source: SOURCE_FILES[args.source]}
        if args.source
        else SOURCE_FILES
    )

    total_chunks = 0
    for name, path in sources.items():
        total_chunks += ingest_source(
            source_name=name,
            path=path,
            embedder=embedder,
            store=store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    logger.info(
        f"Ingestion complete. {total_chunks} chunks indexed. "
        f"Collection total: {store.count()} chunks."
    )


if __name__ == "__main__":
    main()
