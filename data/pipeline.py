"""
Data Pipeline
-------------
Orchestrates all three loaders (FCA, EUR-Lex, FinanceBench) into a single
unified ingestion pipeline.

This is the main entry point for populating the knowledge base. Run it once
before starting the API server.

Usage:
    # From repo root
    python -m data.pipeline

    # Or programmatically
    pipeline = DataPipeline()
    docs = pipeline.run(sources=["fca", "eurlex", "financebench"])
"""

import logging
import json
from pathlib import Path
from typing import Optional

from .loaders import FCALoader, EURLexLoader, FinanceBenchLoader
from .models import RegulatoryDocument

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent / "raw"


class DataPipeline:
    """
    Unified data pipeline for regiq.

    Runs all configured loaders, deduplicates on doc_id, and returns a
    flat list of RegulatoryDocument instances ready for ingestion.

    Also saves a raw JSON cache to data/raw/ so you don't need to re-download
    on every run.
    """

    def __init__(
        self,
        cache: bool = True,
        cache_dir: Path = RAW_DATA_DIR,
    ):
        self.cache = cache
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        sources: Optional[list[str]] = None,
        fca_max: int = 50,
        eurlex_max: int = 200,
        financebench_max: int = 50,
    ) -> list[RegulatoryDocument]:
        """
        Run the full pipeline.

        Args:
            sources: List of source names to run. Defaults to all three.
                     Options: "fca", "eurlex", "financebench"
            fca_max:          Max FCA documents to load.
            eurlex_max:       Max EUR-Lex documents to load.
            financebench_max: Max FinanceBench source documents to load.

        Returns:
            Deduplicated list of RegulatoryDocument instances.
        """
        if sources is None:
            sources = ["fca", "eurlex", "financebench"]

        all_docs: dict[str, RegulatoryDocument] = {}  # keyed by doc_id for dedup

        if "fca" in sources:
            fca_docs = self._load_fca(max_docs=fca_max)
            for d in fca_docs:
                all_docs[d.doc_id] = d
            logger.info(f"Pipeline: {len(fca_docs)} FCA docs loaded.")

        if "eurlex" in sources:
            eurlex_docs = self._load_eurlex(max_docs=eurlex_max)
            for d in eurlex_docs:
                all_docs[d.doc_id] = d
            logger.info(f"Pipeline: {len(eurlex_docs)} EUR-Lex docs loaded.")

        if "financebench" in sources:
            fb_docs = self._load_financebench(max_docs=financebench_max)
            for d in fb_docs:
                all_docs[d.doc_id] = d
            logger.info(f"Pipeline: {len(fb_docs)} FinanceBench docs loaded.")

        docs = list(all_docs.values())
        logger.info(f"Pipeline: {len(docs)} total documents after deduplication.")

        if self.cache:
            self._save_cache(docs)

        return docs

    def load_from_cache(self) -> list[RegulatoryDocument]:
        """
        Load previously cached documents without re-downloading.
        Useful for faster dev iteration.
        """
        from .models import DataSource
        cache_file = self.cache_dir / "documents.jsonl"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"No cache found at {cache_file}. Run pipeline.run() first."
            )

        docs = []
        with open(cache_file, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                doc = RegulatoryDocument(
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
                )
                docs.append(doc)

        logger.info(f"Pipeline: loaded {len(docs)} documents from cache.")
        return docs

    def print_summary(self, docs: list[RegulatoryDocument]) -> None:
        """Prints a breakdown of loaded documents by source and category."""
        from collections import Counter
        sources = Counter(d.source.value for d in docs)
        categories = Counter(d.category for d in docs if d.category)
        languages = Counter(d.language for d in docs)

        print("\n── regiq Data Pipeline Summary ──────────────────────────")
        print(f"  Total documents: {len(docs)}")
        print("\n  By source:")
        for src, count in sources.most_common():
            print(f"    {src:<20} {count}")
        print("\n  By category (top 10):")
        for cat, count in categories.most_common(10):
            print(f"    {cat:<30} {count}")
        print("\n  By language:")
        for lang, count in languages.most_common():
            print(f"    {lang:<10} {count}")
        print("─────────────────────────────────────────────────────────\n")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_fca(self, max_docs: int) -> list[RegulatoryDocument]:
        cache_file = self.cache_dir / "fca.jsonl"
        if self.cache and cache_file.exists():
            logger.info("  FCA: loading from cache...")
            return self._read_cache(cache_file)
        loader = FCALoader()
        docs = loader.load(max_docs=max_docs)
        if self.cache:
            self._write_cache(docs, cache_file)
        return docs

    def _load_eurlex(self, max_docs: int) -> list[RegulatoryDocument]:
        cache_file = self.cache_dir / "eurlex.jsonl"
        if self.cache and cache_file.exists():
            logger.info("  EUR-Lex: loading from cache...")
            return self._read_cache(cache_file)
        loader = EURLexLoader()
        docs = loader.load(max_docs=max_docs, financial_only=True)
        if self.cache:
            self._write_cache(docs, cache_file)
        return docs

    def _load_financebench(self, max_docs: int) -> list[RegulatoryDocument]:
        cache_file = self.cache_dir / "financebench.jsonl"
        if self.cache and cache_file.exists():
            logger.info("  FinanceBench: loading from cache...")
            return self._read_cache(cache_file)
        loader = FinanceBenchLoader()
        docs = loader.load_documents(max_docs=max_docs)
        if self.cache:
            self._write_cache(docs, cache_file)
        return docs

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _save_cache(self, docs: list[RegulatoryDocument]) -> None:
        cache_file = self.cache_dir / "documents.jsonl"
        self._write_cache(docs, cache_file)
        logger.info(f"  Cache saved → {cache_file}")

    @staticmethod
    def _write_cache(docs: list[RegulatoryDocument], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for doc in docs:
                record = {
                    "doc_id": doc.doc_id,
                    "source": doc.source.value,
                    "title": doc.title,
                    "text": doc.text,
                    "url": doc.url,
                    "jurisdiction": doc.jurisdiction,
                    "language": doc.language,
                    "category": doc.category,
                    "date_published": doc.date_published,
                    "eval_question": doc.eval_question,
                    "eval_answer": doc.eval_answer,
                    "metadata": doc.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_cache(path: Path) -> list[RegulatoryDocument]:
        """Reads a JSONL cache file back into RegulatoryDocument instances."""
        from .models import DataSource
        docs = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                doc = RegulatoryDocument(
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
                )
                docs.append(doc)
        return docs


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sources = sys.argv[1:] if len(sys.argv) > 1 else None
    pipeline = DataPipeline(cache=True)
    docs = pipeline.run(sources=sources)
    pipeline.print_summary(docs)