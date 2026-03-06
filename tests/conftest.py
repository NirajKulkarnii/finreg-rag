"""
Shared pytest fixtures for the finreg-rag test suite.

Provides lightweight, in-memory stand-ins for:
  - RegulatoryDocument
  - Chunk
  - RetrievalResult
  - Fake Embedder / VectorStore / Reranker
  - Async LLM client stub
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.models import DataSource, RegulatoryDocument
from ingestion.chunker import Chunk
from retrieval.models import RetrievalResult


# ── Document fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def fca_doc() -> RegulatoryDocument:
    return RegulatoryDocument(
        doc_id="fca-ps-23-3",
        source=DataSource.FCA,
        title="PS23/3: Consumer Duty — final rules and guidance",
        text=(
            "The Consumer Duty sets higher and clearer standards of consumer protection "
            "across financial services. Firms must act to deliver good outcomes for retail customers. "
            "The duty applies to manufacturers and distributors of products and services.\n\n"
            "Under the Duty, firms must consider the needs, characteristics and objectives of "
            "their customers — including those with characteristics of vulnerability. "
            "Firms must also monitor and regularly review whether they are delivering good outcomes."
        ),
        url="https://www.fca.org.uk/publication/policy/ps23-3.pdf",
        jurisdiction="UK",
        category="consumer_duty",
        date_published="2023-07-31",
    )


@pytest.fixture()
def eurlex_doc() -> RegulatoryDocument:
    return RegulatoryDocument(
        doc_id="celex-32019R2088",
        source=DataSource.EURLEX,
        title="Regulation (EU) 2019/2088 on sustainability-related disclosures (SFDR)",
        text=(
            "Financial market participants and financial advisers shall publish on their websites "
            "information about their policies on the integration of sustainability risks in their "
            "investment decision-making process.\n\n"
            "Where financial market participants consider principal adverse impacts of investment "
            "decisions on sustainability factors, they shall publish a statement on due diligence "
            "policies with respect to those principal adverse impacts."
        ),
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R2088",
        jurisdiction="EU",
        category="sustainable_finance",
        date_published="2019-12-09",
    )


@pytest.fixture()
def long_doc() -> RegulatoryDocument:
    """A document long enough to produce multiple chunks."""
    paragraph = (
        "Capital requirements under the Capital Requirements Regulation (CRR) "
        "establish minimum thresholds that institutions must maintain. "
        "The Common Equity Tier 1 (CET1) ratio is the primary measure of capital adequacy. "
    )
    return RegulatoryDocument(
        doc_id="doc-long-001",
        source=DataSource.FCA,
        title="Capital Requirements Overview",
        text=(paragraph * 50),  # ~3 500 chars → multiple chunks at default size
        url="https://example.com/crr",
        jurisdiction="EU",
        category="banking",
        date_published="2024-01-01",
    )


# ── Chunk fixtures ─────────────────────────────────────────────────────────────

def make_chunk(
    doc_id: str = "doc-001",
    chunk_index: int = 0,
    text: str = "Sample regulatory text about Consumer Duty obligations.",
    source: str = "fca",
    jurisdiction: str = "UK",
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::chunk{chunk_index}",
        doc_id=doc_id,
        text=text,
        chunk_index=chunk_index,
        total_chunks=3,
        metadata={
            "doc_id": doc_id,
            "source": source,
            "title": "Test Document",
            "url": "https://example.com",
            "jurisdiction": jurisdiction,
            "language": "en",
            "category": "test",
            "date_published": "2024-01-01",
            "chunk_index": chunk_index,
            "chunk_id": f"{doc_id}::chunk{chunk_index}",
            "total_chunks": 3,
        },
    )


@pytest.fixture()
def sample_chunks() -> list[Chunk]:
    return [make_chunk(chunk_index=i) for i in range(3)]


# ── RetrievalResult fixtures ──────────────────────────────────────────────────

def make_retrieval_result(
    chunk_id: str = "doc-001::chunk0",
    text: str = "Consumer Duty requires firms to deliver good outcomes for retail customers.",
    score: float = 0.85,
    source: str = "fca",
    title: str = "PS23/3: Consumer Duty",
    url: str = "https://www.fca.org.uk/publication/policy/ps23-3.pdf",
    jurisdiction: str = "UK",
    doc_id: str = "doc-001",
    date_published: str = "2023-07-31",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        dense_score=score,
        bm25_score=0.0,
        doc_id=doc_id,
        source=source,
        title=title,
        url=url,
        jurisdiction=jurisdiction,
        category="consumer_duty",
        date_published=date_published,
        chunk_index=0,
        total_chunks=3,
    )


@pytest.fixture()
def sample_retrieval_results() -> list[RetrievalResult]:
    return [
        make_retrieval_result(
            chunk_id=f"doc-{i:03d}::chunk0",
            doc_id=f"doc-{i:03d}",
            score=0.9 - i * 0.1,
        )
        for i in range(5)
    ]


# ── Fake infrastructure stubs ────────────────────────────────────────────────

class FakeEmbedder:
    """Returns fixed-length zero vectors — no model loading."""
    dimension = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


class FakeVectorStore:
    """In-memory ChromaDB stand-in."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self._data: dict[str, dict] = {}
        if chunks:
            for c in chunks:
                self._data[c.chunk_id] = {
                    "text": c.text,
                    "metadata": c.metadata,
                    "embedding": [0.0] * 1024,
                }
        # Expose _collection for BM25Retriever compatibility
        self._collection = _FakeCollection(self._data)

    def upsert(self, chunks, embeddings):
        for c, e in zip(chunks, embeddings):
            self._data[c.chunk_id] = {
                "text": c.text,
                "metadata": c.metadata,
                "embedding": e,
            }
        self._collection = _FakeCollection(self._data)
        return len(chunks)

    def query(self, query_vector, top_k=20, where=None):
        items = list(self._data.values())[:top_k]
        return [
            {
                "chunk_id": list(self._data.keys())[i],
                "text": item["text"],
                "score": 0.75,
                "metadata": item["metadata"],
            }
            for i, item in enumerate(items)
        ]

    def count(self):
        return len(self._data)

    def delete_collection(self):
        self._data.clear()
        self._collection = _FakeCollection(self._data)


class _FakeCollection:
    """Minimal ChromaDB collection stub for BM25Retriever._build_index."""

    def __init__(self, data: dict):
        self._data = data

    def count(self):
        return len(self._data)

    def get(self, include=None):
        ids = list(self._data.keys())
        docs = [v["text"] for v in self._data.values()]
        metas = [v["metadata"] for v in self._data.values()]
        return {"ids": ids, "documents": docs, "metadatas": metas}


@pytest.fixture()
def fake_vector_store(sample_chunks) -> FakeVectorStore:
    return FakeVectorStore(chunks=sample_chunks)


# ── LLM response builder ──────────────────────────────────────────────────────

def make_llm_response(
    answer: str = "Consumer Duty requires firms to deliver good outcomes.",
    confidence: str = "high",
    follow_up_questions: list[str] | None = None,
    reasoning: str | None = None,
) -> MagicMock:
    """Build a fake openai ChatCompletion response object."""
    payload: dict = {
        "answer": answer,
        "confidence": confidence,
        "follow_up_questions": follow_up_questions or [
            "What are the four consumer outcomes under the Consumer Duty?",
            "When did the Consumer Duty come into force?",
            "How does the Consumer Duty apply to distributors?",
        ],
    }
    if reasoning is not None:
        payload["reasoning"] = reasoning

    content = json.dumps(payload)
    choice = MagicMock()
    choice.message.content = content
    choice.delta.content = None
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture()
def llm_response():
    return make_llm_response()


# ── Async LLM client stub ─────────────────────────────────────────────────────

@pytest.fixture()
def mock_openai_client(llm_response):
    """AsyncOpenAI stub that returns a fixed response."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=llm_response)
    return client


# ── Fake retriever ────────────────────────────────────────────────────────────

class FakeRetriever:
    """Returns a fixed list of RetrievalResults for any query."""

    def __init__(self, results: list[RetrievalResult] | None = None):
        self._results = results or [make_retrieval_result()]

    def retrieve(self, query: str, **kwargs) -> list[RetrievalResult]:
        return self._results


@pytest.fixture()
def fake_retriever(sample_retrieval_results) -> FakeRetriever:
    return FakeRetriever(results=sample_retrieval_results)
