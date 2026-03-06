"""
Unit tests for ingestion.chunker
"""

from data.models import DataSource, RegulatoryDocument
from ingestion.chunker import Chunk, chunk_document, iter_chunks


def make_doc(text: str, doc_id: str = "test-doc") -> RegulatoryDocument:
    return RegulatoryDocument(
        doc_id=doc_id,
        source=DataSource.FCA,
        title="Test Document",
        text=text,
        url="https://example.com",
        jurisdiction="UK",
        category="test",
        date_published="2024-01-01",
    )


# ── chunk_document ────────────────────────────────────────────────────────────

class TestChunkDocument:
    def test_empty_text_returns_empty(self):
        chunks = chunk_document(make_doc(""))
        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        chunks = chunk_document(make_doc("   \n\n   "))
        assert chunks == []

    def test_short_text_produces_single_chunk(self):
        chunks = chunk_document(
            make_doc("Short regulatory note about Consumer Duty."),
            chunk_size=512,
        )
        assert len(chunks) == 1
        assert "Consumer Duty" in chunks[0].text

    def test_long_text_produces_multiple_chunks(self, long_doc):
        chunks = chunk_document(long_doc, chunk_size=128, chunk_overlap=16)
        assert len(chunks) > 1

    def test_chunk_id_format(self):
        chunks = chunk_document(make_doc("Some text here."))
        assert chunks[0].chunk_id == "test-doc::chunk0"

    def test_chunk_ids_are_unique(self, long_doc):
        chunks = chunk_document(long_doc, chunk_size=128, chunk_overlap=16)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_total_chunks_backfilled(self, long_doc):
        chunks = chunk_document(long_doc, chunk_size=128, chunk_overlap=16)
        total = len(chunks)
        for c in chunks:
            assert c.total_chunks == total
            assert c.metadata["total_chunks"] == total

    def test_chunk_index_sequential(self, long_doc):
        chunks = chunk_document(long_doc, chunk_size=128, chunk_overlap=16)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.metadata["chunk_index"] == i

    def test_metadata_preserved(self, fca_doc):
        chunks = chunk_document(fca_doc)
        for c in chunks:
            assert c.metadata["source"] == "fca"
            assert c.metadata["jurisdiction"] == "UK"
            assert c.metadata["title"] == fca_doc.title
            assert c.metadata["url"] == fca_doc.url

    def test_doc_id_in_every_chunk(self, fca_doc):
        chunks = chunk_document(fca_doc)
        for c in chunks:
            assert c.doc_id == fca_doc.doc_id
            assert c.metadata["doc_id"] == fca_doc.doc_id

    def test_chunk_text_is_stripped(self):
        doc = make_doc("  Leading spaces.  \n\nSecond paragraph.  ")
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.text == c.text.strip()

    def test_all_text_covered(self):
        """Every word in the original text appears in at least one chunk."""
        doc = make_doc(
            "Alpha beta gamma delta epsilon.\n\n"
            "Zeta eta theta iota kappa."
        )
        combined = " ".join(c.text for c in chunk_document(doc))
        for word in ["Alpha", "beta", "epsilon", "Zeta", "kappa"]:
            assert word in combined

    def test_overlap_creates_shared_content(self):
        """With overlap enabled, consecutive chunks share some words."""
        doc = make_doc(("Word " * 200) * 5)
        chunks = chunk_document(doc, chunk_size=128, chunk_overlap=32)
        if len(chunks) > 1:
            shared = set(chunks[0].text.split()) & set(chunks[1].text.split())
            assert shared  # non-empty intersection

    def test_chunk_size_respected_approximately(self, long_doc):
        chunk_size = 64  # tokens; ≈ 256 chars
        chunks = chunk_document(
            long_doc, chunk_size=chunk_size, chunk_overlap=8
        )
        # Allow 2× budget as a generous upper bound for segment merging
        max_chars = chunk_size * 4 * 2
        for c in chunks:
            assert len(c.text) < max_chars, (
                f"Chunk too long: {len(c.text)} chars"
            )


# ── iter_chunks ───────────────────────────────────────────────────────────────

class TestIterChunks:
    def test_yields_chunks_for_all_docs(self, fca_doc, eurlex_doc):
        chunks = list(iter_chunks([fca_doc, eurlex_doc]))
        doc_ids = {c.doc_id for c in chunks}
        assert fca_doc.doc_id in doc_ids
        assert eurlex_doc.doc_id in doc_ids

    def test_empty_docs_list(self):
        assert list(iter_chunks([])) == []

    def test_returns_chunk_instances(self, fca_doc):
        chunks = list(iter_chunks([fca_doc]))
        assert all(isinstance(c, Chunk) for c in chunks)
