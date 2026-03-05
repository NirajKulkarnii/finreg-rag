"""
Text Chunker
------------
Splits RegulatoryDocument text into overlapping chunks suitable for embedding.

Strategy:
  - Split on sentence/paragraph boundaries where possible (avoids mid-sentence cuts)
  - Respect chunk_size (in tokens approximated as chars/4) and chunk_overlap
  - Each chunk carries a copy of the parent document metadata plus chunk position
"""

import re
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Chunk:
    chunk_id: str          # "{doc_id}::chunk{n}"
    doc_id: str
    text: str
    chunk_index: int
    total_chunks: int      # filled in after all chunks are known
    metadata: dict


def chunk_document(doc, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    """
    Splits a RegulatoryDocument into overlapping text chunks.

    Args:
        doc:           RegulatoryDocument instance.
        chunk_size:    Target chunk size in tokens (approximated as chars / 4).
        chunk_overlap: Overlap between consecutive chunks in tokens.

    Returns:
        List of Chunk instances.
    """
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4

    segments = _split_into_segments(doc.text)
    raw_chunks = _merge_segments(segments, char_size, char_overlap)

    base_meta = doc.to_metadata()

    chunks = []
    for i, text in enumerate(raw_chunks):
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}::chunk{i}",
            doc_id=doc.doc_id,
            text=text.strip(),
            chunk_index=i,
            total_chunks=len(raw_chunks),
            metadata={
                **base_meta,
                "chunk_index": i,
                "chunk_id": f"{doc.doc_id}::chunk{i}",
            },
        ))

    # Back-fill total_chunks now we know the count
    for chunk in chunks:
        chunk.total_chunks = len(chunks)
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


def iter_chunks(docs, chunk_size: int = 512, chunk_overlap: int = 64) -> Iterator[Chunk]:
    """Yields chunks for a sequence of documents."""
    for doc in docs:
        yield from chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_into_segments(text: str) -> list[str]:
    """Split text into paragraph/sentence segments for clean boundary chunking."""
    # First split on double newlines (paragraph breaks)
    paragraphs = re.split(r"\n{2,}", text)
    segments = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # If a paragraph is very long, further split on sentence boundaries
        if len(para) > 1200:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            segments.extend(s for s in sentences if s.strip())
        else:
            segments.append(para)
    return segments


def _merge_segments(segments: list[str], char_size: int, char_overlap: int) -> list[str]:
    """Greedily merge segments into chunks of ~char_size, with overlap."""
    if not segments:
        return []

    chunks = []
    current_parts: list[str] = []
    current_len = 0

    for seg in segments:
        seg_len = len(seg)
        if current_len + seg_len > char_size and current_parts:
            chunks.append("\n\n".join(current_parts))
            # Build overlap: keep trailing segments that fit within char_overlap
            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current_parts):
                if overlap_len + len(part) <= char_overlap:
                    overlap_parts.insert(0, part)
                    overlap_len += len(part)
                else:
                    break
            current_parts = overlap_parts
            current_len = overlap_len

        current_parts.append(seg)
        current_len += seg_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks
