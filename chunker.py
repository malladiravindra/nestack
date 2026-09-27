"""Character-based sliding-window chunking.

This module contains all chunking logic for the pipeline. It is deliberately
independent of PDF parsing, embeddings and storage so it can be reused and
tested in isolation.
"""

from typing import TypedDict

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 75


class Chunk(TypedDict):
    """A single chunk of text tied to the PDF page it came from."""

    text: str
    page_number: int


def chunk_page_text(
    text: str,
    page_number: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split one page of text into overlapping, fixed-size character chunks.

    The window advances by ``stride = chunk_size - overlap`` characters, so with
    the defaults (500 / 75) chunks cover characters 0-499, 425-924, 850-1349, ...

    Chunking is done per page, so every chunk belongs to exactly one page and
    page numbers are never mixed.

    Args:
        text: The extracted text of a single page.
        page_number: The 1-based page number the text came from.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared by consecutive chunks.

    Returns:
        A list of ``{"text": ..., "page_number": ...}`` dictionaries. Text that
        is shorter than ``chunk_size`` yields a single chunk; empty or
        whitespace-only text yields an empty list.

    Raises:
        ValueError: If the size/overlap/page arguments are invalid.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )
    if page_number < 1:
        raise ValueError(f"page_number must be 1-based, got {page_number}")

    stride = chunk_size - overlap
    chunks: list[Chunk] = []

    for start in range(0, len(text), stride):
        # Keep the exact character window; strip() is only used to skip
        # whitespace-only windows so no empty chunk is ever stored.
        window = text[start : start + chunk_size]
        if window.strip():
            chunks.append({"text": window, "page_number": page_number})
        # Once a window reaches the end of the text, any further window would
        # only repeat characters already covered, so stop here.
        if start + chunk_size >= len(text):
            break

    return chunks
