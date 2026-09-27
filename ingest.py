"""PDF ingestion: extract text, chunk it, embed it and store it in ChromaDB.

Usage:
    python ingest.py --file document.pdf

The embedding / database constants and helpers defined here are imported by
``server.py`` so that documents and queries are always embedded with exactly
the same model and configuration.
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Harmless on Windows without Developer Mode; hide it to keep output readable.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import chromadb
import pymupdf as fitz  # PyMuPDF; the bare `fitz` alias is deprecated since 1.24
from sentence_transformers import SentenceTransformer

from chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, Chunk, chunk_page_text

# --- Shared configuration (also used by server.py) --------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "pdf_chunks"
# Cosine distance on L2-normalised embeddings: distance = 1 - cosine_similarity.
COLLECTION_METADATA = {"hnsw:space": "cosine"}

EMBEDDING_BATCH_SIZE = 32
_WHITESPACE_RE = re.compile(r"\s+")


def load_embedding_model() -> SentenceTransformer:
    """Load the local sentence-transformers model (downloaded once, then cached)."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Embed texts with the single embedding configuration used everywhere.

    Embeddings are L2-normalised so cosine distance is well defined and the
    returned score is a true cosine similarity.
    """
    vectors = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


# --- Ingestion steps ---------------------------------------------------------
def normalise_text(text: str) -> str:
    """Collapse runs of whitespace (line breaks, tabs, spaces) into single spaces."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def extract_pages(pdf_path: Path) -> tuple[list[tuple[int, str]], int]:
    """Extract all text page by page. No text is removed; only whitespace is normalised.

    Returns:
        A tuple of (list of (1-based page number, text) for pages with text,
        total number of pages in the document). Pages without extractable text
        (blank or image-only) are reported with a warning and skipped.
    """
    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        for index, page in enumerate(doc):
            page_number = index + 1
            try:
                text = normalise_text(page.get_text("text"))
            except Exception as exc:  # a single bad page must not abort ingestion
                print(f"WARNING: Page {page_number} could not be read ({exc}). Skipping.")
                continue
            if not text:
                print(f"WARNING: Page {page_number} contains no extractable text. Skipping.")
                continue
            pages.append((page_number, text))
    return pages, total_pages


def build_chunks(pages: list[tuple[int, str]]) -> list[tuple[str, Chunk, int]]:
    """Chunk every page and assign stable IDs.

    Returns:
        A list of (id, chunk, chunk_index) where chunk_index restarts at 0 on
        every page, giving IDs like ``page-1-chunk-0``.
    """
    records: list[tuple[str, Chunk, int]] = []
    for page_number, text in pages:
        page_chunks = chunk_page_text(
            text, page_number, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP
        )
        for chunk_index, chunk in enumerate(page_chunks):
            chunk_id = f"page-{page_number}-chunk-{chunk_index}"
            records.append((chunk_id, chunk, chunk_index))
    return records


def reset_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Drop and recreate the collection so re-ingestion never leaves duplicates
    or stale chunks from a previously ingested PDF."""
    existing = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    return client.create_collection(name=COLLECTION_NAME, metadata=COLLECTION_METADATA)


def print_summary(total_pages: int, pages_with_text: int, chunk_count: int) -> None:
    """Print the ingestion summary."""
    line = "=" * 40
    print(line)
    print("PDF INGESTION COMPLETE")
    print(line)
    print(f"Pages processed: {total_pages}")
    print(f"Pages with text: {pages_with_text}")
    print(f"Pages skipped: {total_pages - pages_with_text}")
    print(f"Chunks created: {chunk_count}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print("Vector database: ChromaDB")
    print(f"Collection: {COLLECTION_NAME}")
    print("Persistence: ./chroma_db")
    print(line)


def ingest(pdf_path: Path) -> None:
    """Run the full ingestion pipeline for one PDF."""
    print(f"Reading PDF: {pdf_path}")
    pages, total_pages = extract_pages(pdf_path)
    if not pages:
        raise RuntimeError("No extractable text found in the PDF; nothing to ingest.")

    records = build_chunks(pages)
    print(f"Created {len(records)} chunks from {len(pages)} page(s).")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = load_embedding_model()
    texts = [chunk["text"] for _, chunk, _ in records]
    embeddings = embed_texts(model, texts)
    print(f"Generated {len(embeddings)} embeddings (dimension {len(embeddings[0])}).")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = reset_collection(client)
    collection.add(
        ids=[chunk_id for chunk_id, _, _ in records],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"page_number": chunk["page_number"], "chunk_index": chunk_index}
            for _, chunk, chunk_index in records
        ],
    )

    print_summary(total_pages, len(pages), collection.count())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract, chunk, embed and store a PDF in ChromaDB."
    )
    parser.add_argument("--file", required=True, type=Path, help="Path to the PDF file.")
    return parser.parse_args()


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args()
    pdf_path: Path = args.file
    if not pdf_path.is_file():
        print(f"ERROR: File not found: {pdf_path}", file=sys.stderr)
        return 1
    if pdf_path.suffix.lower() != ".pdf":
        print(f"ERROR: Expected a .pdf file, got: {pdf_path}", file=sys.stderr)
        return 1

    try:
        ingest(pdf_path)
    except (fitz.FileDataError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
