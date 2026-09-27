"""Embedding model and vector-store configuration shared by ingest.py and server.py.

Documents and queries are always embedded here, with exactly the same model and
settings. The model is ``all-MiniLM-L6-v2`` run through ONNX Runtime, using
ChromaDB's built-in ``ONNXMiniLM_L6_V2`` embedding function: the same weights,
tokenizer, 256-token truncation, attention-masked mean pooling and L2
normalisation as the sentence-transformers version, but without PyTorch. That
keeps the retrieval server small enough for a serverless bundle (Vercel's
Python limit is 500 MB; PyTorch alone is ~715 MB installed on Linux).

This module imports only ChromaDB, so it never pulls in PyMuPDF or PyTorch.
"""

import os
import shutil
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

PROJECT_DIR = Path(__file__).resolve().parent

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# ONNX model files (~90 MB), downloaded once by `python embedding.py` or on first use.
MODEL_DIR = PROJECT_DIR / "models" / EMBEDDING_MODEL_NAME

CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "pdf_chunks"
# Cosine distance on L2-normalised embeddings: distance = 1 - cosine_similarity.
COLLECTION_METADATA = {"hnsw:space": "cosine"}

# Vercel sets VERCEL=1. Its deployment filesystem is read-only except /tmp.
ON_VERCEL = os.environ.get("VERCEL") == "1"
_WRITABLE_CHROMA_DIR = Path("/tmp") / "chroma_db"


def load_embedding_model() -> ONNXMiniLM_L6_V2:
    """Load all-MiniLM-L6-v2 (ONNX), downloading it into MODEL_DIR if missing."""
    model = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
    model.DOWNLOAD_PATH = MODEL_DIR
    model._download_model_if_not_exists()
    return model


def embed_texts(model: ONNXMiniLM_L6_V2, texts: list[str]) -> list[list[float]]:
    """Embed texts with the single embedding configuration used everywhere.

    Embeddings are L2-normalised by the model, so cosine distance is well
    defined and the returned score is a true cosine similarity.
    """
    return [vector.tolist() for vector in model(texts)]


def open_chroma_client() -> chromadb.ClientAPI:
    """Open the persisted ChromaDB client for querying.

    On Vercel the bundled ./chroma_db is read-only, so it is copied once per
    instance to /tmp, which is writable.
    """
    path = CHROMA_DIR
    if ON_VERCEL:
        if not _WRITABLE_CHROMA_DIR.is_dir():
            shutil.copytree(CHROMA_DIR, _WRITABLE_CHROMA_DIR)
        path = _WRITABLE_CHROMA_DIR
    return chromadb.PersistentClient(path=str(path))


if __name__ == "__main__":
    # Pre-download the model; used as the Vercel build step so it ships in the bundle.
    load_embedding_model()
    # The extracted files are all that is needed; drop the ~80 MB archive.
    (MODEL_DIR / ONNXMiniLM_L6_V2.ARCHIVE_FILENAME).unlink(missing_ok=True)
    print(f"Embedding model ready: {MODEL_DIR}")
