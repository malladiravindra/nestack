"""Retrieval API: embed a query and return the most similar stored chunks.

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000

The server never reads the PDF and never runs ingestion; it only queries the
ChromaDB collection populated by ``ingest.py``. Retrieval is pure vector
similarity search - there is no keyword, full-text or LLM component.
"""

from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

import chromadb
from chromadb.errors import NotFoundError
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints

from ingest import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, embed_texts, load_embedding_model

MAX_TOP_K = 20
NOT_INGESTED_MESSAGE = (
    "The vector database is empty or missing. Run ingest.py before querying: "
    "python ingest.py --file document.pdf"
)

_state: dict = {}


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = Field(
        description="Plain-text query. Must not be empty.",
        examples=["What is vector similarity search?"],
    )
    top_k: int = Field(
        default=3, ge=1, le=MAX_TOP_K, description="Number of chunks to return."
    )


class QueryResult(BaseModel):
    """A single retrieved chunk."""

    chunk_text: str
    page_number: int
    score: float = Field(description="1 - cosine distance (= cosine similarity), unrounded.")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model once at startup."""
    _state["model"] = load_embedding_model()
    yield
    _state.clear()


app = FastAPI(
    title="PDF Vectorisation Pipeline - Retrieval API",
    description=f"Vector similarity search over PDF chunks embedded with {EMBEDDING_MODEL_NAME}.",
    lifespan=lifespan,
)
# Lets the static browser UI (ui/index.html) call the API from another origin.
# Adds response headers only; it does not add any endpoint. The API has no
# authentication or cookies, so allowing any origin exposes nothing extra.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def get_collection() -> chromadb.Collection:
    """Open the persisted collection, or raise 503 if ingestion has not run.

    The collection is looked up per request so that re-running ingest.py
    (which recreates the collection) is picked up without restarting.
    The existence check avoids PersistentClient creating an empty ./chroma_db.
    """
    if not CHROMA_DIR.is_dir():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, NOT_INGESTED_MESSAGE)
    if "client" not in _state:
        _state["client"] = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = _state["client"].get_collection(COLLECTION_NAME)
    except (NotFoundError, ValueError):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, NOT_INGESTED_MESSAGE)
    if collection.count() == 0:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, NOT_INGESTED_MESSAGE)
    return collection


@app.post("/query", response_model=list[QueryResult])
def query(request: QueryRequest) -> list[QueryResult]:
    """Embed the query and return the top_k most similar chunks by cosine similarity."""
    collection = get_collection()
    query_embedding = embed_texts(_state["model"], [request.query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(request.top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    return [
        QueryResult(
            chunk_text=document,
            page_number=int(metadata["page_number"]),
            score=1.0 - distance,
        )
        for document, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
