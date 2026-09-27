# PDF Vectorisation Pipeline

A two-part pipeline that turns a PDF into searchable vector embeddings and exposes
pure vector-similarity retrieval over HTTP.

1. **Ingestion** (`ingest.py`): extract text page by page, split it into overlapping
   chunks, embed each chunk locally, and store chunk text, page number and embedding
   in a persistent ChromaDB collection.
2. **Retrieval** (`server.py`): a single `POST /query` endpoint that embeds the query
   with the same model and returns the most similar stored chunks.

There is no LLM, no conversational layer, no generated answer and no keyword, full-text
or `LIKE` search. The API returns raw retrieved chunks only.

## Live Deployment

Live deployment: TODO — replace with deployed API URL

---

## Overview

```text
PDF
 ↓
PyMuPDF                 page-by-page text extraction (1-based page numbers kept)
 ↓
500-character chunks    character-based sliding window, per page
 ↓
75-character overlap    stride = 500 - 75 = 425
 ↓
all-MiniLM-L6-v2        local sentence-transformers model, 384-dim, L2-normalised
 ↓
ChromaDB                persistent collection "pdf_chunks" in ./chroma_db (cosine space)
 ↓
FastAPI
 ↓
POST /query             embed query → vector similarity search → [{chunk_text, page_number, score}]
```

### Project structure

```text
nestack-pdf-vectorization/
├── ingest.py          PDF extraction + embeddings + ChromaDB ingestion (also holds shared config)
├── chunker.py         chunking only (chunk_page_text)
├── server.py          FastAPI app: query embedding + vector retrieval
├── requirements.txt   pinned direct dependencies
├── README.md
├── results.json       raw API responses for 6 queries against the assessment PDF
├── ui/index.html      optional browser UI for POST /query (static, no build step)
├── .gitignore
└── chroma_db/         created by ingest.py at runtime — not committed
```

`server.py` imports the model name, embedding function, database path and collection
name from `ingest.py`, so documents and queries are always embedded with exactly the
same model and settings.

---

## Design decisions

### Chunk size: 500 characters, overlap: 75 characters

The text of each page is split with a character-based sliding window:

```text
chunk_size = 500
overlap    = 75
stride     = chunk_size - overlap = 500 - 75 = 425

Chunk 1: characters    0 – 499
Chunk 2: characters  425 – 924
Chunk 3: characters  850 – 1349
...
```

- **500 characters** (roughly 80–100 words) gives a reasonably focused semantic
  passage, typically one or two related requirements or a short paragraph. The chunk
  is specific enough that its embedding represents one topic, while each result stays
  compact enough to read at a glance. It also sits comfortably inside the model's
  256-token input limit: measured on the assessment PDF, chunks are 54–142 tokens, so
  no chunk text is silently truncated before embedding.
- **75 characters of overlap** (15%) means a sentence or phrase that falls on a chunk
  boundary also appears in the neighbouring chunk. This reduces the chance that
  information at a boundary is lost or split so that neither chunk
  represents it well. It is small enough to keep duplication and index size low.
- Chunking is done **per page**, so every chunk belongs to exactly one page and page
  numbers are never mixed. Each chunk is the exact character slice `text[start:start+500]`
  (it is not trimmed), so it can begin or end with a space or mid-word. Short pages yield a
  single chunk, whitespace-only windows are never stored, and the final window stops at
  the end of the text, so no chunk is fully contained in the previous one. The chunker
  rejects `overlap >= chunk_size`, a negative overlap and a non-positive chunk size.

These are practical choices for this assessment, which uses a short, dense requirements
document. They are not universally optimal: longer narrative documents may benefit from
larger chunks or sentence-aware splitting.

### Text extraction: all text is kept

- **All extracted text is preserved.** The assessment asks to extract all text from the
  PDF, so nothing is filtered out, including the running header and footer lines on each
  page (`Technical Assessment - Nestack`, `Do not share or distribute this document.`,
  `Page PAGE`). They appear at the start of each page's first chunk.
- The only change is whitespace normalisation: line breaks, tabs and repeated spaces
  are collapsed to single spaces, so the character budget is spent on content. Every
  stored chunk is a verbatim substring of its page's normalised text; the audit checks this.
- Pages with no extractable text (blank or image-only) print a warning and are skipped,
  and ingestion continues with the remaining pages at their true page numbers. OCR is
  intentionally not implemented.

*Considered and rejected:* removing lines repeated at the top or bottom of every page.
An A/B test on the assessment PDF with six queries improved the rank of the
answer-bearing chunk for one query and left the other five unchanged. That gain was too
small to justify departing from "extract all text", so the source text is kept as extracted.

### Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)

- **Runs locally**: no API key, no network call per request, and no usage cost. The
  model weights (87 MB) are downloaded once from Hugging Face on first run and then cached.
- **Lightweight**: 6 transformer layers and 384-dimensional vectors (the dimension is read
  from the generated embeddings and printed during ingestion). It runs on CPU, and no GPU is needed.
- **Suitable semantic embeddings**: trained for sentence and short-paragraph similarity,
  which matches 500-character chunks and short natural-language queries.
- **Reproducible**: a fixed, open model pinned through `sentence-transformers`, so anyone
  can reproduce the same vectors from these instructions.
- **Same model for documents and queries**: one symmetric model embeds both sides, so there are no
  separate query and document encoders to keep in sync.

Embeddings are L2-normalised (`normalize_embeddings=True`) at both ingestion and
query time.

Alternatives such as OpenAI `text-embedding-3-small`, Cohere Embed or larger local models
(e.g. `all-mpnet-base-v2`, BGE/E5) could give higher retrieval quality but add either an
API key and external dependency or more compute. For this assessment a local,
dependency-free setup was preferred.

### Vector database: ChromaDB

- **Persistent local storage** in `./chroma_db` (`chromadb.PersistentClient`), so data
  survives restarts and ingestion and serving are separate processes.
- **Simple Python API**: `add()` and `query()` with embeddings, documents and metadata.
- **Vector similarity search** via an HNSW index, configured for cosine distance
  (`hnsw:space = cosine`).
- **Metadata support**: `page_number` and `chunk_index` are stored alongside each vector.
- **No external database service**: it runs in-process, which keeps local development
  and review easy.
- **Suitable scale** for a single-document assessment.

FAISS (a library with no built-in persistence or metadata), Pinecone (a managed cloud
service that needs an API key) and pgvector (which needs a PostgreSQL server) are valid
alternatives. ChromaDB is not universally better; it is the simplest fit for this scope.

### Retrieval: vector similarity only

`server.py` embeds the query with the same `embed_texts()` function and model used during
ingestion, then makes a single ChromaDB vector query:

```python
collection.query(query_embeddings=[query_embedding], n_results=min(top_k, collection.count()),
                 include=["documents", "metadatas", "distances"])
```

There is no `where_document` filter, keyword matching, BM25, regex, substring check or
re-ranking step; the results are exactly ChromaDB's nearest neighbours.

### Score

ChromaDB returns a **distance**, not a similarity. The collection is created with
`metadata={"hnsw:space": "cosine"}` (verified in the stored collection configuration:
`space: cosine`), and embeddings are L2-normalised, so:

```text
distance = cosine distance = 1 - cosine_similarity
score    = 1 - distance    = cosine similarity   (returned unrounded, as a JSON float)
```

Higher scores mean more similar. The theoretical range is -1 to 1; the scores in
`results.json` range from 0.0367 to 0.6508. Results are returned in descending score
order. Scores are only meaningful relative to one another within this model and document.

---

## Setup

Tested with **Python 3.12.6 on Windows 11** (CPU only). Python 3.11+ is expected to
work with these pins but has not been tested.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`sentence-transformers` installs PyTorch, which is a large download. On the first
ingestion or server start, the embedding model is downloaded from Hugging Face and
cached. After that, both run without network access. A message about unauthenticated
Hugging Face requests may appear; no token is needed.

---

## Ingestion

```bash
python ingest.py --file document.pdf
```

For the assessment PDF (placed in the project folder):

```bash
python ingest.py --file assessment_standard_vectorization.pdf
```

What happens:

1. The file path is validated (it must exist and have a `.pdf` extension).
2. PyMuPDF opens the PDF and extracts all text page by page, keeping 1-based page numbers.
   Pages without text print `WARNING: Page N contains no extractable text. Skipping.`
3. Whitespace is normalised; no text is removed.
4. `chunker.chunk_page_text` splits each page into 500-character chunks with 75-character overlap.
5. Every chunk is embedded with `all-MiniLM-L6-v2`.
6. The `pdf_chunks` collection in `./chroma_db` is **dropped and recreated**, and all
   chunks are stored with:
   - document: the chunk text
   - metadata: `page_number`, `chunk_index`
   - embedding: the 384-dim vector
   - ID: stable and unique, e.g. `page-1-chunk-0`, `page-1-chunk-1`, `page-2-chunk-0`
7. A summary is printed with counts generated from the actual run.

**Duplicate protection:** because the collection is recreated on every run, running
ingestion repeatedly never accumulates duplicate vectors. It also never keeps stale
chunks from a previously ingested PDF. The collection always reflects exactly the
last ingested file. Tested results: ingesting the assessment PDF twice leaves 9 records,
not 18. Ingesting it (9 chunks) and then a different, smaller PDF (3 chunks) leaves
exactly 3 records, not 12. A running server picks up the new collection on the next
request, with no restart needed.

---

## Start the server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

- API: `http://localhost:8000`
- Swagger UI (interactive docs, "Try it out" for `POST /query`): `http://localhost:8000/docs`

On startup the server loads `all-MiniLM-L6-v2` and connects to `./chroma_db`. It never
reads the PDF, never runs ingestion and never re-creates document embeddings.

### `POST /query`

Request:

```json
{
  "query": "What is vector similarity search?",
  "top_k": 3
}
```

| Field   | Type    | Rules                                                     |
|---------|---------|-----------------------------------------------------------|
| `query` | string  | required; must not be empty or whitespace-only            |
| `top_k` | integer | optional, default `3`; minimum `1`, maximum `20`          |

Response: an array ordered by descending score. Each object contains exactly:

```json
[
  {
    "chunk_text": "string",
    "page_number": 1,
    "score": 0.1234
  }
]
```

(Schema illustration only; see the sample output below for real values.)

If `top_k` exceeds the number of stored chunks, all stored chunks are returned.

| Situation                                          | Status | Body                                                                 |
|----------------------------------------------------|--------|----------------------------------------------------------------------|
| Success                                            | 200    | array of results                                                     |
| Empty/whitespace `query`, missing `query`, `top_k` < 1 or > 20, non-integer `top_k` | 422 | Pydantic validation error describing the field |
| `./chroma_db` missing, collection missing or empty | 503    | `{"detail": "The vector database is empty or missing. Run ingest.py before querying: python ingest.py --file document.pdf"}` |

### Example requests

Linux/macOS (bash/zsh):

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is vector similarity search?","top_k":3}'
```

Windows PowerShell (works in both Windows PowerShell 5.1 and PowerShell 7):

```powershell
$body = @{ query = "What is vector similarity search?"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/query" `
  -ContentType "application/json" -Body $body
```

Windows PowerShell 7.3+ with curl (use `curl.exe`; in Windows PowerShell 5.1, `curl`
is an alias for `Invoke-WebRequest`):

```powershell
curl.exe -X POST "http://localhost:8000/query" `
  -H "Content-Type: application/json" `
  -d '{"query":"What is vector similarity search?","top_k":3}'
```

Windows Command Prompt (cmd.exe):

```bat
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"query\":\"What is vector similarity search?\",\"top_k\":3}"
```

### Browser UI (optional)

`ui/index.html` is a single static page for searching through the API. It calls only
`POST /query`, plus FastAPI's built-in `GET /openapi.json` for the "API online"
indicator, and displays the response unchanged:

- Result cards show rank, page number, score (4 decimals, with the exact value on hover)
  and the chunk text, plus a copy button.
- A **JSON** view shows the exact request body and the unmodified response.
- Validation mirrors the API (non-empty query, `top_k` 1–20). The not-ingested (503) and
  unreachable-server cases show the command that fixes them.
- It supports light and dark themes, keyboard use (`/` focuses search) and a 375 px
  mobile layout, and it respects reduced-motion settings.
- Deep links such as `?q=your+question&k=5` run the search on load. The API URL is
  editable under **API connection**, so the page also works against a deployed API.

With the API running, serve the page from the project folder:

```bash
python -m http.server 5500 --bind 127.0.0.1 --directory ui
```

Then open `http://127.0.0.1:5500`. Opening `ui/index.html` directly from disk also works.
To let the page call the API from another origin, `server.py` enables CORS
(`CORSMiddleware`, any origin, `GET`/`POST`). This adds response headers only, not
endpoints; the API has no authentication or cookies, so this exposes nothing extra.
Retrieval behaviour is unchanged: the API tests and `results.json` were re-run after the
change, with identical results.

---

## Sample output (from the provided assessment PDF)

Real output from a clean environment (fresh virtual environment, `pip install -r requirements.txt`, no pre-existing `chroma_db`).

Ingestion of `assessment_standard_vectorization.pdf`:

```text
Reading PDF: assessment_standard_vectorization.pdf

Created 9 chunks from 2 page(s).
Loading embedding model: all-MiniLM-L6-v2
Generated 9 embeddings (dimension 384).
========================================
PDF INGESTION COMPLETE
========================================
Pages processed: 2
Pages with text: 2
Pages skipped: 0
Chunks created: 9
Embedding model: all-MiniLM-L6-v2
Vector database: ChromaDB
Collection: pdf_chunks
Persistence: ./chroma_db
========================================
```

`POST /query` with `{"query": "What is vector similarity search?", "top_k": 3}`:

```json
[
  {
    "chunk_text": "2 - RETRIEVAL ENDPOINT Expose a single HTTP endpoint that embeds the incoming query using the same model used during ingestion and performs a similarity search against the stored vectors. Method Endpoint Description POST /query Request body: { \"query\": string, \"top_k\": int }. Response: array of { chunk_text, page_number, score }. Retrieval must use vector similarity search against the database. Do not use keyword search, SQL LIKE, or full-text search of any kind. CONSTRAINTS • No LangChain, Llam",
    "page_number": 1,
    "score": 0.34982830286026
  },
  {
    "chunk_text": "ntically similar chunks from the stored embeddings. There is no LLM call on the retrieval side and no conversational layer. Return raw chunk results only. PART 1 - INGESTION Write a script that accepts a PDF file path and executes the following steps in order: 1. Extract all text from the PDF, preserving page numbers. Handle multi-page documents. 2. Split the extracted text into overlapping chunks. Chunk size and overlap are your decision - justify both in the README. 3. Generate a vector embedd",
    "page_number": 1,
    "score": 0.31477218866348267
  },
  {
    "chunk_text": "Technical Assessment - Nestack Do not share or distribute this document. Page PAGE PDF Vectorisation Pipeline Language Python or Node.js Vector DB Your choice OVERVIEW Build a pipeline with two components: (1) an ingestion script that extracts text from a PDF, chunks it, generates vector embeddings, and stores them in a vector database; and (2) a retrieval endpoint that accepts a plain-text query and returns the most semantically similar chunks from the stored embeddings. There is no LLM call on",
    "page_number": 1,
    "score": 0.31066739559173584
  }
]
```

Chunks are exact 500-character windows, so they can begin or end mid-word (e.g. `ntically`,
`Llam`). This is expected with a fixed character window and is what the overlap
compensates for. The running header (`Technical Assessment - Nestack Do not share or
distribute this document. Page PAGE`) appears at the start of each page's first chunk,
because all text is preserved.

---

## results.json

`results.json` contains the **exact HTTP response bodies** from `POST /query` for six
queries against the supplied assessment PDF. It was generated by a script that calls the
running API and writes the parsed responses unchanged; nothing was edited or invented, and scores are
not rounded. Re-running the same queries in a separate clean environment produced an
identical file.

Structure: a JSON array with one object per query. **Each `response` value is the
exact, unmodified `/query` response array**, in the required response format:

```json
[
  {
    "query": "What command is required to run the ingestion script?",
    "query_type": "specific_factual",
    "request": { "query": "What command is required to run the ingestion script?", "top_k": 3 },
    "response": [
      { "chunk_text": "...", "page_number": 1, "score": 0.3242228031158447 }
    ]
  }
]
```

`query`, `query_type` and `request` only label the entry; `request` is the exact request
body that was sent.

### Relevance review of the top 3 results

This review is based on reading the returned chunk text, not only the scores.

| # | Type | Query | Top-3 results (page, score) | Assessment |
|---|------|-------|-----------------------------|------------|
| 1 | Specific factual | What command is required to run the ingestion script? | p1 0.3242 · p1 0.2130 · p1 **0.2113** | The chunk containing `python ingest.py --file document.pdf` is **3rd**. The 1st is the deliverables chunk ("Ingestion script and retrieval server, fully runnable from your README instructions"), which is related but does not contain the command. |
| 2 | Specific factual | Which contributors must be added to the GitHub repository? | p2 0.3587 · p2 **0.3100** · p2 0.0367 | The chunk listing the three contributor emails is **2nd**. The 1st is the neighbouring submission-instructions chunk. |
| 3 | Broad / thematic | What are the main components and requirements of this PDF vectorisation pipeline? | p1 **0.6508** · p1 0.5304 · p1 0.4985 | All three are relevant: overview and two components (1st), constraints (2nd), deliverables (3rd). |
| 4 | Edge / constraint | What retrieval methods are prohibited by the assessment? | p2 0.4035 · p1 **0.4002** · p1 0.3192 | The 2nd result starts "SQL LIKE, or full-text search of any kind. CONSTRAINTS • No LangChain, LlamaIndex, or any agent/RAG framework", so it is relevant. The 1st (evaluation criteria) is not a good match. The chunk with the full sentence "Retrieval must use vector similarity search … Do not use keyword search" ranks 4th, just outside the top 3. |
| 5 | Edge / ambiguous | What restrictions apply to the retrieval process? | p2 0.3550 · p1 **0.3445** · p1 0.3216 | The 2nd result is the retrieval-endpoint chunk with the vector-only rule, the most relevant. The 1st (evaluation criteria: "retrieval returns relevant results", "Retrieval quality") is a partial match for an ambiguous question. |
| 6 | Additional | How will the submission be evaluated and weighted? | p2 **0.3667** · p2 0.1833 · p1 0.1707 | The evaluation table is 1st, clearly ahead; the 2nd is its continuation. |

Scores in this table are shown to 4 decimals for readability; `results.json` holds the
unrounded values. Bold marks the chunk that best answers the query. For every query, at least one
answer-bearing chunk is in the top 3.

**Observations (honest assessment)**

- Broad and descriptive queries work well. Precise lookups are weaker with a small
  general-purpose model. The command chunk (query 1) also contains text about embedding
  models, vector databases and the start of Part 2, and MiniLM embeds a chunk's overall
  topic, not a literal command string. The score gaps are small (0.2130 vs 0.2113).
- The page-2 evaluation chunk ranks high for several "retrieval"/"assessment" queries,
  because it contains the words *Technical Assessment*, *retrieval* and *relevant*.
- The 75-character overlap visibly helps: in query 4, the prohibition sentence is split
  across a chunk boundary, and the overlap puts "SQL LIKE, or full-text search of any
  kind" into the next chunk, which is returned 2nd.
- Checked and ruled out as causes: the collection is configured for cosine distance, and
  embeddings are L2-normalised (unit length). Queries and documents use the same
  `embed_texts()` function, and no chunk exceeds the model's 256-token limit (max 142).
- No keyword search or re-ranking was added to push specific chunks up; results are
  pure vector similarity. Possible improvements outside this assessment's scope: smaller or
  sentence-aware chunks, a larger embedding model, or a cross-encoder re-ranker.

### Regenerating results.json

1. Run ingestion: `python ingest.py --file assessment_standard_vectorization.pdf`
2. Start FastAPI: `uvicorn server:app --host 0.0.0.0 --port 8000`
3. Execute at least three `POST /query` requests (curl, PowerShell or `/docs`).
4. Copy the actual JSON responses.
5. Save them into `results.json` using the structure above, one `response` per query.
6. Validate that `results.json` contains real results from the supplied PDF. The chunk
   text should match the PDF, page numbers should be 1 or 2, and scores should be the
   unmodified API values.

Scores are deterministic for the same model version and input, apart from small
floating-point differences across hardware or library versions.

---

## Dependencies

`requirements.txt` pins every direct dependency (tested together on Python 3.12.6):

```text
PyMuPDF==1.28.2
sentence-transformers==6.1.0
chromadb==1.5.9
fastapi==0.141.1
uvicorn==0.54.0
pydantic==2.13.5
```

No LangChain, LlamaIndex or other RAG/agent framework is used, only raw library calls.
PyMuPDF is imported as `pymupdf` (aliased to `fitz`), because the bare `import fitz`
name is deprecated in current PyMuPDF releases.

---

## Deployment notes

Not deployed yet; see [Live Deployment](#live-deployment).

- **Storage persistence.** `./chroma_db` is a local directory. On a host whose filesystem
  is ephemeral (many container and PaaS platforms reset it on restart or redeploy), the
  vectors disappear, and `/query` then returns HTTP 503 until ingestion runs again. Use
  one of these:
  1. **A persistent disk or volume** mounted at the project's `chroma_db/` directory.
     Run `python ingest.py --file assessment_standard_vectorization.pdf` once after the
     first deploy.
  2. **Populate at build/start time.** Run the ingestion command as part of the build or
     the start command (before `uvicorn`), so every new instance rebuilds `chroma_db`
     from the PDF. The PDF must then be available to the deployment privately, since it
     is not in the repository.
- **Start command:** `uvicorn server:app --host 0.0.0.0 --port $PORT` (use the port the platform provides).
- **Memory.** The sentence-transformers/PyTorch runtime and embedding model need a
  memory budget suitable for the chosen hosting plan. Measured locally on Windows 11 /
  Python 3.12 (CPU), the server process used about 570 MB resident memory (1.5 GB
  committed) after startup and about 25 queries. This is one measurement, not a
  guarantee; approximately 1 GB or more may be appropriate depending on the environment.
- The model is downloaded from Hugging Face on first start, so the host needs outbound
  internet access at least once (or a pre-populated Hugging Face cache).

---

## GitHub submission

- The repository must be **PRIVATE**. Public repositories are disqualified. Never make it public.
- Repository name: `{yourName}_Nestack_Submission`, e.g. `ravindra_Nestack_Submission`.
- Add these contributors:
  - bishal@nestack.com
  - sannidhya@nestack.com
  - sanjay@nestack.com
- The submission must include:
  - Private GitHub repository
  - Complete source code
  - ZIP file of the complete codebase
  - Live deployment link (also stated at the top of this README)
  - README.md
- No authentication is used, so no credentials are required.
- `chroma_db/`, virtual environments and `__pycache__/` are ignored by `.gitignore`. PDFs
  are ignored too, because the assessment PDF is marked "Do not share or distribute this
  document". Obtain it from the assessment platform and place it in the project folder
  before running ingestion.
