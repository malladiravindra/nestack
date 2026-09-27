#!/usr/bin/env bash
# Render build step: install CPU-only dependencies, then build ./chroma_db.
#
# The assessment PDF is never committed (it is marked "Do not share or
# distribute"). Upload it to Render as a Secret File named
# assessment_standard_vectorization.pdf; Render exposes it in the repo root
# during the build and at /etc/secrets/ at runtime.
set -euo pipefail

PDF_NAME="assessment_standard_vectorization.pdf"

pip install --upgrade pip
pip install -r requirements-ingest.txt

if [ -f "$PDF_NAME" ]; then
  PDF_PATH="$PDF_NAME"
elif [ -f "/etc/secrets/$PDF_NAME" ]; then
  PDF_PATH="/etc/secrets/$PDF_NAME"
elif [ -f "$PDF_NAME.b64" ] || [ -f "/etc/secrets/$PDF_NAME.b64" ]; then
  # Secret Files are pasted as text, so the PDF may be supplied base64-encoded.
  B64="$PDF_NAME.b64"
  [ -f "$B64" ] || B64="/etc/secrets/$PDF_NAME.b64"
  base64 -d "$B64" > "/tmp/$PDF_NAME"
  PDF_PATH="/tmp/$PDF_NAME"
else
  echo "ERROR: $PDF_NAME (or $PDF_NAME.b64) not found. Add it as a Render Secret File." >&2
  exit 1
fi

# Also downloads the ONNX embedding model into ./models so the server starts offline.
python ingest.py --file "$PDF_PATH"
