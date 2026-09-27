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
# CPU-only PyTorch: the default Linux wheel pulls several GB of CUDA libraries
# that a CPU-only host never uses.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

if [ -f "$PDF_NAME" ]; then
  PDF_PATH="$PDF_NAME"
elif [ -f "/etc/secrets/$PDF_NAME" ]; then
  PDF_PATH="/etc/secrets/$PDF_NAME"
else
  echo "ERROR: $PDF_NAME not found. Add it as a Render Secret File." >&2
  exit 1
fi

# Also downloads the embedding model into $HF_HOME so the server starts offline.
python ingest.py --file "$PDF_PATH"
