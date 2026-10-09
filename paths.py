"""Shared filesystem paths for this project."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
# Downloaded/derived indexes (BM25, dense embeddings) - kept separate from
# DATA_DIR's downloaded-and-unpacked raw data, so that directory holds only
# the raw corpus.
INDEX_DIR = DATA_DIR / "indexes"
# Fine-tuned embedding models (see scripts/train.py), one directory per run.
MODELS_DIR = DATA_DIR / "models"
