"""
Pipeline principal d'ingestion.
Lance ce fichier pour indexer tes documents dans Qdrant.

Usage :
    cd src/ingestion
    python pipeline.py
"""

import logging
import sys
import os

# Remonter automatiquement à la racine du projet
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from src.ingestion.loader   import load_document, load_directory
from src.ingestion.splitter import split_documents
from src.ingestion.embedder import store_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Dossier source fixé automatiquement → data/raw à la racine du projet
SOURCE_PATH = os.path.join(ROOT, "data", "raw")


def run_pipeline():
    """Lance le pipeline complet : load → split → embed → store."""

    logger.info("=" * 50)
    logger.info("PIPELINE D'INGESTION RAG — DÉMARRAGE")
    logger.info(f"Source : {SOURCE_PATH}")
    logger.info("=" * 50)

    # 1. Chargement
    logger.info("[1/3] Chargement des documents...")
    if os.path.isdir(SOURCE_PATH):
        docs = load_directory(SOURCE_PATH)
    else:
        docs = load_document(SOURCE_PATH)

    if not docs:
        logger.error("Aucun document trouvé dans data/raw/ — ajoutes des fichiers PDF/DOCX/TXT")
        return

    # 2. Découpage
    logger.info("[2/3] Découpage en chunks...")
    chunks = split_documents(docs)

    # 3. Embeddings + stockage Qdrant
    logger.info("[3/3] Génération embeddings + stockage Qdrant...")
    store_documents(chunks)

    logger.info("=" * 50)
    logger.info(f"INGESTION TERMINÉE — {len(chunks)} chunks indexés dans Qdrant")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_pipeline()