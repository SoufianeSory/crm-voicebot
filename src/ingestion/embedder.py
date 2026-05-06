from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.schema import Document
from typing import List
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import (
    OLLAMA_BASE_URL, EMBED_MODEL,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION
)

logger = logging.getLogger(__name__)


def get_embeddings() -> OllamaEmbeddings:
    """Retourne le modèle d'embeddings Ollama."""
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_qdrant_client() -> QdrantClient:
    """Retourne le client Qdrant."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_collection_if_not_exists(client: QdrantClient, vector_size: int = 768):
    """Crée la collection Qdrant si elle n'existe pas encore."""
    existing = [c.name for c in client.get_collections().collections]

    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Collection créée : {QDRANT_COLLECTION}")
    else:
        logger.info(f"Collection existante : {QDRANT_COLLECTION}")


def store_documents(chunks: List[Document]) -> QdrantVectorStore:
    """
    Génère les embeddings et stocke les chunks dans Qdrant.
    Retourne le vector store prêt pour la recherche.
    """
    embeddings = get_embeddings()
    client     = get_qdrant_client()

    # Taille du vecteur nomic-embed-text = 768
    create_collection_if_not_exists(client, vector_size=768)

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        collection_name=QDRANT_COLLECTION,
    )

    logger.info(f"Stockés dans Qdrant : {len(chunks)} chunks")
    return vector_store


def get_vector_store() -> QdrantVectorStore:
    """
    Retourne le vector store existant (pour la recherche).
    À utiliser dans les agents RAG.
    """
    embeddings = get_embeddings()
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        collection_name=QDRANT_COLLECTION,
    )