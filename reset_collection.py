"""
Vide et recrée la collection Qdrant proprement.
Lance depuis la racine : python reset_collection.py
A utiliser avant chaque réingestion complète.
"""
import sys, os
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config.settings import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Supprimer si existante
existing = [c.name for c in client.get_collections().collections]
if QDRANT_COLLECTION in existing:
    client.delete_collection(QDRANT_COLLECTION)
    print(f"✅ Collection '{QDRANT_COLLECTION}' supprimée")

# Recréer propre
client.create_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)
print(f"✅ Collection '{QDRANT_COLLECTION}' recréée (vide)")
print("→ Lance maintenant : cd src/ingestion && python pipeline.py")
