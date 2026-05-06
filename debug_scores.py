"""
Script de debug — affiche les scores bruts retournés par Qdrant
Lance depuis la racine : python debug_scores.py
"""
import sys, os
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.ingestion.embedder import get_vector_store

vector_store = get_vector_store()
question = "How can I contact customer support?"

results = vector_store.similarity_search_with_score(question, k=5)

print(f"\nQuestion : {question}")
print(f"Nombre de résultats : {len(results)}")
print("\n--- Scores bruts Qdrant ---")
for i, (doc, score) in enumerate(results, 1):
    print(f"[{i}] score brut = {score:.6f}  |  texte = {doc.page_content[:80]}...")
