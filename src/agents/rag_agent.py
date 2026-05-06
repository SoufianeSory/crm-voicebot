"""
RAG Agent — recherche sémantique dans Qdrant.
Fixes appliqués :
- Score agrégé (max + moyenne pondérée)
- Filtrage qualité MIN_SCORE
- Déduplication chunks
- Multi-query retrieval (question originale + reformulée)
- Contexte propre sans mentions de sources
- fast_mode : 1 seule query pour le voicebot (latence réduite)
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np
import logging
from langchain.schema import Document
from typing import List, Tuple
from src.ingestion.embedder import get_vector_store
from config.settings import TOP_K

logger = logging.getLogger(__name__)

MIN_SCORE = 0.55


def retrieve_with_scores(
    question: str,
    rewritten: str = "",
    intent: str = "faq",
    fast_mode: bool = False,   # True pour le voicebot → 1 query, top_k réduit
) -> Tuple[List[Document], float]:
    """
    Recherche multi-query avec score agrégé pondéré et filtrage qualité.

    Args:
        fast_mode : si True, désactive le multi-query et réduit top_k.
                    À utiliser pour le voicebot (latence ~2s au lieu de ~6s).
    """
    vector_store = get_vector_store()

    # ─── top_k ────────────────────────────────────────────────────────────────
    if fast_mode:
        top_k = 4   # 1 seul embed, moins de chunks → réponse plus rapide
    elif intent == "troubleshoot":
        top_k = 12
    else:
        top_k = TOP_K

    # ─── Queries ──────────────────────────────────────────────────────────────
    if fast_mode:
        # Voicebot : on utilise uniquement la question reformulée (meilleure)
        queries = [rewritten if rewritten else question]
    else:
        # Chatbot texte : multi-query pour meilleure précision
        queries = [question]
        if rewritten and rewritten != question:
            queries.append(rewritten)

    # ─── Retrieval ────────────────────────────────────────────────────────────
    all_results = {}
    for query in queries:
        results = vector_store.similarity_search_with_score(query, k=top_k)
        for doc, score in results:
            key = doc.page_content[:120]
            if key not in all_results or score > all_results[key][1]:
                all_results[key] = (doc, score)

    if not all_results:
        return [], 0.0

    # ─── Filtrage qualité ─────────────────────────────────────────────────────
    filtered = [(doc, score) for _, (doc, score) in all_results.items() if score >= MIN_SCORE]

    if not filtered:
        logger.warning(f"Aucun chunk au-dessus du seuil {MIN_SCORE} — scores bruts : {[round(s,3) for _,(d,s) in all_results.items()]}")
        filtered = sorted(all_results.values(), key=lambda x: x[1], reverse=True)[:3]

    # ─── Reranking ────────────────────────────────────────────────────────────
    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]

    docs   = [doc for doc, _ in filtered]
    scores = [score for _, score in filtered]

    # ─── Score agrégé pondéré ─────────────────────────────────────────────────
    top3_mean  = float(np.mean(scores[:3]))
    confidence = float(0.6 * max(scores) + 0.4 * top3_mean)

    mode_label = "FAST" if fast_mode else "FULL"
    logger.info(f"[RAG/{mode_label}] Query: {rewritten or question}")
    logger.info(f"[RAG/{mode_label}] Chunks après filtre: {len(docs)} | Scores: {[round(s,3) for s in scores[:5]]}")
    logger.info(f"[RAG/{mode_label}] Confidence agrégée: {round(confidence, 3)}")

    return docs, confidence


def format_context(docs: List[Document]) -> str:
    """Contexte propre sans mention de sources."""
    if not docs:
        return "Aucun contexte trouvé."
    return "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    question  = "j'ai un problème avec mon application"
    rewritten = "Comment signaler un problème technique avec l'application ?"

    print("=== Mode FULL (chatbot) ===")
    docs, score = retrieve_with_scores(question, rewritten, intent="troubleshoot", fast_mode=False)
    print(f"Score: {score:.3f} | Chunks: {len(docs)}")

    print("\n=== Mode FAST (voicebot) ===")
    docs, score = retrieve_with_scores(question, rewritten, intent="faq", fast_mode=True)
    print(f"Score: {score:.3f} | Chunks: {len(docs)}")