"""
RAG Agent v2 — multi-query + fast_mode voicebot.
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

MIN_SCORE = 0.45


def retrieve_with_scores(
    question : str,
    rewritten: str = "",
    intent   : str = "faq",
    fast_mode: bool = False,
) -> Tuple[List[Document], float]:

    vector_store = get_vector_store()

    if fast_mode:
        top_k      = 4
        retrieve_k = 4
    elif intent == "troubleshoot":
        top_k      = 10
        retrieve_k = 15
    else:
        top_k      = TOP_K
        retrieve_k = TOP_K * 2

    queries = [rewritten if rewritten else question] if fast_mode else [question]
    if not fast_mode and rewritten and rewritten != question:
        queries.append(rewritten)

    all_results = {}
    for query in queries:
        results = vector_store.similarity_search_with_score(query, k=retrieve_k)
        for doc, score in results:
            key = doc.page_content[:120]
            if key not in all_results or score > all_results[key][1]:
                all_results[key] = (doc, score)

    if not all_results:
        return [], 0.0

    filtered = [(doc, score) for _, (doc, score) in all_results.items() if score >= MIN_SCORE]
    if not filtered:
        filtered = sorted(all_results.values(), key=lambda x: x[1], reverse=True)[:5]

    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]
    docs     = [doc for doc, _ in filtered]
    scores   = [score for _, score in filtered]

    top3_mean  = float(np.mean(scores[:3]))
    confidence = float(0.6 * max(scores) + 0.4 * top3_mean)

    logger.info(f"[RAG/{'FAST' if fast_mode else 'FULL'}] Chunks: {len(docs)} | Confidence: {round(confidence, 3)}")
    return docs, confidence


def format_context(docs: List[Document]) -> str:
    if not docs:
        return "Aucun contexte trouvé."
    return "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs, score = retrieve_with_scores("mot de passe oublié", fast_mode=True)
    print(f"Score: {score:.3f} | Chunks: {len(docs)}")