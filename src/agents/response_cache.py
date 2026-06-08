"""
Response Cache — cache en mémoire pour les réponses fréquentes.
Évite de passer par RAG + LLM pour les questions déjà vues.
100% gratuit — pas de Redis, juste un dict avec TTL.
"""

import time
import logging
import hashlib
from typing import Optional
from dataclasses import dataclass

from config.settings import RESPONSE_CACHE_TTL

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    response: str
    intent: str
    confidence: float
    nb_docs: int
    created_at: float
    hit_count: int = 0


class ResponseCache:
    """Cache LRU simple avec TTL. Thread-safe pas nécessaire car asyncio single-thread."""

    def __init__(self, max_size: int = 200, ttl: int = RESPONSE_CACHE_TTL):
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._ttl = ttl
        self._hits = 0
        self._misses = 0

    def _make_key(self, question: str) -> str:
        """Normalise la question pour maximiser les cache hits."""
        q = question.lower().strip()
        # Supprimer ponctuation et espaces multiples
        q = ''.join(c for c in q if c.isalnum() or c == ' ')
        q = ' '.join(q.split())
        return hashlib.md5(q.encode()).hexdigest()

    def get(self, question: str) -> Optional[dict]:
        """Retourne la réponse cachée ou None."""
        key = self._make_key(question)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Vérifier TTL
        if time.time() - entry.created_at > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None

        entry.hit_count += 1
        self._hits += 1
        logger.info(f"[Cache] HIT '{question[:40]}' (hits={entry.hit_count})")

        return {
            "response": entry.response,
            "intent": entry.intent,
            "confidence": entry.confidence,
            "nb_docs": entry.nb_docs,
            "cached": True,
        }

    def put(self, question: str, response: str, intent: str,
            confidence: float, nb_docs: int):
        """Cache une réponse si elle est de bonne qualité."""
        # Ne pas cacher les escalades ou réponses basse confiance
        if confidence < 0.65:
            return

        key = self._make_key(question)

        # Éviction si plein
        if len(self._cache) >= self._max_size:
            self._evict()

        self._cache[key] = CacheEntry(
            response=response,
            intent=intent,
            confidence=confidence,
            nb_docs=nb_docs,
            created_at=time.time(),
        )
        logger.debug(f"[Cache] PUT '{question[:40]}' (conf={confidence:.2f})")

    def _evict(self):
        """Supprime les entrées les plus anciennes ou les moins utilisées."""
        if not self._cache:
            return
        # Supprimer l'entrée la plus ancienne avec le moins de hits
        worst_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k].hit_count, -self._cache[k].created_at)
        )
        del self._cache[worst_key]

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / total * 100:.1f}%" if total > 0 else "N/A",
        }

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("[Cache] Vidé")


# Singleton global
response_cache = ResponseCache()