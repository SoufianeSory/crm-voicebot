"""
config/settings.py — Configuration centralisée et validée.
"""

from dotenv import load_dotenv
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# ─── LLM Provider ─────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# ─── Ollama (local) ───────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "phi3:mini")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ─── Groq (API) ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─── Qdrant ───────────────────────────────────────────────────────────────────
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chatbot_rag")

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# ─── RAG ──────────────────────────────────────────────────────────────────────
TOP_K                = int(os.getenv("TOP_K", 5))
MIN_SCORE            = float(os.getenv("MIN_SCORE", 0.55))
MIN_SCORE_FALLBACK   = float(os.getenv("MIN_SCORE_FALLBACK", 0.45))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.60))

# ─── Session Memory ───────────────────────────────────────────────────────────
SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", 10))

# ─── Voice ────────────────────────────────────────────────────────────────────
VOICE_SILENCE_FRAMES  = int(os.getenv("VOICE_SILENCE_FRAMES", 10))
VOICE_INACTIVITY_SEC  = int(os.getenv("VOICE_INACTIVITY_SEC", 8))
TTS_VOICE             = os.getenv("TTS_VOICE", "fr-FR-DeniseNeural")
TTS_RATE              = os.getenv("TTS_RATE", "+8%")

# ─── Concurrence ──────────────────────────────────────────────────────────────
MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", 50))
LLM_TIMEOUT_SEC         = int(os.getenv("LLM_TIMEOUT_SEC", 15))

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def validate_config():
    """Vérifie la config au startup. Crash immédiat si invalide."""
    errors = []

    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        errors.append("LLM_PROVIDER=groq mais GROQ_API_KEY est vide")

    if errors:
        for e in errors:
            logger.error(f"[Config] ❌ {e}")
        raise RuntimeError(f"Config invalide : {'; '.join(errors)}")

    logger.info(
        f"[Config] ✅ Provider={LLM_PROVIDER} | Model={GROQ_MODEL if LLM_PROVIDER == 'groq' else LLM_MODEL} | "
        f"Qdrant={QDRANT_HOST}:{QDRANT_PORT}/{QDRANT_COLLECTION} | "
        f"TOP_K={TOP_K} | CONFIDENCE={CONFIDENCE_THRESHOLD} | "
        f"MaxSessions={MAX_CONCURRENT_SESSIONS}"
    )
