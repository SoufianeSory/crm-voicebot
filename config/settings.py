from dotenv import load_dotenv
import os

load_dotenv()

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
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ─── RAG ──────────────────────────────────────────────────────────────────────
TOP_K                = 5
MIN_SCORE            = 0.55   # Seuil qualité chunks (filtre)
MIN_SCORE_FALLBACK   = 0.45   # Seuil permissif pour la 2e recherche
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.60))