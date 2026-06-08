"""
FastAPI Backend v5 — Sans cache, prêt pour le multi-sessions.
"""

import sys, os, logging, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import validate_config
from src.graph.workflow import chat
from src.agents.llm_agent import reset_memory, llm
from src.agents.session_memory import reset_session, reset_all, get_all_sessions_stats

logger = logging.getLogger(__name__)


class Metrics:
    def __init__(self):
        self.total = 0
        self.errors = 0
        self.escalations = 0
        self.latencies: list[float] = []

    def record(self, latency: float, action: str):
        self.total += 1
        if action == "escalate": self.escalations += 1
        self.latencies.append(latency)
        if len(self.latencies) > 200:
            self.latencies = self.latencies[-200:]

    def summary(self) -> dict:
        avg = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        p95 = sorted(self.latencies)[int(len(self.latencies) * 0.95)] if len(self.latencies) >= 5 else avg
        return {
            "total": self.total, "errors": self.errors, "escalations": self.escalations,
            "avg_ms": round(avg * 1000, 1), "p95_ms": round(p95 * 1000, 1),
        }

metrics = Metrics()


app = FastAPI(title="Chatbot RAG — Support Client", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    question  : str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=100)

class ChatResponse(BaseModel):
    response   : str
    intent     : str
    confidence : float
    action     : str
    reason     : str
    nb_docs    : int
    rewritten  : str


@app.on_event("startup")
async def startup():
    validate_config()
    logger.info("Préchauffage LLM...")
    try:
        llm.invoke("Bonjour")
        logger.info("LLM prêt ✅")
    except Exception as e:
        logger.warning(f"Préchauffage échoué : {e}")


@app.get("/")
def root():
    return {"status": "ok", "version": "5.0.0"}


@app.get("/health")
def health():
    checks = {}
    try:
        t = time.time()
        llm.invoke("test")
        checks["llm"] = {"ok": True, "ms": round((time.time() - t) * 1000)}
    except Exception as e:
        checks["llm"] = {"ok": False, "error": str(e)[:80]}
    try:
        from src.ingestion.embedder import get_qdrant_client
        from config.settings import QDRANT_COLLECTION
        info = get_qdrant_client().get_collection(QDRANT_COLLECTION)
        checks["qdrant"] = {"ok": True, "points": info.points_count}
    except Exception as e:
        checks["qdrant"] = {"ok": False, "error": str(e)[:80]}

    checks["sessions"] = get_all_sessions_stats()
    checks["metrics"] = metrics.summary()
    all_ok = all(v.get("ok", True) for v in checks.values() if isinstance(v, dict) and "ok" in v)
    return {"status": "healthy" if all_ok else "degraded", "checks": checks}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    q = request.question.strip()
    if not q:
        raise HTTPException(400, "Question vide")

    start = time.time()
    try:
        result = chat(q, session_id=request.session_id)
        lat = time.time() - start
        metrics.record(lat, result["action"])
        logger.info(f"[Chat] '{q[:50]}' | {result['intent']} | {result['conf_score']:.2f} | {result['action']} | {lat*1000:.0f}ms")
        return ChatResponse(
            response=result["response"], intent=result["intent"],
            confidence=result["conf_score"], action=result["action"],
            reason=result["reason"], nb_docs=result["nb_docs"],
            rewritten=result.get("rewritten", ""),
        )
    except Exception as e:
        metrics.errors += 1
        logger.error(f"[Chat] Erreur : {e}", exc_info=True)
        raise HTTPException(500, "Erreur interne")


@app.post("/reset")
def reset_endpoint(session_id: str = "default"):
    reset_session(session_id)
    return {"ok": True}


@app.post("/reset-all")
def reset_all_endpoint():
    reset_all()
    return {"ok": True}


@app.get("/metrics")
def metrics_endpoint():
    return metrics.summary()


@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Erreur non gérée : {exc}", exc_info=True)
    return JSONResponse(500, {"detail": "Erreur interne"})


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8000)
