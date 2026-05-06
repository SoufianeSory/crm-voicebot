"""
FastAPI Backend — endpoint /chat
Lance : cd src/api && python main.py
"""

import sys, os, logging
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.graph.workflow import chat
from src.agents.llm_agent import reset_memory, llm

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chatbot RAG — Support Client",
    description="API RAG multi-agents : LangGraph + Qdrant + Llama 3.3-70B",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Modèles ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question  : str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response   : str
    intent     : str
    confidence : float
    action     : str
    reason     : str
    nb_docs    : int
    rewritten  : str


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def warmup():
    logger.info("Préchauffage du LLM...")
    try:
        llm.invoke("Bonjour")
        logger.info("LLM prêt ✅")
    except Exception as e:
        logger.warning(f"Préchauffage échoué : {e}")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")

    try:
        result = chat(request.question)
        logger.info(f"Q='{request.question}' | intent={result['intent']} | conf={result['conf_score']} | action={result['action']}")
        return ChatResponse(
            response   = result["response"],
            intent     = result["intent"],
            confidence = result["conf_score"],
            action     = result["action"],
            reason     = result["reason"],
            nb_docs    = result["nb_docs"],
            rewritten  = result["rewritten"],
        )
    except Exception as e:
        logger.error(f"Erreur chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset_session():
    reset_memory()
    return {"status": "ok", "message": "Session réinitialisée"}


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)