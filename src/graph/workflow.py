"""
LangGraph Workflow — orchestre les 5 agents.

Flux :
    START → rewriter → router → rag → llm → evaluate → END
                          ↓                      ↓
                   conversationnel          respond | escalate(+retry)
                   hors_sujet

Modes :
    mode="chat"  → multi-query RAG, top_k élevé (précision max)
    mode="voice" → single-query RAG, top_k réduit (latence min ~2s)
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import logging
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from src.agents.router_agent     import route_question, get_conversational_response
from src.agents.rewriter_agent   import rewrite_question
from src.agents.rag_agent        import retrieve_with_scores, format_context
from src.agents.llm_agent        import generate_response
from src.agents.confidence_agent import evaluate_confidence

logger = logging.getLogger(__name__)


# ─── État ─────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    question   : str
    rewritten  : str
    intent     : str
    context    : str
    rag_score  : float
    nb_docs    : int
    response   : str
    conf_score : float
    action     : str
    reason     : str
    retry      : bool
    mode       : str   # "chat" | "voice"


# ─── Nœuds ────────────────────────────────────────────────────────────────────

def node_rewriter(state: ChatState) -> ChatState:
    rewritten = rewrite_question(state["question"])
    logger.info(f"[Rewriter] '{state['question']}' → '{rewritten}'")
    return {**state, "rewritten": rewritten}


def node_router(state: ChatState) -> ChatState:
    intent = route_question(state["rewritten"] or state["question"])
    logger.info(f"[Router] intent={intent}")
    return {**state, "intent": intent}


def node_conversational(state: ChatState) -> ChatState:
    """Répond directement aux messages conversationnels sans passer par le RAG."""
    response = get_conversational_response(state["question"])
    logger.info("[Conversational] réponse directe")
    return {
        **state,
        "response"  : response,
        "action"    : "respond",
        "conf_score": 1.0,
        "reason"    : "intent conversationnel — réponse directe",
    }


def node_rag(state: ChatState) -> ChatState:
    fast_mode = state.get("mode", "chat") == "voice"
    docs, score = retrieve_with_scores(
        question  = state["question"],
        rewritten = state["rewritten"],
        intent    = state["intent"],
        fast_mode = fast_mode,
    )
    context = format_context(docs)
    return {**state, "context": context, "rag_score": score, "nb_docs": len(docs)}


def node_rag_fallback(state: ChatState) -> ChatState:
    """2e tentative avec top_k élargi si confiance trop basse."""
    logger.info("[RAG Fallback] Recherche élargie top_k=15")
    from src.ingestion.embedder import get_vector_store
    import numpy as np

    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(
        state["rewritten"] or state["question"], k=15
    )

    filtered = [(doc, s) for doc, s in results if s >= 0.45]
    if not filtered:
        filtered = sorted(results, key=lambda x: x[1], reverse=True)[:5]

    docs   = [doc for doc, _ in filtered]
    scores = [s for _, s in filtered]
    top3   = float(np.mean(scores[:3]))
    score  = float(0.6 * max(scores) + 0.4 * top3)

    context = format_context(docs)
    return {**state, "context": context, "rag_score": score, "nb_docs": len(docs), "retry": True}


def node_llm(state: ChatState) -> ChatState:
    response = generate_response(state["question"], state["context"])
    return {**state, "response": response}


def node_evaluate(state: ChatState) -> ChatState:
    result = evaluate_confidence(
        rag_score = state["rag_score"],
        intent    = state["intent"],
        response  = state["response"],
        nb_docs   = state["nb_docs"],
    )
    return {
        **state,
        "conf_score": result["confidence"],
        "action"    : result["action"],
        "reason"    : result["reason"],
    }


def node_respond(state: ChatState) -> ChatState:
    return state


def node_escalate(state: ChatState) -> ChatState:
    return {
        **state,
        "response": (
            "Je ne suis pas en mesure de répondre à cette demande directement. "
            "Je vais vous mettre en relation avec un de nos conseillers qui pourra vous aider."
        ),
        "action": "escalate",
    }


# ─── Edges conditionnelles ────────────────────────────────────────────────────

def route_after_router(state: ChatState) -> Literal["rag", "conversational", "escalate"]:
    intent = state["intent"]
    if intent == "hors_sujet":
        return "escalate"
    if intent == "conversationnel":
        return "conversational"
    return "rag"


def decide_action(state: ChatState) -> Literal["respond", "fallback", "escalate"]:
    if state["action"] == "respond":
        return "respond"
    # En mode voice, on skip le fallback (trop lent) → escalade directe
    if state.get("mode", "chat") == "voice":
        return "escalate"
    if not state.get("retry", False):
        return "fallback"
    return "escalate"


# ─── Graphe ───────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("rewriter",       node_rewriter)
    graph.add_node("router",         node_router)
    graph.add_node("conversational", node_conversational)
    graph.add_node("rag",            node_rag)
    graph.add_node("rag_fallback",   node_rag_fallback)
    graph.add_node("llm",            node_llm)
    graph.add_node("evaluate",       node_evaluate)
    graph.add_node("respond",        node_respond)
    graph.add_node("escalate",       node_escalate)

    graph.set_entry_point("rewriter")
    graph.add_edge("rewriter", "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "rag"           : "rag",
            "conversational": "conversational",
            "escalate"      : "escalate",
        }
    )

    graph.add_edge("conversational", END)
    graph.add_edge("rag",   "llm")
    graph.add_edge("llm",   "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        decide_action,
        {
            "respond" : "respond",
            "fallback": "rag_fallback",
            "escalate": "escalate",
        }
    )

    graph.add_edge("rag_fallback", "llm")
    graph.add_edge("respond",      END)
    graph.add_edge("escalate",     END)

    return graph.compile()


# ─── Fonctions publiques ──────────────────────────────────────────────────────

def chat(question: str) -> dict:
    """Chatbot texte — précision max (multi-query RAG)."""
    return _run(question, mode="chat")


def voice_chat(question: str) -> dict:
    """Voicebot — latence min (single-query RAG, pas de fallback)."""
    return _run(question, mode="voice")


def _run(question: str, mode: str = "chat") -> dict:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    app = build_graph()

    initial_state: ChatState = {
        "question"  : question,
        "rewritten" : "",
        "intent"    : "",
        "context"   : "",
        "rag_score" : 0.0,
        "nb_docs"   : 0,
        "response"  : "",
        "conf_score": 0.0,
        "action"    : "",
        "reason"    : "",
        "retry"     : False,
        "mode"      : mode,
    }

    return app.invoke(initial_state)


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "j'ai un problème avec mon application",
        "quels sont vos tarifs premium ?",
        "bonjour",
        "merci beaucoup",
    ]
    for q in tests:
        print("\n" + "=" * 55)
        print(f"Q : {q}")
        r = chat(q)
        print(f"Intent     : {r['intent']}")
        print(f"Mode       : {r['mode']}")
        print(f"Confidence : {r['conf_score']:.2f}")
        print(f"Action     : {r['action']}")
        print(f"Réponse    : {r['response']}")