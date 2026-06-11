"""
LangGraph Workflow v5.2 — unified : suggestions + barge-in context + hors_sujet flow.
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from src.agents.router_agent     import route_question, fast_classify
from src.agents.rewriter_agent   import rewrite_question
from src.agents.rag_agent        import retrieve_with_scores, format_context
from src.agents.llm_agent        import generate_response, generate_conversational, generate_suggestions
from src.agents.confidence_agent import evaluate_confidence
from src.agents.session_memory   import (
    add_turn, get_history_text, get_client_name,
    mark_first_turn_done
)

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=8)


# ─── État ─────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    question    : str
    rewritten   : str
    intent      : str
    context     : str
    rag_score   : float
    nb_docs     : int
    response    : str
    conf_score  : float
    action      : str
    reason      : str
    retry       : bool
    mode        : str
    session_id  : str
    history     : str
    client_name : str


# ─── Nœuds ────────────────────────────────────────────────────────────────────

def node_rewrite_and_route(state: ChatState) -> ChatState:
    """
    TOUJOURS fast_classify en premier (0ms).
    Si ça matche → on skip rewriter + LLM router.
    Sinon → fallback LLM.
    """
    question = state["question"]
    is_voice = state.get("mode", "chat") == "voice"

    try:
        # ════ FAST PATH (0ms) ════
        fast_intent = fast_classify(question)

        if fast_intent == "conversationnel":
            logger.info(f"[Route/FAST] '{question[:40]}' → conversationnel (skip rewriter)")
            return {**state, "rewritten": question, "intent": "conversationnel"}

        if fast_intent == "troubleshoot":
            rewritten = rewrite_question(question, "troubleshoot")
            logger.info(f"[Route/FAST] '{question[:40]}' → troubleshoot | rewritten='{rewritten[:50]}'")
            return {**state, "rewritten": rewritten, "intent": "troubleshoot"}

        # ════ SLOW PATH (LLM) ════
        if is_voice:
            future_rewrite = _executor.submit(rewrite_question, question, "faq")
            future_route   = _executor.submit(route_question, question, True)
            rewritten = future_rewrite.result(timeout=10)
            intent    = future_route.result(timeout=10)
        else:
            rewritten = rewrite_question(question)
            intent    = route_question(rewritten or question, fast_mode=False)

    except Exception as e:
        logger.error(f"[Route] Erreur : {e}")
        rewritten, intent = question, "faq"

    logger.info(f"[Route/LLM] '{question[:50]}' → intent={intent} | rewritten='{rewritten[:50]}'")
    return {**state, "rewritten": rewritten, "intent": intent}


def node_conversational(state: ChatState) -> ChatState:
    """Réponse conversationnelle via LLM — pas de template."""
    try:
        response = generate_conversational(
            question=state["question"],
            mode=state.get("mode", "chat"),
            history=state.get("history", ""),
            client_name=state.get("client_name", ""),
        )
    except Exception as e:
        logger.error(f"[Conversational] Erreur : {e}")
        response = "Oui, je vous écoute !"

    sid = state.get("session_id", "default")
    try:
        add_turn(sid, state["question"], response, "conversationnel", "respond")
        mark_first_turn_done(sid)
    except Exception:
        pass

    return {
        **state,
        "response": response, "action": "respond",
        "conf_score": 1.0, "reason": "conversationnel — réponse LLM",
    }


def node_rag(state: ChatState) -> ChatState:
    fast = state.get("mode", "chat") == "voice"
    try:
        docs, score = retrieve_with_scores(
            question=state["question"], rewritten=state["rewritten"],
            intent=state["intent"], fast_mode=fast,
        )
        context = format_context(docs)
    except Exception as e:
        logger.error(f"[RAG] Erreur : {e}")
        context, docs, score = "Aucun contexte trouvé.", [], 0.0
    return {**state, "context": context, "rag_score": score, "nb_docs": len(docs)}


def node_rag_fallback(state: ChatState) -> ChatState:
    logger.info("[RAG Fallback] Recherche élargie")
    try:
        from src.ingestion.embedder import get_vector_store
        from config.settings import MIN_SCORE_FALLBACK
        import numpy as np

        vs = get_vector_store()
        results = vs.similarity_search_with_score(state["rewritten"] or state["question"], k=15)
        filtered = [(d, s) for d, s in results if s >= MIN_SCORE_FALLBACK]
        if not filtered:
            filtered = sorted(results, key=lambda x: x[1], reverse=True)[:5]
        docs = [d for d, _ in filtered]
        scores = [s for _, s in filtered]
        top3 = float(__import__('numpy').mean(scores[:3])) if len(scores) >= 3 else float(__import__('numpy').mean(scores))
        score = float(0.6 * max(scores) + 0.4 * top3) if scores else 0.0
        context = format_context(docs)
    except Exception as e:
        logger.error(f"[RAG Fallback] Erreur : {e}")
        context, docs, score = state.get("context", ""), [], 0.0
    return {**state, "context": context, "rag_score": score, "nb_docs": len(docs), "retry": True}


def node_llm(state: ChatState) -> ChatState:
    try:
        response = generate_response(
            question=state["question"], context=state["context"],
            mode=state.get("mode", "chat"), history=state.get("history", ""),
            client_name=state.get("client_name", ""),
        )
    except Exception as e:
        logger.error(f"[LLM] Erreur : {e}")
        response = "Excusez-moi, un souci technique. Je vais vous mettre en relation avec un conseiller humain."
    return {**state, "response": response}


def node_evaluate(state: ChatState) -> ChatState:
    r = evaluate_confidence(
        rag_score=state["rag_score"], intent=state["intent"],
        response=state["response"], nb_docs=state["nb_docs"],
    )
    return {**state, "conf_score": r["confidence"], "action": r["action"], "reason": r["reason"]}


def _build_suggest_response(alternatives: list, intro: str) -> str:
    """Formate une réponse avec alternatives + invitation conseiller."""
    if alternatives:
        opts = "\n".join(f"• {q}" for q in alternatives)
        return (
            f"{intro}\n{opts}\n\n"
            f"Souhaitez-vous que je réponde à l'une de ces questions ? "
            f"Ou préférez-vous parler à un conseiller ?"
        )
    return (
        "Je n'ai pas de réponse précise pour cette demande. "
        "Pourriez-vous reformuler votre question ? "
        "Ou souhaitez-vous parler à un conseiller ?"
    )


def node_suggest_alternatives(state: ChatState) -> ChatState:
    """
    Confiance trop basse après fallback RAG (chat uniquement).
    Propose des questions alternatives basées sur le contexte récupéré.
    """
    alternatives = generate_suggestions(state["question"], state.get("context", ""))
    response = _build_suggest_response(
        alternatives,
        "Je n'ai pas trouvé de réponse précise à votre demande. Peut-être cherchez-vous à savoir :"
    )
    sid = state.get("session_id", "default")
    try:
        add_turn(sid, state["question"], response, state.get("intent", "faq"), "suggest")
        mark_first_turn_done(sid)
    except Exception:
        pass
    return {**state, "response": response, "action": "suggest", "reason": "alternatives — confiance basse"}


def node_suggest_hors_sujet(state: ChatState) -> ChatState:
    """
    Question hors-sujet en mode chat : fait un RAG rapide pour trouver
    des sujets voisins et les propose plutôt qu'escalader directement.
    """
    question = state["question"]
    try:
        docs, _ = retrieve_with_scores(question=question, intent="faq", fast_mode=True)
        context = format_context(docs)
    except Exception:
        context = ""

    alternatives = generate_suggestions(question, context) if context else []
    response = _build_suggest_response(
        alternatives,
        "Cette question est hors de mon domaine du service client. Peut-être cherchez-vous à savoir :"
    )
    sid = state.get("session_id", "default")
    try:
        add_turn(sid, question, response, "hors_sujet", "suggest")
        mark_first_turn_done(sid)
    except Exception:
        pass
    return {**state, "response": response, "action": "suggest", "reason": "hors_sujet → alternatives",
            "context": context, "intent": "hors_sujet"}


def node_respond(state: ChatState) -> ChatState:
    sid = state.get("session_id", "default")
    try:
        add_turn(sid, state["question"], state["response"], state["intent"], state["action"])
        mark_first_turn_done(sid)
    except Exception:
        pass
    return state


def node_escalate(state: ChatState) -> ChatState:
    sid    = state.get("session_id", "default")
    mode   = state.get("mode", "chat")
    intent = state.get("intent", "")

    # Message adapté selon la raison de l'escalade
    if intent == "escalade":
        # Le client a explicitement demandé un humain
        if mode == "voice":
            msg = "Bien sûr, je vous transfère tout de suite vers un conseiller humain de notre équipe. Un instant s'il vous plaît."
        else:
            msg = "Bien sûr, je vous mets en relation avec un conseiller humain de notre équipe."
    elif intent == "hors_sujet":
        if mode == "voice":
            msg = "Je ne suis pas en mesure de répondre à cette question. Je vais vous transférer vers un conseiller humain."
        else:
            msg = "Cette question sort de mon domaine. Un conseiller humain va prendre le relais."
    else:
        # Confiance trop basse ou erreur technique
        if mode == "voice":
            msg = "Je n'ai pas cette information sous les yeux. Je vais vous transférer vers un conseiller humain de notre équipe."
        else:
            msg = "Je ne suis pas en mesure de répondre directement. Un conseiller humain va prendre le relais pour vous aider."

    try:
        add_turn(sid, state["question"], msg, intent, "escalate")
        mark_first_turn_done(sid)
    except Exception:
        pass
    return {**state, "response": msg, "action": "escalate"}


# ─── Edges ────────────────────────────────────────────────────────────────────

def route_after_router(state: ChatState) -> Literal["rag", "conversational", "escalate", "suggest_hors_sujet"]:
    intent = state["intent"]
    mode   = state.get("mode", "chat")
    if intent == "escalade":   return "escalate"
    if intent == "hors_sujet":
        # Voice → escalade directe ; Chat → propose des alternatives d'abord
        return "escalate" if mode == "voice" else "suggest_hors_sujet"
    if intent == "conversationnel": return "conversational"
    return "rag"

def decide_action(state: ChatState) -> Literal["respond", "fallback", "escalate", "suggest"]:
    if state["action"] == "respond":    return "respond"
    if state.get("mode") == "voice":    return "escalate"   # voice : escalade directe
    if not state.get("retry", False):   return "fallback"
    return "suggest"                                        # chat : alternatives avant escalade


# ─── Graphe compilé UNE SEULE FOIS ───────────────────────────────────────────

def _build():
    g = StateGraph(ChatState)
    g.add_node("rewrite_route",       node_rewrite_and_route)
    g.add_node("conversational",      node_conversational)
    g.add_node("rag",                 node_rag)
    g.add_node("rag_fallback",        node_rag_fallback)
    g.add_node("llm",                 node_llm)
    g.add_node("evaluate",            node_evaluate)
    g.add_node("respond",             node_respond)
    g.add_node("escalate",            node_escalate)
    g.add_node("suggest_alternatives", node_suggest_alternatives)  # confiance basse (faq/troubleshoot)
    g.add_node("suggest_hors_sujet",   node_suggest_hors_sujet)    # question hors-sujet (chat)

    g.set_entry_point("rewrite_route")
    g.add_conditional_edges("rewrite_route", route_after_router, {
        "rag": "rag", "conversational": "conversational",
        "escalate": "escalate", "suggest_hors_sujet": "suggest_hors_sujet",
    })
    g.add_edge("conversational",      END)
    g.add_edge("suggest_hors_sujet",  END)
    g.add_edge("rag",                 "llm")
    g.add_edge("llm",                 "evaluate")
    g.add_conditional_edges("evaluate", decide_action, {
        "respond":  "respond",
        "fallback": "rag_fallback",
        "escalate": "escalate",
        "suggest":  "suggest_alternatives",
    })
    g.add_edge("rag_fallback",        "llm")
    g.add_edge("respond",             END)
    g.add_edge("escalate",            END)
    g.add_edge("suggest_alternatives", END)
    return g.compile()

_app = _build()
logger.info("[Workflow] Graphe compilé ✅")


# ─── API publique ─────────────────────────────────────────────────────────────

def chat(question: str, session_id: str = "default") -> dict:
    return _run(question, mode="chat", session_id=session_id)

def voice_chat(question: str, session_id: str = "default", interrupted_response: str = "") -> dict:
    """interrupted_response : ce que le bot disait quand le client l'a interrompu (barge-in)."""
    return _run(question, mode="voice", session_id=session_id, interrupted_response=interrupted_response)

def _run(question: str, mode: str = "chat", session_id: str = "default", interrupted_response: str = "") -> dict:
    history     = get_history_text(session_id, max_turns=3)
    client_name = get_client_name(session_id) or ""

    # Contexte de barge-in : le LLM sait ce qu'il disait quand il a été interrompu
    if interrupted_response:
        history += f"\n[Le client t'a interrompu pendant que tu disais : \"{interrupted_response[:150]}\"]\n"

    initial: ChatState = {
        "question": question, "rewritten": "", "intent": "", "context": "",
        "rag_score": 0.0, "nb_docs": 0, "response": "", "conf_score": 0.0,
        "action": "", "reason": "", "retry": False, "mode": mode,
        "session_id": session_id, "history": history, "client_name": client_name,
    }
    try:
        return _app.invoke(initial)
    except Exception as e:
        logger.error(f"[Workflow] Crash : {e}", exc_info=True)
        msg = "Excusez-moi, un problème technique." if mode == "voice" else "Un problème technique est survenu."
        return {**initial, "response": msg, "action": "escalate", "reason": f"crash: {e}", "conf_score": 0.0}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    for q in ["bonjour", "comment créer un ticket ?", "merci", "au revoir"]:
        print(f"\n{'='*50}\nQ : {q}")
        r = chat(q, session_id="test")
        print(f"Intent={r['intent']} | Conf={r['conf_score']:.2f} | Action={r['action']}")
        print(f"Réponse : {r['response']}")