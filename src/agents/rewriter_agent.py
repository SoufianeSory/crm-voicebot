"""
Rewriter Agent v4 — Reformulation enrichie pour le RAG.
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import logging
from config.settings import LLM_PROVIDER, OLLAMA_BASE_URL, LLM_MODEL, GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

if LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq
    _llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY, max_tokens=60, timeout=8)
    def _invoke(prompt: str) -> str:
        return _llm.invoke(prompt).content
else:
    from langchain_ollama import OllamaLLM
    _llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_predict=60)
    def _invoke(prompt: str) -> str:
        return _llm.invoke(prompt)


FAQ_REWRITER = """Reformule ce message client en une question claire pour une recherche FAQ.
Règles : une seule question formelle, UNIQUEMENT la question reformulée, sans guillemets.
Si le message est déjà clair → retourne-le tel quel.

Exemples :
- "mot de passe oublié" → Comment réinitialiser mon mot de passe ?
- "j'arrive pas à me connecter" → Pourquoi je n'arrive pas à me connecter à mon compte ?
- "parler à quelqu'un" → Comment contacter un agent humain du support ?
- "horaires" → Quels sont vos horaires d'ouverture ?
- "changer mon email" → Comment modifier l'adresse email de mon compte ?
- "voir les tickets des autres" → Un client peut-il voir les tickets d'un autre client ?
- "confidentialité entre utilisateurs" → Les données sont-elles séparées entre clients ?
- "rôles utilisateurs" → Quels rôles utilisateurs existent dans le système ?
- "archivage" → Combien de temps les tickets sont-ils conservés ?
- "navigateurs compatibles" → Quels navigateurs sont compatibles avec le système ?
- "double authentification" → Le système supporte-t-il la double authentification ?
- "api limites" → Quelles sont les limites de l'API ?
- "fichiers joints" → Puis-je joindre des fichiers à un ticket ?

Message : {question}
Question reformulée :"""

TROUBLESHOOT_REWRITER = """Reformule ce problème technique en une question claire.
Préserve les codes d'erreur et termes techniques. Retourne UNIQUEMENT la question.

Exemples :
- "erreur 502 depuis ce matin" → Comment résoudre l'erreur 502 ?
- "le module paiement plante" → Pourquoi le module paiement génère-t-il une erreur ?
- "impossible d'exporter en PDF" → Comment résoudre le problème d'export en PDF ?

Message : {question}
Question reformulée :"""


def _clean(text: str) -> str:
    text = text.strip().strip('"').strip("'").strip('«').strip('»')
    for prefix in ["question reformulée :", "reformulée :", "→ "]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def rewrite_question(question: str, intent: str = "faq") -> str:
    try:
        prompt = TROUBLESHOOT_REWRITER.format(question=question) if intent == "troubleshoot" \
            else FAQ_REWRITER.format(question=question)
        rewritten = _clean(_invoke(prompt))
        if len(rewritten) > 250 or "\n" in rewritten or len(rewritten) < 5:
            return question
        return rewritten
    except Exception as e:
        logger.warning(f"[Rewriter] Erreur : {e}")
        return question
