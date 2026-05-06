"""
Router Agent — détecte l'intention de la question client.
Catégories : faq | troubleshoot | hors_sujet | conversationnel
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from config.settings import LLM_PROVIDER, OLLAMA_BASE_URL, LLM_MODEL, GROQ_API_KEY, GROQ_MODEL

if LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq
    _llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
    def _invoke(prompt: str) -> str:
        return _llm.invoke(prompt).content
else:
    from langchain_ollama import OllamaLLM
    _llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    def _invoke(prompt: str) -> str:
        return _llm.invoke(prompt)


# ─── Réponses hardcodées pour les intents conversationnels ────────────────────

CONVERSATIONAL_RESPONSES = {
    "salutation": "Bonjour ! Je suis votre assistant virtuel. Comment puis-je vous aider aujourd'hui ?",
    "remerciement": "Avec plaisir ! N'hésitez pas si vous avez d'autres questions.",
    "aurevoir": "Au revoir ! Bonne journée à vous.",
    "bien_etre": "Je suis un assistant virtuel, mais je suis là pour vous aider du mieux possible ! Que puis-je faire pour vous ?",
    "default": "Bonjour ! Je suis votre assistant virtuel. Comment puis-je vous aider aujourd'hui ?",
}

ROUTER_PROMPT = """Tu es un assistant qui classe les messages clients.
Réponds UNIQUEMENT par un seul mot parmi : faq | troubleshoot | hors_sujet | conversationnel

- faq             : question générale, information, tarif, horaire, politique, profil, demande de contact support
- troubleshoot    : problème technique, panne, erreur, bug, ça ne marche pas, disponibilité de service
- conversationnel : salutation, remerciement, au revoir, question sur le bot, "comment vas-tu", "tu peux m'aider"
- hors_sujet      : question sans rapport avec le service client (météo, sport, blagues, devoirs...)

Note importante : si le client demande à parler à un humain ou un agent, classe en faq — 
le système de confiance gérera l'escalade si nécessaire.

Exemples :
- "comment contacter le support ?" → faq
- "mon appli plante" → troubleshoot
- "bonjour" → conversationnel
- "salut" → conversationnel
- "merci beaucoup" → conversationnel
- "au revoir" → conversationnel
- "comment vas-tu ?" → conversationnel
- "tu peux m'aider ?" → conversationnel
- "je veux parler à un humain" → faq
- "passez-moi un agent" → faq
- "quelle est la météo ?" → hors_sujet
- "tu peux faire mes devoirs ?" → hors_sujet

Message : {question}
Catégorie :"""

CONVERSATIONAL_SUBTYPE_PROMPT = """Classe ce message en un seul mot parmi :
salutation | remerciement | aurevoir | bien_etre | default

- salutation  : bonjour, salut, bonsoir, hello, coucou
- remerciement: merci, super merci, parfait merci
- aurevoir    : au revoir, bonne journée, à bientôt, bye
- bien_etre   : comment vas-tu, ça va, tu vas bien
- default     : autre message conversationnel

Message : {question}
Sous-type :"""


def route_question(question: str) -> str:
    """Retourne l'intention : faq | troubleshoot | hors_sujet | conversationnel"""
    prompt = ROUTER_PROMPT.format(question=question)
    result = _invoke(prompt).strip().lower()

    for intent in ["conversationnel", "troubleshoot", "hors_sujet", "faq"]:
        if intent in result:
            return intent

    return "faq"


def get_conversational_response(question: str) -> str:
    """Retourne une réponse directe pour les messages conversationnels."""
    prompt = CONVERSATIONAL_SUBTYPE_PROMPT.format(question=question)
    subtype = _invoke(prompt).strip().lower()

    for key in CONVERSATIONAL_RESPONSES:
        if key in subtype:
            return CONVERSATIONAL_RESPONSES[key]

    return CONVERSATIONAL_RESPONSES["default"]


if __name__ == "__main__":
    print(f"Provider actif : {LLM_PROVIDER}")
    tests = [
        ("Quels sont vos horaires ?", None),
        ("Mon application plante", None),
        ("Je veux parler à un agent humain", None),
        ("Quelle est la météo demain ?", None),
        ("bonjour", "conversationnel"),
        ("merci beaucoup !", "conversationnel"),
        ("comment vas-tu ?", "conversationnel"),
        ("au revoir", "conversationnel"),
        ("salut", "conversationnel"),
    ]
    for q, expected in tests:
        intent = route_question(q)
        marker = "✅" if (expected is None or intent == expected) else "❌"
        print(f"{marker} Q: {q!r}  →  {intent}")
        if intent == "conversationnel":
            print(f"   Réponse : {get_conversational_response(q)}")