"""
Router Agent v5.1 — Fast classify TOUJOURS actif (chat et voice).

Fix : "bonjour" prenait 11s car fast_mode=False en mode chat.
Maintenant fast_classify tourne TOUJOURS en premier. Le LLM n'est
appelé que si le keyword matching ne matche pas.
"""

import sys, os, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import logging
from config.settings import LLM_PROVIDER, OLLAMA_BASE_URL, LLM_MODEL, GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

if LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq
    _llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY, timeout=10)
    def _invoke(prompt: str) -> str:
        return _llm.invoke(prompt).content
else:
    from langchain_ollama import OllamaLLM
    _llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    def _invoke(prompt: str) -> str:
        return _llm.invoke(prompt)


# ─── Keywords pour classification rapide ──────────────────────────────────────

_GREETINGS_KW = {
    "bonjour", "bonsoir", "salut", "hello", "coucou", "hey", "hi",
    "allô", "allo", "yo",
}
_THANKS_KW = [
    "merci", "parfait", "super", "génial", "excellent", "top", "nickel",
    "ok merci", "d'accord merci", "merci beaucoup", "merci bien",
    "c'est parfait", "impeccable", "c'est super",
]
_GOODBYES_KW = [
    "au revoir", "bye", "bonne journée", "bonne soirée", "à bientôt",
    "ciao", "à plus", "bonne nuit", "à la prochaine",
]
_WELLBEING_KW = [
    "comment vas-tu", "ça va", "comment tu vas", "tu vas bien",
    "comment allez-vous",
]
_CONVERSATIONAL_PHRASES = [
    "parle moi de toi", "tu es qui", "quel âge", "c'est quoi ton nom",
    "qui es-tu", "tu t'appelles comment", "ton genre", "es-tu un robot",
    "es-tu humain", "tu es un bot", "tu es une ia",
]
_TROUBLESHOOT_KW = [
    "erreur", "bug", "plante", "crash", "bloqué", "marche pas",
    "fonctionne pas", "panne", "impossible", "échoue",
    "ne charge pas", "ne s'ouvre pas", "lent", "timeout",
    "502", "500", "404", "ne répond pas", "page blanche",
    "freeze", "gelé",
]


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\sàâäéèêëïîôùûüÿç\'-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def fast_classify(question: str) -> str | None:
    """
    Classification rapide (~0ms). Retourne l'intent ou None.
    Tourne TOUJOURS en premier, en mode chat comme en voice.
    """
    q = _normalize(question)
    words = set(q.split())

    # ═══ ESCALADE DIRECTE — demande d'humain ═══
    # Doit être AVANT tout le reste — priorité absolue
    _ESCALATE_PHRASES = [
        "parler à un humain", "parler a un humain",
        "un humain", "un agent humain", "un vrai humain",
        "agent humain", "conseiller humain", "personne réelle",
        "humain réel", "vrai personne",
        "transférez", "transferez", "transférer", "transferer",
        "passez-moi", "passe-moi", "passez moi", "passe moi",
        "remonte", "remonter", "escalade", "escalader",
        "je ne veux pas parler au bot", "pas parler au bot",
        "pas un bot", "pas un robot", "pas une machine",
        "parler à quelqu'un", "parler a quelqu'un",
        "agent réel", "vrai agent", "être humain",
    ]
    if any(p in q for p in _ESCALATE_PHRASES):
        return "escalade"

    # Greetings — "bonjour", "bonjour je m'appelle Karim", etc.
    if words & _GREETINGS_KW:
        # Mais si la phrase contient aussi une vraie question → faq
        faq_markers = {"comment", "quoi", "quel", "combien", "pourquoi", "est-ce", "puis-je", "quand", "où"}
        if len(words) <= 6 or not (words & faq_markers):
            return "conversationnel"

    # Goodbyes — "au revoir", "merci bonne journée", "ok super à bientôt"
    if any(g in q for g in _GOODBYES_KW):
        return "conversationnel"

    # Thanks — "merci", "ok merci", etc.
    if any(t in q for t in _THANKS_KW):
        return "conversationnel"

    # Wellbeing — "comment vas-tu", "ça va"
    if any(w in q for w in _WELLBEING_KW):
        return "conversationnel"

    # Questions personnelles sur le bot
    if any(p in q for p in _CONVERSATIONAL_PHRASES):
        return "conversationnel"

    # Troubleshoot
    if any(kw in q for kw in _TROUBLESHOOT_KW):
        return "troubleshoot"

    return None


# ─── LLM classification (fallback) ───────────────────────────────────────────

ROUTER_PROMPT = """Tu es un classificateur de messages clients.
Réponds UNIQUEMENT par un seul mot parmi : faq | troubleshoot | hors_sujet | conversationnel

- faq             : question sur le service, ticket, information, tarif, horaire, profil, fonctionnalité
- troubleshoot    : problème technique, panne, erreur, bug, ça ne marche pas
- conversationnel : salutation, remerciement, au revoir, question personnelle sur le bot, bavardage
- hors_sujet      : sans rapport avec le service client (météo, sport, blagues, questions personnelles)

Message : {question}
Catégorie :"""


def route_question(question: str, fast_mode: bool = False) -> str:
    """
    Retourne l'intent. fast_classify tourne TOUJOURS en premier.
    Le LLM n'est appelé que si les keywords ne matchent pas.
    """
    # TOUJOURS essayer fast classify d'abord
    intent = fast_classify(question)
    if intent:
        logger.debug(f"[Router/FAST] '{question[:40]}' → {intent}")
        return intent

    # Fallback LLM si ambiguë
    try:
        result = _invoke(ROUTER_PROMPT.format(question=question)).strip().lower()
        for label in ["conversationnel", "troubleshoot", "hors_sujet", "faq"]:
            if label in result:
                logger.debug(f"[Router/LLM] '{question[:40]}' → {label}")
                return label
        logger.warning(f"[Router] Réponse inattendue '{result}' → faq")
        return "faq"
    except Exception as e:
        logger.error(f"[Router] Erreur LLM : {e} → faq")
        return "faq"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tests = [
        "bonjour",
        "bonjour, je m'appelle Karim",
        "merci beaucoup",
        "ok super, à bientôt",
        "parfait, au revoir",
        "merci, bonne journée",
        "au revoir",
        "parle moi de toi",
        "quel âge as-tu",
        "comment créer un ticket ?",
        "erreur 502",
        "quelle est la météo ?",
    ]
    for q in tests:
        intent = route_question(q)
        print(f"  '{q}' → {intent}")