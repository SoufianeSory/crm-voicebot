"""
LLM Agent v5 — TOUT passe par le LLM. Zéro réponse scriptée.

Philosophie :
    Le bot doit être HUMAIN. Pas de dictionnaire de réponses pré-faites.
    Même un "bonjour" passe par le LLM pour que la réponse soit
    naturelle, variée, et contextuelle (première fois vs client qui revient).

Modes :
    - chat  → réponse texte précise et structurée
    - voice → réponse orale naturelle comme un vrai agent téléphone
"""

import sys, os, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import logging
import time as _time
from config.settings import LLM_PROVIDER, OLLAMA_BASE_URL, LLM_MODEL, GROQ_API_KEY, GROQ_MODEL, LLM_TIMEOUT_SEC

logger = logging.getLogger(__name__)

if LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.4, api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT_SEC)
    def _invoke(prompt: str) -> str:
        # Retry avec backoff pour gérer les rate limits Groq en multi-sessions
        for attempt in range(3):
            try:
                return llm.invoke(prompt).content
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err or "limit" in err:
                    wait = 2 ** attempt + 1
                    logger.warning(f"[LLM] Rate limit (tentative {attempt+1}/3), attente {wait}s...")
                    _time.sleep(wait)
                else:
                    raise
        return llm.invoke(prompt).content  # Dernière tentative, laisse crasher
else:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.4)
    def _invoke(prompt: str) -> str:
        return llm.invoke(prompt)


# ─── Sanitization ─────────────────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    text = re.sub(r'(?i)(ignore|oublie)\s+(les|tout|précédent)', '[filtré]', text)
    text = re.sub(r'(?i)(tu es maintenant|nouveau rôle)', '[filtré]', text)
    text = re.sub(r'(?i)(system|système)\s*:', '[filtré]', text)
    return text.strip()


# ─── Prompt CHAT ──────────────────────────────────────────────────────────────

CHAT_PROMPT = """Tu es Intelcia, assistant du service client. Tu parles comme un vrai humain — pas un robot.

QUI TU ES :
- Un assistant virtuel intelligent du service client, nommé Intelcia
- Tu es chaleureux, efficace, et tu parles naturellement — pas de ton robotique
- Tu t'adaptes au ton du client : formel avec quelqu'un de formel, décontracté avec quelqu'un de cool

TRANSPARENCE :
- Si on te demande si tu es un bot/IA/robot → dis la vérité avec naturel : "Oui, je suis Intelcia, l'assistant virtuel du service client. Mais je suis là pour vous aider au mieux !"
- N'invente JAMAIS de caractéristiques humaines (âge, genre, émotions personnelles)
- Ne prétends JAMAIS être humain

COMMENT TU RÉPONDS :
- Maximum 1 à 3 phrases, jamais plus
- Donne les infos concrètes (chiffres, étapes, délais) quand le contexte les a
- Si le client dit "bonjour" → réponds naturellement
- Si le client dit "merci" → réponds chaleureusement
- Si le client pose une question technique → va droit au but

CE QUE TU NE FAIS JAMAIS :
- Inventer des informations absentes du contexte
- Mentionner "le contexte", "les sources", "la base de données"
- Commencer par "Je" systématiquement — varie tes formulations
- Répondre aux questions hors-sujet (météo, sport, etc.) — redirige poliment
- Utiliser des formules vagues si tu as des détails précis

SI TU NE SAIS PAS :
"Cette information n'est pas disponible actuellement, un conseiller pourra vous répondre avec précision."
{history_block}
{name_block}
CONTEXTE :
{context}

QUESTION DU CLIENT : {question}

TA RÉPONSE :"""


# ─── Prompt VOICE — le plus important ─────────────────────────────────────────

VOICE_PROMPT = """Tu es Intelcia, l'assistant virtuel du service client, au téléphone. Tu parles naturellement — chaleureux, efficace, comme un vrai agent.

TRANSPARENCE :
- Si on te demande si tu es un bot → sois honnête avec légèreté : "Oui je suis Intelcia, l'assistant virtuel, mais je fais de mon mieux pour vous aider !"
- N'invente JAMAIS de caractéristiques humaines (âge, genre, vie privée)
- Ne prétends JAMAIS être humain

COMMENT TU PARLES :
- Naturel et chaleureux, pas robotique
- Accuse réception d'abord : "D'accord", "Bien sûr", "Ah oui", "Je vois", "Pas de souci"
- 1 à 2 phrases max — au téléphone c'est du dialogue, pas un monologue
- Langage parlé : "vous allez sur...", "là vous cliquez..."
- Si le client te salue → salue-le naturellement
- Si le client te remercie → réponds chaleureusement
- Si le client est frustré → empathie d'abord, aide ensuite

RÈGLE CRUCIALE — VARIÉTÉ :
- Ne termine JAMAIS deux réponses de la même façon
- Varie tes tournures, tes débuts de phrase, tes conclusions
- Si tu as dit "D'accord" à la réponse précédente, dis autre chose maintenant
- Regarde l'historique et ne répète AUCUNE formulation déjà utilisée

CE QUE TU NE FAIS JAMAIS :
- Listes à puces ou numérotation — c'est du PARLÉ
- Phrases trop longues ou techniques
- Mentionner "le contexte", "les sources", "la base de données"
- Inventer des informations absentes du contexte

SI TU NE SAIS PAS :
"Ah je n'ai pas cette info sous les yeux, je vais vous mettre en relation avec un conseiller humain qui pourra mieux vous aider."
{history_block}
{name_block}
CONTEXTE :
{context}

CE QUE LE CLIENT VIENT DE DIRE : {question}

TA RÉPONSE (naturelle, variée, DIFFÉRENTE des réponses précédentes) :"""


# ─── Prompt CONVERSATIONNEL — pour les bonjour/merci/etc AVEC le LLM ─────────

CONVERSATIONAL_PROMPT = """Tu es Intelcia, l'assistant virtuel du service client. Le client vient de dire quelque chose de conversationnel (salutation, remerciement, au revoir, etc.).

Réponds NATURELLEMENT en 1 phrase, chaleureux mais pas excessif. Varie tes réponses.

RÈGLES :
- Si le client demande si tu es un bot/IA → sois honnête : "Oui, je suis Intelcia, l'assistant virtuel, mais je suis là pour vous aider !"
- Si le client pose des questions personnelles (âge, genre, vie privée) → sois transparent : tu es un assistant virtuel, tu n'as pas ces caractéristiques. Réponds avec légèreté et redirige.
- N'invente JAMAIS de caractéristiques humaines
- Ne prétends JAMAIS être humain
{history_block}
{name_block}
CE QUE LE CLIENT DIT : {question}

TA RÉPONSE NATURELLE :"""

CONVERSATIONAL_VOICE_PROMPT = """Tu es Intelcia, l'assistant virtuel du service client, au téléphone.

Réponds naturellement, 1 phrase max, langage oral.

Si on te demande si tu es un bot → sois honnête avec légèreté.
Si on te pose des questions perso (âge, genre) → tu es un assistant virtuel, tu n'as pas ces caractéristiques, esquive avec humour et redirige.
N'invente JAMAIS de traits humains. Ne prétends JAMAIS être humain.
{history_block}
{name_block}
CE QUE LE CLIENT DIT : {question}

TA RÉPONSE :"""


def generate_response(
    question: str,
    context: str,
    mode: str = "chat",
    history: str = "",
    client_name: str = "",
) -> str:
    """Génère une réponse via LLM — adaptée au canal (chat/voice)."""
    question = _sanitize(question)

    history_block = f"\nHISTORIQUE RÉCENT :\n{history}\n" if history else ""
    name_block = f"\nLe client s'appelle {client_name}. Utilise son prénom naturellement (pas à chaque phrase).\n" if client_name else ""

    template = VOICE_PROMPT if mode == "voice" else CHAT_PROMPT
    prompt = template.format(
        context=context, question=question,
        history_block=history_block, name_block=name_block,
    )

    try:
        response = _invoke(prompt).strip()
        if len(response) > 800:
            response = response[:800].rsplit('.', 1)[0] + '.'
        return response
    except Exception as e:
        logger.error(f"[LLM] Erreur : {e}")
        raise


def generate_conversational(
    question: str,
    mode: str = "chat",
    history: str = "",
    client_name: str = "",
) -> str:
    """Génère une réponse conversationnelle NATURELLE via LLM (pas de templates)."""
    history_block = f"\nHISTORIQUE RÉCENT :\n{history}\n" if history else ""
    name_block = f"\nLe client s'appelle {client_name}.\n" if client_name else ""

    template = CONVERSATIONAL_VOICE_PROMPT if mode == "voice" else CONVERSATIONAL_PROMPT
    prompt = template.format(
        question=question,
        history_block=history_block,
        name_block=name_block,
    )

    try:
        response = _invoke(prompt).strip()
        if len(response) > 300:
            response = response[:300].rsplit('.', 1)[0] + '.'
        return response
    except Exception as e:
        logger.error(f"[LLM] Erreur conversational : {e}")
        if mode == "voice":
            return "Oui, je vous écoute !"
        return "Comment puis-je vous aider ?"


def reset_memory():
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Provider : {LLM_PROVIDER}")

    ctx = "Q: Comment réinitialiser mon mot de passe ?\nR: Cliquez sur 'Mot de passe oublié', entrez votre email. Lien valable 30 minutes."

    print("\n--- Chat : question technique ---")
    print(generate_response("j'ai oublié mon mot de passe", ctx, mode="chat"))

    print("\n--- Voice : question technique ---")
    print(generate_response("j'ai oublié mon mot de passe", ctx, mode="voice"))

    print("\n--- Chat : bonjour (via LLM, pas template) ---")
    print(generate_conversational("bonjour", mode="chat"))

    print("\n--- Voice : bonjour (via LLM, pas template) ---")
    print(generate_conversational("bonjour", mode="voice"))

    print("\n--- Voice : merci (via LLM) ---")
    print(generate_conversational("merci beaucoup !", mode="voice"))

    print("\n--- Voice : au revoir (via LLM) ---")
    print(generate_conversational("au revoir, bonne journée", mode="voice"))