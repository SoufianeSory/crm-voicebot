"""
LLM Agent — génère une réponse à partir du contexte RAG + question.
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from config.settings import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL, LLM_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
)

# ─── Choix du LLM ─────────────────────────────────────────────────────────────

if LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.3,
        api_key=GROQ_API_KEY,
    )

    def _invoke(prompt: str) -> str:
        return llm.invoke(prompt).content

else:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
    )

    def _invoke(prompt: str) -> str:
        return llm.invoke(prompt)


# ─── Prompt système ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es Alex, l'assistant virtuel du service client. Tu es professionnel, bienveillant et efficace.

Règles de réponse :
- Réponds en 2-3 phrases maximum, de façon claire et naturelle
- Utilise UNIQUEMENT le contexte fourni ci-dessous pour répondre
- Adopte un ton chaleureux mais professionnel — tu parles à un client, pas à une machine
- Ne mentionne jamais "Source 1", "Source 2", "le contexte" etc.
- Ne commence jamais ta réponse par "Je" — varie les formulations
- Si la réponse n'est pas dans le contexte, dis : "Cette information n'est pas disponible directement, mais un de nos conseillers pourra vous répondre avec précision."

Contexte :
{context}

Question du client : {question}

Réponse (courte, naturelle, basée uniquement sur le contexte) :"""


def generate_response(question: str, context: str) -> str:
    """Génère une réponse en utilisant uniquement le contexte RAG."""
    prompt = SYSTEM_PROMPT.format(
        context=context,
        question=question,
    )
    return _invoke(prompt).strip()


def reset_memory():
    """Conservé pour compatibilité — mémoire désactivée (pas de contamination cross-session)."""
    pass


if __name__ == "__main__":
    print(f"Provider actif : {LLM_PROVIDER}")
    contexte = "Pour réinitialiser votre mot de passe, rendez-vous sur la page de connexion et cliquez sur 'Mot de passe oublié'. Un email vous sera envoyé avec les instructions."
    tests = [
        "Comment je réinitialise mon mot de passe ?",
        "comment changer mon mot de passe",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        print(f"R: {generate_response(q, contexte)}")