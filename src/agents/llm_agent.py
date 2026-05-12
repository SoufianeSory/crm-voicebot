"""
LLM Agent v2 — Prompt anti-vague renforcé.
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from config.settings import LLM_PROVIDER, OLLAMA_BASE_URL, LLM_MODEL, GROQ_API_KEY, GROQ_MODEL

if LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.2, api_key=GROQ_API_KEY)
    def _invoke(prompt: str) -> str:
        return llm.invoke(prompt).content
else:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
    def _invoke(prompt: str) -> str:
        return llm.invoke(prompt)


SYSTEM_PROMPT = """Tu es Alex, l'assistant virtuel du service client. Tu es direct, précis et bienveillant.

RÈGLES ABSOLUES :
1. Réponds en 1 à 3 phrases MAXIMUM
2. Utilise UNIQUEMENT les informations du contexte ci-dessous
3. Donne les détails concrets (chiffres, étapes, délais) si disponibles
4. Ne dis JAMAIS des formules vagues comme "nous pouvons vous aider" si le contexte donne des détails précis
5. Ne commence jamais par "Je" — varie les formulations
6. Ne mentionne jamais "le contexte", "les sources", "les documents"
7. Si la réponse est absente du contexte : "Cette information n'est pas disponible ici — un conseiller pourra vous répondre avec précision."

CONTEXTE :
{context}

QUESTION DU CLIENT : {question}

RÉPONSE DIRECTE ET PRÉCISE :"""


def generate_response(question: str, context: str) -> str:
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    return _invoke(prompt).strip()


def reset_memory():
    pass


if __name__ == "__main__":
    print(f"Provider : {LLM_PROVIDER}")
    ctx = "Question : Comment réinitialiser mon mot de passe ?\nRéponse : Cliquez sur 'Mot de passe oublié', entrez votre email. Lien valable 30 minutes."
    print(generate_response("j'ai oublié mon mot de passe", ctx))