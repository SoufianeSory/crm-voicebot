"""
Rewriter Agent — reformule la question utilisateur pour mieux matcher le FAQ.
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


REWRITER_PROMPT = """Tu es un assistant qui reformule les messages des clients en questions \
claires et formelles pour une recherche dans une FAQ de support client.

Règles :
- Reformule en une seule question claire et formelle
- Détecte l'intention réelle même si le message est indirect ou conversationnel
- Retourne UNIQUEMENT la question reformulée, rien d'autre
- Si le message contient plusieurs questions, garde la plus importante

Exemples de reformulation :
- "j'ai un problème avec mon appli" → "Comment signaler un problème technique avec l'application ?"
- "ça marche pas" → "Comment signaler un problème technique avec l'application ?"
- "mot de passe oublié" → "Comment réinitialiser mon mot de passe ?"
- "j'arrive pas à me connecter" → "Pourquoi je n'arrive pas à me connecter à mon compte ?"
- "parler à quelqu'un" → "Puis-je parler à un agent humain ?"
- "agent humain" → "Puis-je parler à un agent humain ?"
- "je veux un humain" → "Puis-je parler à un agent humain ?"
- "horaires" → "Quels sont vos horaires d'ouverture ?"
- "vous êtes disponibles quand ?" → "Quels sont vos horaires d'ouverture ?"
- "je bosse la nuit, quelqu'un disponible à 3h ?" → "Quels sont vos horaires d'ouverture ?"
- "vous êtes ouverts le weekend ?" → "Quels sont vos horaires d'ouverture ?"
- "changer mon numéro" → "Comment modifier les informations de mon profil ?"
- "mettre à jour mon email" → "Comment modifier les informations de mon profil ?"
- "modifier mon profil" → "Comment modifier les informations de mon profil ?"
- "pas eu de réponse" → "Combien de temps faut-il pour recevoir une réponse du support ?"
- "où en est mon ticket" → "Puis-je suivre l'état de ma demande de support ?"
- "suivre ma demande" → "Puis-je suivre l'état de ma demande de support ?"
- "aide en ligne" → "Où puis-je trouver la documentation d'aide ?"
- "tutoriel" → "Où puis-je trouver la documentation d'aide ?"
- "contacter support" → "Comment contacter le support client ?"
- "numéro de téléphone" → "Comment contacter le support client ?"

Message client : {question}
Question reformulée :"""


def rewrite_question(question: str) -> str:
    """Reformule la question pour améliorer la recherche RAG."""
    prompt = REWRITER_PROMPT.format(question=question)
    rewritten = _invoke(prompt).strip()

    # Nettoyer les guillemets si le LLM les ajoute
    rewritten = rewritten.strip('"').strip("'")

    # Sécurité : si réponse trop longue ou contient plusieurs lignes → garder l'originale
    if len(rewritten) > 200 or "\n" in rewritten:
        return question

    return rewritten


if __name__ == "__main__":
    tests = [
        "j'ai un problème avec mon appli",
        "je bosse la nuit, quelqu'un disponible à 3h du matin ?",
        "j'ai changé de numéro de téléphone",
        "j'ai envoyé un message hier, toujours pas de réponse",
        "je veux parler à un vrai humain",
        "vous êtes ouverts le samedi ?",
        "comment je change mon adresse email ?",
        "où trouver de l'aide ?",
        "ça fait 2 jours que j'attends",
    ]
    for q in tests:
        print(f"Original  : {q}")
        print(f"Reformulé : {rewrite_question(q)}\n")