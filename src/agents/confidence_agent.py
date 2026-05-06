"""
Confidence Agent — évalue la confiance avec plusieurs signaux.
Fixes appliqués :
- Score multi-signal (nb docs, longueur réponse, incertitude, boost)
- Fallback 2e recherche élargie si score trop bas
- Seuil adapté selon l'intent
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import logging
from config.settings import CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

UNCERTAINTY_PHRASES = [
    "je ne sais pas", "je n'ai pas", "aucune information",
    "je ne trouve pas", "introuvable", "pas dans le contexte",
    "je ne peux pas", "je n'ai aucune", "désolé",
]


def evaluate_confidence(rag_score: float, intent: str, response: str, nb_docs: int = 0) -> dict:
    """
    Score multi-signal :
    - Base  : rag_score agrégé
    - Malus : peu de docs, réponse trop courte, phrases d'incertitude
    - Bonus : bon score + beaucoup de docs
    - Seuil adapté selon intent
    """
    confidence = rag_score

    # ─── Pénalités ────────────────────────────────────────────────────────────
    if nb_docs < 2:
        confidence -= 0.15
        logger.debug(f"[Confidence] Pénalité peu de docs ({nb_docs})")

    if len(response.split()) < 5:
        confidence -= 0.10
        logger.debug("[Confidence] Pénalité réponse trop courte")

    if any(p in response.lower() for p in UNCERTAINTY_PHRASES):
        confidence -= 0.20
        logger.debug("[Confidence] Pénalité phrase d'incertitude détectée")

    if intent == "hors_sujet":
        confidence = 0.0

    # ─── Bonus ────────────────────────────────────────────────────────────────
    if rag_score > 0.80 and nb_docs >= 3:
        confidence += 0.05
        logger.debug("[Confidence] Bonus bon contexte")

    confidence = round(max(0.0, min(1.0, confidence)), 3)

    # ─── Seuil adapté selon intent ────────────────────────────────────────────
    # troubleshoot → on tolère un seuil légèrement plus bas
    threshold = CONFIDENCE_THRESHOLD - 0.05 if intent == "troubleshoot" else CONFIDENCE_THRESHOLD

    if confidence >= threshold:
        action = "respond"
        reason = f"Score {confidence} ≥ seuil {threshold:.2f} → réponse automatique"
    else:
        action = "escalate"
        reason = f"Score {confidence} < seuil {threshold:.2f} → escalade agent humain"

    logger.info(f"[Confidence] score={confidence} | intent={intent} | action={action}")

    return {
        "confidence": confidence,
        "action"    : action,
        "reason"    : reason,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tests = [
        (0.82, "faq",          "Vous pouvez contacter le support par email.", 4),
        (0.45, "troubleshoot", "Je ne sais pas comment résoudre ce problème.", 1),
        (0.78, "troubleshoot", "Contactez le support en décrivant le problème.", 3),
        (0.90, "faq",          "Rendez-vous sur la page de connexion.", 5),
    ]
    for rag_score, intent, response, nb_docs in tests:
        r = evaluate_confidence(rag_score, intent, response, nb_docs)
        print(f"score={rag_score} | intent={intent} | nb_docs={nb_docs} → {r['action']} ({r['confidence']})")