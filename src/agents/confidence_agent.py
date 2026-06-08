"""
Confidence Agent v2 — Scoring multi-signal.
"""

import sys, os, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import logging
from config.settings import CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

UNCERTAINTY_PHRASES = [
    "je ne sais pas", "je n'ai pas", "aucune information",
    "je ne trouve pas", "introuvable", "pas dans le contexte",
    "je ne peux pas", "je n'ai aucune", "désolé",
    "pas disponible actuellement", "un conseiller pourra",
    "je ne dispose pas", "pas en mesure",
    "impossible de répondre", "sous les yeux",
]

CONCRETE_PATTERNS = [r'\d+', r'cliquez sur', r'rendez-vous', r'étape']


def evaluate_confidence(rag_score: float, intent: str, response: str, nb_docs: int = 0) -> dict:
    confidence = rag_score
    adjustments = []

    if nb_docs < 2:
        confidence -= 0.08
        adjustments.append(f"docs={nb_docs} → -0.08")

    if len(response.split()) < 5:
        confidence -= 0.10
        adjustments.append("courte → -0.10")

    resp_lower = response.lower()
    if any(p in resp_lower for p in UNCERTAINTY_PHRASES):
        confidence -= 0.20
        adjustments.append("incertitude → -0.20")

    if intent == "hors_sujet":
        confidence = 0.0

    if rag_score > 0.80 and nb_docs >= 3:
        confidence += 0.05
        adjustments.append("bon contexte → +0.05")

    if len(response.split()) >= 10 and not any(p in resp_lower for p in UNCERTAINTY_PHRASES):
        if sum(1 for p in CONCRETE_PATTERNS if re.search(p, resp_lower)) >= 2:
            confidence += 0.03
            adjustments.append("concret → +0.03")

    confidence = round(max(0.0, min(1.0, confidence)), 3)
    threshold = CONFIDENCE_THRESHOLD - 0.05 if intent == "troubleshoot" else CONFIDENCE_THRESHOLD

    action = "respond" if confidence >= threshold else "escalate"
    reason = f"Score {confidence} {'≥' if action == 'respond' else '<'} seuil {threshold:.2f}"

    logger.info(f"[Confidence] {confidence} | {intent} | {action} | {adjustments}")
    return {"confidence": confidence, "action": action, "reason": reason}
