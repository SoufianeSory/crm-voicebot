"""
Session Memory v2 — Thread-safe pour multi-sessions concurrentes.
"""

import re
import time
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config.settings import SESSION_MAX_TURNS

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    question : str
    response : str
    intent   : str
    action   : str
    ts       : str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class SessionContext:
    session_id    : str
    turns         : deque = field(default_factory=lambda: deque(maxlen=SESSION_MAX_TURNS))
    client_name   : Optional[str] = None
    is_first_turn : bool = True
    turn_count    : int = 0
    created_at    : float = field(default_factory=time.time)
    last_activity : float = field(default_factory=time.time)


_sessions: dict[str, SessionContext] = {}
_lock = threading.Lock()
SESSION_TIMEOUT_SEC = 7200


def get_session(session_id: str) -> SessionContext:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = SessionContext(session_id=session_id)
            logger.info(f"[Memory] Nouvelle session '{session_id}'")
            _cleanup()
        return _sessions[session_id]


def add_turn(session_id: str, question: str, response: str,
             intent: str = "faq", action: str = "respond"):
    ctx = get_session(session_id)
    with _lock:
        ctx.turns.append(Turn(question=question, response=response, intent=intent, action=action))
        ctx.turn_count += 1
        ctx.last_activity = time.time()
    if ctx.client_name is None:
        name = _extract_name(question)
        if name:
            ctx.client_name = name
            logger.info(f"[Memory] Prénom : '{name}' (session={session_id})")


def mark_first_turn_done(session_id: str):
    get_session(session_id).is_first_turn = False


def get_history_text(session_id: str, max_turns: int = 3) -> str:
    ctx = get_session(session_id)
    if not ctx.turns:
        return ""
    with _lock:
        recent = list(ctx.turns)[-max_turns:]
    lines = []
    for t in recent:
        lines.append(f"Client : {t.question}")
        lines.append(f"Intelcia   : {t.response}")
    return "\n".join(lines)


def get_client_name(session_id: str) -> Optional[str]:
    return get_session(session_id).client_name


def is_first_turn(session_id: str) -> bool:
    return get_session(session_id).is_first_turn


def get_all_sessions_stats() -> dict:
    with _lock:
        return {
            "active": len(_sessions),
            "total_turns": sum(s.turn_count for s in _sessions.values()),
        }


def reset_session(session_id: str):
    with _lock:
        _sessions.pop(session_id, None)


def reset_all():
    with _lock:
        _sessions.clear()


def _cleanup():
    now = time.time()
    expired = [sid for sid, ctx in _sessions.items() if now - ctx.last_activity > SESSION_TIMEOUT_SEC]
    for sid in expired:
        del _sessions[sid]
    if expired:
        logger.info(f"[Memory] {len(expired)} sessions expirées nettoyées")


_NAME_PATTERNS = [
    # UNIQUEMENT les formulations explicites d'introduction
    r"(?:je m'appelle|mon prénom est|moi c'est|je me nomme|on m'appelle)\s+([A-ZÀ-Ÿa-zà-ÿ][a-zà-ÿ\-]{2,20})",
]

# Tous les mots courants que le STT peut transcrire avec une majuscule
# ou qui suivent "moi c'est" sans être un prénom
_NOT_NAMES = {
    # Mots courants
    "bonjour", "bonsoir", "merci", "salut", "hello", "bien", "oui", "non",
    "super", "parfait", "erreur", "comment", "quoi", "combien", "pourquoi",
    "alors", "voici", "juste", "tout", "rien", "encore", "aussi", "comme",
    "avec", "pour", "dans", "mais", "donc", "possible", "pas", "bon",
    "vrai", "faux", "urgent", "simple", "normal", "pareil", "certain",
    # Pronoms et déterminants
    "lui", "elle", "eux", "nous", "vous", "moi", "toi", "que", "qui",
    # Mots que le STT confond souvent
    "bot", "botte", "client", "agent", "ticket", "support", "aide",
    "problème", "question", "réponse", "info", "compte", "profil",
    "email", "mot", "passe", "page", "site", "appli", "application",
    # Adjectifs courants
    "petit", "grand", "nouveau", "ancien", "premier", "dernier",
    # Exclamations / interjections que le STT capture
    "euh", "hein", "ouais", "okay", "allez", "allô", "allo",
    "voilà", "genre", "quand", "très",
}

def _extract_name(text: str) -> Optional[str]:
    """
    Extraction de prénom STRICTE.
    Ne matche QUE les introductions explicites : "je m'appelle X", "moi c'est X".
    Rejette tout mot courant via blacklist élargie.
    """
    for pattern in _NAME_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().capitalize()
            # Rejeter si trop court, dans la blacklist, ou tout en minuscules avec < 3 lettres
            if len(name) < 3:
                continue
            if name.lower() in _NOT_NAMES:
                continue
            # Rejeter si c'est un mot qui finit par un suffixe verbal courant
            if name.lower().endswith(("er", "ir", "re", "ent", "ons", "ez", "ais", "ait")):
                continue
            logger.info(f"[Memory] Prénom candidat accepté : '{name}'")
            return name
    return None