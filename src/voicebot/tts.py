"""
TTS Module v3 — edge-tts avec streaming + config depuis settings.
Deux modes :
    synthesize()         → retourne tout l'audio d'un coup (compat legacy)
    synthesize_stream()  → yield les chunks audio dès qu'ils arrivent (voicebot)
"""

import asyncio
import logging
import re
from typing import AsyncGenerator

import edge_tts

from config.settings import TTS_VOICE, TTS_RATE

logger = logging.getLogger(__name__)

VOICE  = TTS_VOICE
RATE   = TTS_RATE
VOLUME = "+0%"


async def synthesize(text: str) -> bytes:
    """Synthèse complète — retourne tout l'audio MP3 d'un coup."""
    if not text or not text.strip():
        logger.warning("[TTS] Texte vide")
        return b""
    text = _clean_for_speech(text)
    logger.info(f"[TTS] Synthèse : '{text[:80]}{'...' if len(text) > 80 else ''}'")
    mp3_bytes = await _synthesize_mp3(text)
    logger.info(f"[TTS] Audio généré : {len(mp3_bytes)} bytes")
    return mp3_bytes


async def synthesize_stream(text: str) -> AsyncGenerator[bytes, None]:
    """
    Synthèse en streaming — yield les chunks audio au fur et à mesure.
    Le premier chunk arrive en ~200ms au lieu d'attendre la fin complète.
    Idéal pour le voicebot : le client entend la réponse plus tôt.
    """
    if not text or not text.strip():
        return
    text = _clean_for_speech(text)
    logger.info(f"[TTS/Stream] Synthèse : '{text[:80]}{'...' if len(text) > 80 else ''}'")

    communicate = edge_tts.Communicate(text, voice=VOICE, rate=RATE, volume=VOLUME)
    chunk_count = 0
    async for chunk in communicate.stream():
        if chunk["type"] == "audio" and chunk["data"]:
            chunk_count += 1
            yield chunk["data"]

    logger.info(f"[TTS/Stream] {chunk_count} chunks envoyés")


async def _synthesize_mp3(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice=VOICE, rate=RATE, volume=VOLUME)
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    if not chunks:
        raise RuntimeError("[TTS] edge-tts n'a retourné aucun audio")
    return b"".join(chunks)


def _clean_for_speech(text: str) -> str:
    """
    Nettoie le texte pour la synthèse vocale :
    - Supprime le markdown, les listes à puces, les numérotations
    - Limite à 3 phrases max (oral = concis)
    - Remplace les abréviations courantes
    """
    # Supprimer markdown
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.+?)\*', r'\1', text)       # *italic*
    text = re.sub(r'#+\s*', '', text)               # # headers
    text = re.sub(r'[-•]\s+', '', text)             # - bullets
    text = re.sub(r'\d+\.\s+', '', text)            # 1. numbered

    # Limiter à 3 phrases
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    text = " ".join(sentences[:3])

    # Nettoyer espaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ─── Audio de remplissage pré-généré ──────────────────────────────────────────

_FILLER_CACHE: dict[str, bytes] = {}

async def get_filler_audio(text: str = "Un instant, je vérifie...") -> bytes:
    """
    Retourne un audio de remplissage pré-généré.
    Utilisé quand le traitement prend trop de temps (> 1.5s).
    Le résultat est caché pour les appels suivants.
    """
    if text in _FILLER_CACHE:
        return _FILLER_CACHE[text]

    audio = await synthesize(text)
    _FILLER_CACHE[text] = audio
    return audio


async def preload_fillers():
    """Pré-génère les audios de remplissage au démarrage."""
    fillers = [
        "Un instant, je vérifie.",
        "Je regarde ça tout de suite.",
        "Bien sûr, un petit moment.",
    ]
    for text in fillers:
        await get_filler_audio(text)
    logger.info(f"[TTS] {len(fillers)} fillers pré-générés ✅")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def test():
        # Test classique
        result = await synthesize("Bonjour, je suis votre assistant vocal.")
        print(f"Synthèse complète : {len(result)} bytes")

        # Test streaming
        total = 0
        first_chunk = True
        import time
        start = time.time()
        async for chunk in synthesize_stream("Bonjour, comment puis-je vous aider aujourd'hui ?"):
            if first_chunk:
                print(f"Premier chunk reçu en {time.time() - start:.2f}s")
                first_chunk = False
            total += len(chunk)
        print(f"Streaming total : {total} bytes en {time.time() - start:.2f}s")

    asyncio.run(test())