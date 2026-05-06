"""
TTS Module — Synthèse vocale via edge-tts (Microsoft, gratuit, cloud).
Latence : ~0.5s vs ~9s pour Kokoro CPU.
Voix française : fr-FR-DeniseNeural (féminine, naturelle).
"""

import asyncio
import io
import logging
import tempfile
import os

import edge_tts
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

VOICE       = "fr-FR-DeniseNeural"   # meilleure voix française edge-tts
RATE        = "+0%"                   # vitesse normale
VOLUME      = "+0%"

# Voix alternatives disponibles (toutes gratuites) :
# "fr-FR-HenriNeural"      → masculine
# "fr-FR-EloiseNeural"     → féminine enfant
# "fr-BE-GerardNeural"     → accent belge
# "fr-CA-SylvieNeural"     → accent québécois


def synthesize(text: str) -> bytes:
    """
    Synthétise un texte en audio WAV (bytes) via edge-tts.

    Args:
        text : texte à synthétiser

    Returns:
        Audio WAV en bytes — prêt à envoyer via WebSocket.
    """
    if not text or not text.strip():
        logger.warning("[TTS] Texte vide reçu")
        return b""

    # Tronquer à 2 phrases max pour le voicebot (réponse concise + TTS rapide)
    text = _truncate_to_sentences(text, max_sentences=2)

    logger.info(f"[TTS] Synthèse edge-tts : '{text[:80]}{'...' if len(text) > 80 else ''}'")

    # edge-tts est async — on l'exécute dans une boucle event
    try:
        audio_bytes = asyncio.run(_synthesize_async(text))
    except RuntimeError:
        # Si on est déjà dans une boucle async (ex: FastAPI), on crée une nouvelle
        loop = asyncio.new_event_loop()
        try:
            audio_bytes = loop.run_until_complete(_synthesize_async(text))
        finally:
            loop.close()

    logger.info(f"[TTS] Audio généré : {len(audio_bytes)} bytes")
    return audio_bytes


async def _synthesize_async(text: str) -> bytes:
    """Synthèse async edge-tts → retourne bytes WAV."""
    communicate = edge_tts.Communicate(text, voice=VOICE, rate=RATE, volume=VOLUME)

    # edge-tts génère du MP3 — on collecte les chunks
    mp3_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_chunks.append(chunk["data"])

    if not mp3_chunks:
        raise RuntimeError("edge-tts n'a retourné aucun audio")

    mp3_bytes = b"".join(mp3_chunks)

    # Convertir MP3 → WAV en mémoire (le navigateur accepte les deux,
    # mais WAV est plus simple pour AudioContext.decodeAudioData)
    wav_bytes = _mp3_to_wav(mp3_bytes)
    return wav_bytes


def _mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """Convertit MP3 bytes → WAV bytes via soundfile + scipy."""
    try:
        # Méthode 1 : via fichier temporaire (compatible Windows)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_mp3.write(mp3_bytes)
            tmp_mp3_path = tmp_mp3.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name

        # Utiliser ffmpeg via soundfile ou pydub si disponible
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(tmp_mp3_path)
            audio.export(tmp_wav_path, format="wav")
            with open(tmp_wav_path, "rb") as f:
                wav_bytes = f.read()
        except ImportError:
            # Fallback : retourner le MP3 directement
            # (AudioContext.decodeAudioData supporte MP3 nativement dans Chrome)
            logger.debug("[TTS] pydub non disponible — envoi MP3 direct (compatible Chrome)")
            wav_bytes = mp3_bytes

        return wav_bytes

    finally:
        # Nettoyage fichiers temporaires
        for path in [tmp_mp3_path, tmp_wav_path]:
            try:
                os.unlink(path)
            except Exception:
                pass


def _truncate_to_sentences(text: str, max_sentences: int = 2) -> str:
    """
    Tronque le texte à N phrases max.
    Utile pour le voicebot : réponses courtes = TTS plus rapide.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    truncated = " ".join(sentences[:max_sentences])
    if len(sentences) > max_sentences:
        logger.debug(f"[TTS] Texte tronqué : {len(sentences)} → {max_sentences} phrases")
    return truncated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_text = (
        "Pour réinitialiser votre mot de passe, rendez-vous sur la page de connexion "
        "et cliquez sur 'Mot de passe oublié'. Un email vous sera envoyé avec les instructions."
    )

    result = synthesize(test_text)

    # Sauvegarder pour écouter
    ext = "wav" if result[:4] == b"RIFF" else "mp3"
    output_path = f"test_edge_tts.{ext}"
    with open(output_path, "wb") as f:
        f.write(result)

    print(f"Audio généré → {output_path} ({len(result)} bytes) ✅")