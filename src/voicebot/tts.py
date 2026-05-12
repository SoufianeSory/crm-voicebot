"""
TTS Module v2 — edge-tts async natif.
"""

import asyncio
import logging
import re
import edge_tts

logger = logging.getLogger(__name__)

VOICE  = "fr-FR-DeniseNeural"
RATE   = "+0%"
VOLUME = "+0%"


async def synthesize(text: str) -> bytes:
    if not text or not text.strip():
        logger.warning("[TTS] Texte vide")
        return b""
    text = _truncate_to_sentences(text, max_sentences=2)
    logger.info(f"[TTS] Synthèse : '{text[:80]}{'...' if len(text) > 80 else ''}'")
    mp3_bytes = await _synthesize_mp3(text)
    logger.info(f"[TTS] Audio généré : {len(mp3_bytes)} bytes")
    return mp3_bytes


async def _synthesize_mp3(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice=VOICE, rate=RATE, volume=VOLUME)
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    if not chunks:
        raise RuntimeError("[TTS] edge-tts n'a retourné aucun audio")
    return b"".join(chunks)


def _truncate_to_sentences(text: str, max_sentences: int = 2) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:max_sentences])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(synthesize("Bonjour, je suis votre assistant vocal."))
    with open("test_tts.mp3", "wb") as f:
        f.write(result)
    print(f"Audio généré ✅ ({len(result)} bytes)")