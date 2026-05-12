"""
STT Module v2 — Groq Whisper API + fallback Faster-Whisper local.
"""

import io
import logging
import os
import wave
import numpy as np

logger = logging.getLogger(__name__)

MODEL_SIZE   = "base"
MIN_DURATION = 0.8

STT_PROVIDER = os.getenv("STT_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from faster_whisper import WhisperModel
        logger.info(f"[STT] Chargement Faster-Whisper '{MODEL_SIZE}'...")
        _local_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        logger.info("[STT] Faster-Whisper chargé ✅")
    return _local_model


def preload():
    if STT_PROVIDER == "local":
        _get_local_model()
    else:
        logger.info(f"[STT] Provider={STT_PROVIDER} — pas de préchargement local")


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _transcribe_groq(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    import httpx
    wav_bytes = _pcm_to_wav(audio_bytes, sample_rate)
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model": "whisper-large-v3-turbo", "language": "fr", "response_format": "text"},
            )
            response.raise_for_status()
            text = response.text.strip()
            logger.info(f"[STT/Groq] '{text}'")
            return text
    except httpx.HTTPStatusError as e:
        logger.warning(f"[STT/Groq] Erreur HTTP {e.response.status_code} → fallback local")
        return _transcribe_local(audio_bytes, sample_rate)
    except Exception as e:
        logger.warning(f"[STT/Groq] Erreur réseau ({e}) → fallback local")
        return _transcribe_local(audio_bytes, sample_rate)


def _transcribe_local(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    model    = _get_local_model()
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, info = model.transcribe(audio_np, language="fr", beam_size=5,
                                      vad_filter=False, word_timestamps=False)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    logger.info(f"[STT/Local] '{text}' (lang={info.language}, prob={info.language_probability:.2f})")
    return text


def transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    duration = len(audio_bytes) / (sample_rate * 2)
    if duration < MIN_DURATION:
        logger.debug(f"[STT] Ignoré : {duration:.2f}s < {MIN_DURATION}s")
        return ""
    logger.info(f"[STT] Transcription {duration:.2f}s via {STT_PROVIDER.upper()}")
    if STT_PROVIDER == "groq" and GROQ_API_KEY:
        return _transcribe_groq(audio_bytes, sample_rate)
    else:
        if STT_PROVIDER == "groq" and not GROQ_API_KEY:
            logger.warning("[STT] GROQ_API_KEY manquant → fallback local")
        return _transcribe_local(audio_bytes, sample_rate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preload()
    print(f"STT v2 prêt (provider={STT_PROVIDER}) ✅")