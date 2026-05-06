"""
STT Module — Transcription audio via Faster-Whisper.
- Modèle préchargé au démarrage (élimine le cold start de ~2s)
- Filtre durée minimale (élimine les faux déclenchements VAD courts)
"""

import logging
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_SIZE   = "base"
MIN_DURATION = 0.8   # secondes — segments plus courts = bruit/faux déclenchement

# ─── Singleton ────────────────────────────────────────────────────────────────

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    """Retourne le modèle Whisper (singleton)."""
    global _model
    if _model is None:
        logger.info(f"[STT] Chargement du modèle Whisper '{MODEL_SIZE}'...")
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        logger.info("[STT] Modèle chargé ✅")
    return _model


def preload():
    """
    Précharge le modèle Whisper au démarrage du serveur.
    Appeler cette fonction dans le lifespan FastAPI pour éviter
    le cold start sur la première requête (~2s de délai supplémentaire).
    """
    get_model()


def transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    Transcrit un buffer audio PCM int16 en texte français.

    Args:
        audio_bytes : audio brut PCM 16kHz mono int16 (bytes)
        sample_rate : fréquence d'échantillonnage

    Returns:
        Texte transcrit, ou "" si segment trop court / rien détecté.
    """
    # ─── Filtre durée minimale ─────────────────────────────────────────────────
    duration = len(audio_bytes) / (sample_rate * 2)  # 2 bytes par sample int16
    if duration < MIN_DURATION:
        logger.debug(f"[STT] Segment ignoré : {duration:.2f}s < {MIN_DURATION}s (trop court)")
        return ""

    model = get_model()

    # Convertir bytes → numpy float32 normalisé [-1, 1]
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    segments, info = model.transcribe(
        audio_np,
        language="fr",
        beam_size=5,
        vad_filter=False,   # VAD géré par Silero en amont
        word_timestamps=False,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()

    if text:
        logger.info(f"[STT] Transcription ({duration:.2f}s) : '{text}' "
                    f"(lang={info.language}, prob={info.language_probability:.2f})")
    else:
        logger.debug(f"[STT] Aucun texte détecté ({duration:.2f}s)")

    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    if len(sys.argv) > 1:
        data, sr = sf.read(sys.argv[1], dtype="int16")
        result = transcribe(data.tobytes(), sr)
        print(f"Résultat : {result!r}")
    else:
        # Test préchargement
        preload()
        print("Préchargement OK ✅")