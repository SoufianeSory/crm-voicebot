"""
VAD Module — Détection d'activité vocale via Silero VAD.
Version stable production (anti-crash + filtrage silence + compatibilité Silero).
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ─── Paramètres VAD ───────────────────────────────────────────────────────────

SAMPLE_RATE      = 16000
FRAME_SAMPLES    = 512
FRAME_MS         = int(1000 * FRAME_SAMPLES / SAMPLE_RATE)  # ≈ 32 ms                      # 🔥 60ms (stable pour Silero)
SPEECH_THRESHOLD = 0.5
SILENCE_FRAMES   = 10                      # ~600ms silence → fin segment

MIN_ENERGY       = 0.001                   # 🔥 filtre silence (important)

# ─── Singleton modèle ─────────────────────────────────────────────────────────

_vad_model = None


def get_vad():
    global _vad_model
    if _vad_model is None:
        logger.info("[VAD] Chargement du modèle Silero VAD...")
        _vad_model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True
        )
        logger.info("[VAD] Modèle Silero chargé ✅")
    return _vad_model


# ─── Classe VAD ───────────────────────────────────────────────────────────────

class VoiceActivityDetector:

    def __init__(self):
        self.model = get_vad()
        self.speech_buffer: list[bytes] = []
        self.silence_count = 0
        self.is_speaking = False

    def process_frame(self, frame_bytes: bytes) -> bytes | None:

        # 🔥 Sécurité taille minimale
        if len(frame_bytes) < FRAME_SAMPLES * 2:
            return None

        # Convertir en float32
        audio_np = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # 🔥 sécurité supplémentaire
        if len(audio_np) != FRAME_SAMPLES:
            return None

        # 🔥 filtre silence (CRUCIAL)
        if np.abs(audio_np).mean() < MIN_ENERGY:
            return None

        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        # 🔥 protéger contre crash Silero
        try:
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, SAMPLE_RATE)
                prob = float(speech_prob.item())
        except Exception as e:
            logger.warning(f"[VAD] Frame ignorée (erreur): {e}")
            return None

        # ─── Logique détection ─────────────────────────────

        if prob >= SPEECH_THRESHOLD:
            self.is_speaking = True
            self.silence_count = 0
            self.speech_buffer.append(frame_bytes)

        elif self.is_speaking:
            self.speech_buffer.append(frame_bytes)
            self.silence_count += 1

            if self.silence_count >= SILENCE_FRAMES:
                complete_audio = b"".join(self.speech_buffer)
                self._reset()

                duration = len(complete_audio) / (SAMPLE_RATE * 2)
                logger.info(f"[VAD] Segment détecté : {duration:.2f}s")

                return complete_audio

        return None

    def _reset(self):
        self.speech_buffer = []
        self.silence_count = 0
        self.is_speaking = False

    def flush(self) -> bytes | None:
        if self.speech_buffer:
            complete_audio = b"".join(self.speech_buffer)
            self._reset()
            return complete_audio
        return None


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_vad()
    print("Silero VAD OK ✅")