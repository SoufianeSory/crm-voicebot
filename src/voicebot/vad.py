"""
VAD Module — Détection d'activité vocale via Silero VAD.
Fix bruit de fond : seuil de silence abaissé à 0.3 (au lieu de 0.5)
pour gérer les micros avec bruit de ventilateur.
"""

import logging
import collections
import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE       = 16000
FRAME_SAMPLES     = 512
FRAME_MS          = int(1000 * FRAME_SAMPLES / SAMPLE_RATE)  # ≈ 32ms
SPEECH_THRESHOLD  = 0.5   # prob min pour détecter la parole
SILENCE_THRESHOLD = 0.3   # prob max pour compter comme silence
SILENCE_FRAMES    = 15    # ~480ms de silence → fin segment
PRE_BUFFER_FRAMES = 3

_vad_model = None


def get_vad():
    global _vad_model
    if _vad_model is None:
        logger.info("[VAD] Chargement du modèle Silero VAD...")
        _vad_model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        logger.info("[VAD] Modèle Silero chargé ✅")
    return _vad_model


class VoiceActivityDetector:

    def __init__(self):
        self.model         = get_vad()
        self.speech_buffer : list[bytes] = []
        self.pre_buffer    = collections.deque(maxlen=PRE_BUFFER_FRAMES)
        self.silence_count = 0
        self.is_speaking   = False

    def process_frame(self, frame_bytes: bytes) -> bytes | None:

        if len(frame_bytes) < FRAME_SAMPLES * 2:
            return None

        audio_np = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) != FRAME_SAMPLES:
            return None

        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        try:
            with torch.no_grad():
                prob = float(self.model(audio_tensor, SAMPLE_RATE).item())
        except Exception as e:
            logger.warning(f"[VAD] Frame ignorée : {e}")
            return None

        energy = np.abs(audio_np).mean()
        logger.debug(f"[VAD] prob={prob:.3f} energy={energy:.4f} speaking={self.is_speaking}")

        # ── Début de parole ──────────────────────────────────────────────────
        if prob >= SPEECH_THRESHOLD:
            if not self.is_speaking:
                self.speech_buffer.extend(list(self.pre_buffer))
                self.pre_buffer.clear()
                logger.info(f"[VAD] Début parole prob={prob:.3f}")
            self.is_speaking   = True
            self.silence_count = 0
            self.speech_buffer.append(frame_bytes)

        # ── Silence détecté (seuil bas pour ignorer bruit de fond) ──────────
        elif self.is_speaking and prob < SILENCE_THRESHOLD:
            self.speech_buffer.append(frame_bytes)
            self.silence_count += 1
            logger.debug(f"[VAD] Silence {self.silence_count}/{SILENCE_FRAMES} prob={prob:.3f}")

            if self.silence_count >= SILENCE_FRAMES:
                complete_audio = b"".join(self.speech_buffer)
                duration = len(complete_audio) / (SAMPLE_RATE * 2)
                logger.info(f"[VAD] Segment détecté : {duration:.2f}s")
                self._reset()
                return complete_audio

        # ── Zone grise (0.3 < prob < 0.5) : on continue à enregistrer ───────
        elif self.is_speaking:
            self.speech_buffer.append(frame_bytes)

        # ── Pas encore en train de parler ────────────────────────────────────
        else:
            self.pre_buffer.append(frame_bytes)

        return None

    def _reset(self):
        self.speech_buffer = []
        self.silence_count = 0
        self.is_speaking   = False

    def flush(self) -> bytes | None:
        if self.speech_buffer:
            complete_audio = b"".join(self.speech_buffer)
            self._reset()
            return complete_audio
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_vad()
    print("Silero VAD ✅")