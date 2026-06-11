"""
Voicebot Server v5.1 — Multi-sessions + Approche 2 anti-écho.

Architecture :
    Chaque session WebSocket a son propre worker asyncio.
    Les questions sont traitées SÉQUENTIELLEMENT dans chaque session (une par une).
    Les sessions sont INDÉPENDANTES (pas d'interférence entre appels).

Anti-écho (Approche 2) :
    Le micro reste OUVERT en permanence.
    Pendant que le bot parle → filtre énergie RMS pour distinguer écho vs vraie voix.
    Si le client interrompt (barge-in) :
        1. Le bot se tait (signal stop_audio au client)
        2. Ce que le client dit est enregistré dans le contexte
        3. Le LLM en tient compte dans sa prochaine réponse
    Post-TTS mute réduit à 0.3s (juste pour l'écho hardware résiduel).
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import asyncio
import logging
import time
import re
import struct
import math
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.voicebot.vad import VoiceActivityDetector, FRAME_SAMPLES, get_vad
from src.voicebot.stt import transcribe, preload as preload_whisper
from src.voicebot.tts import synthesize_stream, synthesize, get_filler_audio, preload_fillers
from src.graph.workflow import voice_chat
from config.settings import VOICE_INACTIVITY_SEC, validate_config, MAX_CONCURRENT_SESSIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Anti-écho (Approche 2 : micro ouvert + barge-in contextuel) ─────────────
POST_TTS_MUTE_SEC         = 0.3   # ← APPROCHE 2 : réduit (juste écho hardware)
BARGE_IN_ENERGY_THRESHOLD = 0.04
BARGE_IN_MIN_DURATION_SEC = 0.4   # ← APPROCHE 2 : plus réactif aux interruptions


def _rms_energy(audio_bytes: bytes) -> float:
    if len(audio_bytes) < 2:
        return 0.0
    n = len(audio_bytes) // 2
    try:
        samples = struct.unpack(f'<{n}h', audio_bytes[:n * 2])
        return math.sqrt(sum(s * s for s in samples) / n) / 32768.0
    except struct.error:
        return 0.0


def _estimate_audio_duration(audio_bytes: bytes) -> float:
    return max(1.0, len(audio_bytes) / 6000.0)


# ─── Noise filter ─────────────────────────────────────────────────────────────
_NOISE_RE = re.compile(
    r'^\.+$|^\s*$|^[\?\!]+$|^hm+$|^euh+$|^ah+$|^oh+$'
    r'|^sous-titres|^sous titres|^merci d\'avoir regardé'
    r'|^music$|^\[.*\]$'
    r'|^you$|^thank you\.?$|^thanks\.?$',
    re.IGNORECASE
)

# Hallucinations Whisper fréquentes sur du silence/bruit
# NOTE: "merci" seul n'est PAS bloqué — le client peut vraiment dire merci.
# Les hallucinations Whisper de "merci" sont gérées par l'anti-écho (énergie RMS).
_WHISPER_GHOSTS = {
    "sous-titres", "sous titres",
    "merci d'avoir regardé", "merci de votre attention",
    "...", "..", "!", "?",
    "you", "thank you", "thanks",
    "bye bye",
}

def _is_noise(text: str) -> bool:
    text = text.strip()
    if len(text) < 2:
        return True
    if _NOISE_RE.match(text):
        return True
    if text.lower().rstrip('.!? ') in _WHISPER_GHOSTS:
        return True
    # Moins de 1 mot de plus de 1 caractère → bruit
    if len([w for w in text.split() if len(w) > 1]) < 1:
        return True
    return False


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Démarrage voicebot v5 — multi-sessions ===")
    validate_config()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_vad)
    await loop.run_in_executor(None, preload_whisper)
    await preload_fillers()
    logger.info(f"=== Voicebot prêt ✅ (max {MAX_CONCURRENT_SESSIONS} sessions) ===")
    yield


app = FastAPI(title="Voicebot API", version="5.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

active_sessions: Dict[str, "SessionState"] = {}


# ─── Session State (une par appel, isolée) ────────────────────────────────────

class SessionState:
    def __init__(self, websocket: WebSocket, session_id: str):
        self.ws          = websocket
        self.sid         = session_id
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.worker_task  = None
        self.timeout_task = None

        # Anti-écho
        self.is_speaking         = False
        self.speaking_until      = 0.0
        self.post_tts_mute_until = 0.0

        # ← APPROCHE 2 : contexte d'interruption
        self.last_bot_response   = ""
        self.was_interrupted     = False
        self._barge_in_event     = asyncio.Event()

        # Tracking
        self.last_speech   = time.time()
        self.inactivity_ok = False
        self.turn_count    = 0
        self.last_turn     = 0.0
        self.created       = time.time()

        # Lock pour garantir le traitement séquentiel dans cette session
        self._processing = False

    def start(self):
        self.worker_task  = asyncio.create_task(self._worker())
        self.timeout_task = asyncio.create_task(self._inactivity_watcher())

    async def stop(self):
        for t in [self.worker_task, self.timeout_task]:
            if t:
                t.cancel()
                try: await t
                except asyncio.CancelledError: pass

    # ── Anti-écho ─────────────────────────────────────────────────────────────

    def should_accept(self, audio: bytes) -> tuple[bool, str]:
        now = time.time()
        dur = len(audio) / (16000 * 2)

        # Pendant TTS → filtre par énergie
        if self.is_speaking and now < self.speaking_until:
            energy = _rms_energy(audio)
            if dur < BARGE_IN_MIN_DURATION_SEC:
                return False, f"écho court ({dur:.1f}s)"
            if energy < BARGE_IN_ENERGY_THRESHOLD:
                return False, f"écho faible ({energy:.3f})"
            return True, f"barge-in ({energy:.3f})"

        # Juste après TTS → mute résiduel
        if now < self.post_tts_mute_until:
            return False, "post-TTS mute"

        return True, "ok"

    def mark_speaking(self, duration: float):
        self.is_speaking    = True
        self.speaking_until = time.time() + duration

    def mark_done_speaking(self):
        self.is_speaking         = False
        self.speaking_until      = 0.0
        self.post_tts_mute_until = time.time() + POST_TTS_MUTE_SEC

    def flush_queue(self):
        n = 0
        while not self.queue.empty():
            try: self.queue.get_nowait(); n += 1
            except asyncio.QueueEmpty: break
        if n: logger.info(f"[{self.sid}] 🗑️ {n} segments écho vidés")

    # ← APPROCHE 2 : gestion centralisée du barge-in
    async def handle_barge_in(self):
        """Le client interrompt le bot → on stoppe, on garde le contexte."""
        self.was_interrupted = True
        self.is_speaking = False
        self.speaking_until = 0.0
        self.post_tts_mute_until = time.time() + POST_TTS_MUTE_SEC
        self._barge_in_event.set()       # Réveille le sleep TTS
        self.flush_queue()
        try:
            await self.ws.send_json({"type": "stop_audio"})  # Signal au client
        except Exception:
            pass
        logger.info(f"[{self.sid}] ⚡ Barge-in — bot interrompu, micro ouvert")

    # ── Worker : traitement SÉQUENTIEL des segments ───────────────────────────

    async def _worker(self):
        while True:
            try:
                first = await self.queue.get()
                await asyncio.sleep(0.3)  # laisser arriver les segments de la même phrase

                segments = [first]
                while not self.queue.empty():
                    try: segments.append(self.queue.get_nowait())
                    except asyncio.QueueEmpty: break

                if len(segments) > 1:
                    logger.info(f"[{self.sid}] Fusion {len(segments)} segments")

                self._processing = True
                await self._process(b"".join(segments))
                self._processing = False

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._processing = False
                logger.error(f"[{self.sid}] Erreur worker : {e}", exc_info=True)

    # ── Inactivity ────────────────────────────────────────────────────────────

    async def _inactivity_watcher(self):
        while True:
            await asyncio.sleep(5)
            try:
                now = time.time()
                # Déclencher seulement si :
                # - Le client n'a rien dit depuis VOICE_INACTIVITY_SEC
                # - Le bot a fini de parler depuis au moins 20s
                # - On n'est pas en train de traiter ou parler
                # - On n'a pas déjà envoyé la relance
                # - Au moins 1 échange a eu lieu
                time_since_last_turn = now - self.last_turn if self.last_turn > 0 else 0
                time_since_speech = now - self.last_speech

                if (time_since_speech > VOICE_INACTIVITY_SEC
                    and time_since_last_turn > 20
                    and not self.is_speaking and not self._processing
                    and self.turn_count > 0
                    and not self.inactivity_ok):
                    self.inactivity_ok = True
                    logger.info(f"[{self.sid}] Inactivity ({time_since_speech:.0f}s silence) → relance")
                    msg = "Vous êtes toujours là ?"
                    await self.ws.send_json({"type": "response", "text": msg, "action": "respond"})
                    audio = await synthesize(msg)
                    if audio:
                        dur = _estimate_audio_duration(audio)
                        self.mark_speaking(dur)
                        await self.ws.send_bytes(audio)
                        await asyncio.sleep(dur)
                        self.mark_done_speaking()
            except asyncio.CancelledError:
                raise
            except Exception:
                break

    # ── Traitement d'un segment (une question à la fois) ─────────────────────

    async def _process(self, audio_bytes: bytes):
        loop = asyncio.get_event_loop()

        # 1. STT
        try:
            text = await loop.run_in_executor(None, transcribe, audio_bytes)
        except Exception as e:
            logger.error(f"[{self.sid}] STT erreur : {e}")
            return

        if not text or _is_noise(text):
            return

        # Vérification post-STT
        if time.time() < self.post_tts_mute_until:
            logger.info(f"[{self.sid}] 🔇 '{text}' ignoré (post-TTS)")
            return

        self.last_speech   = time.time()
        self.inactivity_ok = False

        logger.info(f"[{self.sid}] 🗣️ Client : '{text}'")
        try:
            await self.ws.send_json({"type": "transcript", "text": text})
        except Exception:
            return

        # 2. LLM (avec filler si trop long)
        # ← APPROCHE 2 : transmet le contexte d'interruption au LLM
        interrupted = self.last_bot_response if self.was_interrupted else ""
        try:
            llm_future = loop.run_in_executor(None, voice_chat, text, self.sid, interrupted)
            try:
                result = await asyncio.wait_for(asyncio.shield(llm_future), timeout=1.5)
            except asyncio.TimeoutError:
                # Envoyer filler pendant qu'on attend
                try:
                    filler = await get_filler_audio()
                    if filler:
                        dur = _estimate_audio_duration(filler)
                        await self.ws.send_json({"type": "response", "text": "Un instant...", "action": "thinking"})
                        self.mark_speaking(dur)
                        await self.ws.send_bytes(filler)
                        await asyncio.sleep(dur)
                        self.mark_done_speaking()
                except Exception:
                    pass
                result = await llm_future
        except Exception as e:
            logger.error(f"[{self.sid}] LLM erreur : {e}")
            result = {"response": "Excusez-moi, un souci technique. Je vais vous mettre en relation avec un conseiller humain.", "action": "escalate"}

        response = result.get("response", "Je n'ai pas compris, pouvez-vous répéter ?")
        action   = result.get("action", "respond")

        logger.info(f"[{self.sid}] 🤖 Alex ({action}) : '{response[:80]}'")
        try:
            await self.ws.send_json({"type": "response", "text": response, "action": action})
        except Exception:
            return

        # 3. TTS — approche 2 : sleep interruptible par barge-in
        self.turn_count += 1
        self.last_turn   = time.time()
        self.last_bot_response = response  # ← APPROCHE 2 : mémorise pour contexte
        self._barge_in_event.clear()       # ← Reset avant de parler

        try:
            chunks = []
            async for chunk in synthesize_stream(response):
                chunks.append(chunk)

            if chunks:
                full = b"".join(chunks)
                dur  = _estimate_audio_duration(full)
                self.mark_speaking(dur)
                await self.ws.send_bytes(full)

                # ← APPROCHE 2 : attend fin audio OU barge-in du client
                try:
                    await asyncio.wait_for(self._barge_in_event.wait(), timeout=dur)
                    # Barge-in → handle_barge_in() a déjà tout géré
                    logger.info(f"[{self.sid}] 🔇 TTS interrompu par barge-in")
                except asyncio.TimeoutError:
                    # Fin normale → cleanup standard
                    self.mark_done_speaking()
                    self.flush_queue()
                    self.was_interrupted = False

                logger.info(f"[{self.sid}] 🔊 Audio envoyé ({len(full)}B, ~{dur:.1f}s)")
        except Exception as e:
            logger.warning(f"[{self.sid}] TTS erreur : {e}")
            self.is_speaking = False
            self.post_tts_mute_until = time.time() + POST_TTS_MUTE_SEC

    def stats(self) -> dict:
        return {
            "session_id": self.sid, "turns": self.turn_count,
            "speaking": self.is_speaking, "processing": self._processing,
            "age_sec": round(time.time() - self.created),
            "idle_sec": round(time.time() - self.last_speech),
        }


# ─── WebSocket endpoint ──────────────────────────────────────────────────────

@app.websocket("/ws/call/{session_id}")
async def voicebot_ws(websocket: WebSocket, session_id: str):
    # Vérifier la limite de sessions
    if len(active_sessions) >= MAX_CONCURRENT_SESSIONS:
        await websocket.close(code=1013, reason="Trop de sessions actives")
        logger.warning(f"[{session_id}] Refusé : limite {MAX_CONCURRENT_SESSIONS} atteinte")
        return

    await websocket.accept()
    logger.info(f"[{session_id}] ══════ Session ouverte ({len(active_sessions)+1} actives) ══════")

    state = SessionState(websocket, session_id)
    state.start()
    active_sessions[session_id] = state

    vad = VoiceActivityDetector()
    buf = b""

    try:
        while True:
            data = await websocket.receive_bytes()
            buf += data

            frame_bytes = FRAME_SAMPLES * 2
            while len(buf) >= frame_bytes:
                frame = buf[:frame_bytes]
                buf   = buf[frame_bytes:]

                segment = vad.process_frame(frame)
                if not segment:
                    continue

                ok, reason = state.should_accept(segment)
                if ok:
                    if "barge-in" in reason:
                        await state.handle_barge_in()  # ← APPROCHE 2
                    await state.queue.put(segment)
                else:
                    logger.debug(f"[{session_id}] 🔇 Ignoré — {reason}")

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Session fermée")
        remaining = vad.flush()
        if remaining:
            ok, _ = state.should_accept(remaining)
            if ok:
                await state.queue.put(remaining)
                await asyncio.sleep(2)

    except Exception as e:
        logger.error(f"[{session_id}] Erreur : {e}", exc_info=True)

    finally:
        await state.stop()
        active_sessions.pop(session_id, None)
        logger.info(f"[{session_id}] ══════ Session nettoyée ({len(active_sessions)} restantes) ══════")


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status"   : "ok",
        "version"  : "5.0.0",
        "sessions" : len(active_sessions),
        "max"      : MAX_CONCURRENT_SESSIONS,
        "details"  : [s.stats() for s in active_sessions.values()],
    }


if __name__ == "__main__":
    uvicorn.run("voicebot_server:app", host="0.0.0.0", port=8001, reload=False)