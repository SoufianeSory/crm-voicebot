"""
Voicebot Server — FastAPI WebSocket.
Pipeline : Audio PCM → VAD → STT → LangGraph (voice mode) → TTS → Audio

- Whisper et Silero VAD préchargés au démarrage (élimine cold start)
- Endpoint : ws://localhost:8001/ws/call/{session_id}
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.voicebot.vad import VoiceActivityDetector, FRAME_SAMPLES, get_vad
from src.voicebot.stt import transcribe, preload as preload_whisper
from src.voicebot.tts import synthesize
from src.graph.workflow import voice_chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Lifespan — préchargement au démarrage ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Précharge les modèles lourds au démarrage pour éliminer le cold start."""
    logger.info("=== Démarrage voicebot — préchargement des modèles ===")

    loop = asyncio.get_event_loop()

    # Silero VAD (~0.5s)
    await loop.run_in_executor(None, get_vad)

    # Whisper base (~1.5s)
    await loop.run_in_executor(None, preload_whisper)

    logger.info("=== Modèles prêts — voicebot opérationnel ✅ ===")
    yield
    logger.info("=== Arrêt du serveur ===")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Voicebot API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

active_sessions: Dict[str, VoiceActivityDetector] = {}


# ─── Pipeline ─────────────────────────────────────────────────────────────────

async def process_speech(session_id: str, audio_bytes: bytes, websocket: WebSocket):
    """STT → LangGraph (voice) → TTS → envoi audio au client."""

    # 1. STT (filtre durée intégré dans transcribe())
    text = await asyncio.get_event_loop().run_in_executor(None, transcribe, audio_bytes)

    if not text:
        return  # segment trop court ou silence

    logger.info(f"[{session_id}] Transcription : '{text}'")
    await websocket.send_json({"type": "transcript", "text": text})

    # 2. LangGraph voice mode
    logger.info(f"[{session_id}] LangGraph...")
    result = await asyncio.get_event_loop().run_in_executor(None, voice_chat, text)

    response_text = result.get("response", "Je n'ai pas pu traiter votre demande.")
    action        = result.get("action", "respond")

    logger.info(f"[{session_id}] Réponse ({action}) : '{response_text}'")
    await websocket.send_json({"type": "response", "text": response_text, "action": action})

    # 3. TTS
    audio_response = await asyncio.get_event_loop().run_in_executor(None, synthesize, response_text)
    await websocket.send_bytes(audio_response)
    logger.info(f"[{session_id}] Audio envoyé ✅ ({len(audio_response)} bytes)")


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/call/{session_id}")
async def voicebot_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"[{session_id}] Session ouverte")

    vad = VoiceActivityDetector()
    active_sessions[session_id] = vad
    frame_buffer = b""

    try:
        while True:
            data = await websocket.receive_bytes()
            frame_buffer += data

            frame_size_bytes = FRAME_SAMPLES * 2

            while len(frame_buffer) >= frame_size_bytes:
                frame = frame_buffer[:frame_size_bytes]
                frame_buffer = frame_buffer[frame_size_bytes:]

                speech_segment = vad.process_frame(frame)
                if speech_segment:
                    asyncio.create_task(
                        process_speech(session_id, speech_segment, websocket)
                    )

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Session fermée")
        remaining = vad.flush()
        if remaining:
            try:
                await process_speech(session_id, remaining, websocket)
            except Exception:
                pass

    except Exception as e:
        logger.error(f"[{session_id}] Erreur : {e}", exc_info=True)

    finally:
        active_sessions.pop(session_id, None)
        logger.info(f"[{session_id}] Session nettoyée")


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(active_sessions)}


# ─── Lancement ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("voicebot_server:app", host="0.0.0.0", port=8001, reload=False)