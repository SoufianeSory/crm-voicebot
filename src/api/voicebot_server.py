"""
Voicebot Server — FastAPI WebSocket.
Fix double TTS : queue séquentielle FIFO — jamais 2 réponses simultanées.
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Démarrage voicebot — préchargement des modèles ===")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_vad)
    await loop.run_in_executor(None, preload_whisper)
    logger.info("=== Modèles prêts — voicebot opérationnel ✅ ===")
    yield


app = FastAPI(title="Voicebot API", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

active_sessions: Dict[str, "SessionState"] = {}


class SessionState:
    """
    Queue FIFO + 1 seul worker par session.
    Garantit que les segments sont traités un par un — jamais en parallèle.
    """
    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket  = websocket
        self.session_id = session_id
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.worker_task = None

    def start_worker(self):
        self.worker_task = asyncio.create_task(self._worker())

    async def stop_worker(self):
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def _worker(self):
        """Consomme les segments un par un — FIFO, jamais concurrent."""
        while True:
            audio_bytes = await self.queue.get()
            try:
                await self._process(audio_bytes)
            except Exception as e:
                logger.error(f"[{self.session_id}] Erreur : {e}", exc_info=True)
            finally:
                self.queue.task_done()

    async def _process(self, audio_bytes: bytes):
        sid  = self.session_id
        ws   = self.websocket
        loop = asyncio.get_event_loop()

        # 1. STT
        text = await loop.run_in_executor(None, transcribe, audio_bytes)
        if not text:
            return

        logger.info(f"[{sid}] Transcription : '{text}'")
        try:
            await ws.send_json({"type": "transcript", "text": text})
        except Exception:
            logger.warning(f"[{sid}] WS fermé — abandon")
            return

        # 2. LangGraph
        result        = await loop.run_in_executor(None, voice_chat, text)
        response_text = result.get("response", "Je n'ai pas pu traiter votre demande.")
        action        = result.get("action", "respond")

        logger.info(f"[{sid}] Réponse ({action}) : '{response_text}'")
        try:
            await ws.send_json({"type": "response", "text": response_text, "action": action})
        except Exception:
            logger.warning(f"[{sid}] WS fermé — abandon")
            return

        # 3. TTS
        try:
            audio_response = await synthesize(response_text)
            await ws.send_bytes(audio_response)
            logger.info(f"[{sid}] Audio envoyé ✅ ({len(audio_response)} bytes)")
        except Exception:
            logger.warning(f"[{sid}] WS fermé — abandon TTS")


@app.websocket("/ws/call/{session_id}")
async def voicebot_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"[{session_id}] Session ouverte")

    state = SessionState(websocket, session_id)
    state.start_worker()
    active_sessions[session_id] = state

    vad          = VoiceActivityDetector()
    frame_buffer = b""

    try:
        while True:
            data         = await websocket.receive_bytes()
            frame_buffer += data

            frame_size_bytes = FRAME_SAMPLES * 2
            while len(frame_buffer) >= frame_size_bytes:
                frame        = frame_buffer[:frame_size_bytes]
                frame_buffer = frame_buffer[frame_size_bytes:]

                speech_segment = vad.process_frame(frame)
                if speech_segment:
                    await state.queue.put(speech_segment)
                    logger.info(f"[{session_id}] Segment en queue (total={state.queue.qsize()})")

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Session fermée")
        remaining = vad.flush()
        if remaining:
            await state.queue.put(remaining)
            await state.queue.join()

    except Exception as e:
        logger.error(f"[{session_id}] Erreur : {e}", exc_info=True)

    finally:
        await state.stop_worker()
        active_sessions.pop(session_id, None)
        logger.info(f"[{session_id}] Session nettoyée")


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(active_sessions)}


if __name__ == "__main__":
    uvicorn.run("voicebot_server:app", host="0.0.0.0", port=8001, reload=False)