import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import asyncio
import logging
import io

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from pydub import AudioSegment

from src.voicebot.vad import VoiceActivityDetector, FRAME_SAMPLES, get_vad
from src.voicebot.stt import transcribe, preload as preload_whisper
from src.voicebot.tts import synthesize
from src.graph.workflow import voice_chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# MP3 -> PCM16 16k mono
# =========================================================

async def convert_mp3_to_pcm16(mp3_bytes: bytes) -> bytes:

    loop = asyncio.get_event_loop()

    def _convert():

        audio = AudioSegment.from_file(
            io.BytesIO(mp3_bytes),
            format="mp3"
        )

        audio = (
            audio
            .set_frame_rate(16000)
            .set_channels(1)
            .set_sample_width(2)
        )

        return audio.raw_data

    return await loop.run_in_executor(None, _convert)


# =========================================================
# FastAPI lifespan
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("=== PHONE VOICEBOT STARTING ===")

    loop = asyncio.get_event_loop()

    await loop.run_in_executor(None, get_vad)
    await loop.run_in_executor(None, preload_whisper)

    logger.info("=== PHONE VOICEBOT READY ✅ ===")

    yield


app = FastAPI(
    title="Phone Voicebot API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Session state
# =========================================================

class SessionState:

    def __init__(self, websocket: WebSocket, session_id: str):

        self.websocket = websocket
        self.session_id = session_id

        self.queue = asyncio.Queue()

        self.worker_task = None

    def start_worker(self):

        self.worker_task = asyncio.create_task(
            self._worker()
        )

    async def stop_worker(self):

        if self.worker_task:

            self.worker_task.cancel()

            try:
                await self.worker_task

            except asyncio.CancelledError:
                pass

    async def _worker(self):

        while True:

            audio_bytes = await self.queue.get()

            try:
                await self._process(audio_bytes)

            except Exception as e:
                logger.error(
                    f"[{self.session_id}] {e}",
                    exc_info=True
                )

            finally:
                self.queue.task_done()

    async def _process(self, audio_bytes: bytes):

        sid = self.session_id

        loop = asyncio.get_event_loop()

        # =================================================
        # STT
        # =================================================

        text = await loop.run_in_executor(
            None,
            transcribe,
            audio_bytes
        )

        if not text:
            return

        logger.info(f"[{sid}] USER: {text}")

        # =================================================
        # LangGraph
        # =================================================

        result = await loop.run_in_executor(
            None,
            voice_chat,
            text
        )

        response_text = result.get(
            "response",
            "Je n'ai pas compris."
        )

        logger.info(f"[{sid}] BOT: {response_text}")

        # =================================================
        # TTS
        # =================================================

        mp3_bytes = await synthesize(response_text)

        pcm16_bytes = await convert_mp3_to_pcm16(
            mp3_bytes
        )

        # =================================================
        # SEND BACK TO BRIDGE
        # =================================================

        await self.websocket.send_bytes(
            pcm16_bytes
        )


# =========================================================
# WebSocket
# =========================================================

@app.websocket("/ws/phone/{session_id}")
async def phone_ws(websocket: WebSocket, session_id: str):

    await websocket.accept()

    logger.info(f"[{session_id}] PHONE CONNECTED")

    state = SessionState(
        websocket,
        session_id
    )

    state.start_worker()

    vad = VoiceActivityDetector()

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

                    await state.queue.put(
                        speech_segment
                    )

    except WebSocketDisconnect:

        logger.info(
            f"[{session_id}] PHONE DISCONNECTED"
        )

        remaining = vad.flush()

        if remaining:

            await state.queue.put(remaining)

            await state.queue.join()

    finally:

        await state.stop_worker()


# =========================================================
# Health
# =========================================================

@app.get("/health")
async def health():

    return {
        "status": "ok"
    }


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    uvicorn.run(
        "src.api.phone_server:app",
        host="0.0.0.0",
        port=8002,
        reload=False
    )