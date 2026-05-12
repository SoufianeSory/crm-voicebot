import asyncio
import websockets
import uuid
import audioop
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = "0.0.0.0"
PORT = 9092
VOICEBOT_WS = "ws://localhost:8002/ws/phone/"

SILENCE_ULAW = b'\xff' * 160


def build_packet(kind: int, payload: bytes) -> bytes:
    size = len(payload)
    return bytes([kind, (size >> 8) & 0xff, size & 0xff]) + payload


async def read_packet(reader):
    header  = await reader.readexactly(3)
    kind    = header[0]
    size    = (header[1] << 8) | header[2]
    payload = await reader.readexactly(size) if size > 0 else b""
    return kind, payload


async def handle_client(reader, writer):
    ws             = None
    keepalive_task = None

    try:
        # ── UUID envoyé par Asterisk ──────────────────────────────────────
        kind, payload = await read_packet(reader)

        if kind != 0x01:
            logger.error(f"INVALID UUID PACKET TYPE: {kind}")
            writer.close()
            return

        ast_uuid = str(uuid.UUID(bytes=payload))
        logger.info(f"[+] CALL STARTED {ast_uuid}")

        # ── ACK obligatoire — sans ça Asterisk coupe immédiatement ───────
        ack = build_packet(0x00, b"")
        writer.write(ack)
        await writer.drain()
        logger.info("ACK sent to Asterisk ✅")

        # ── Connexion WebSocket vers phone_server ─────────────────────────
        ws = await websockets.connect(VOICEBOT_WS + ast_uuid, max_size=None)
        logger.info("CONNECTED TO PHONE SERVER ✅")

        # ── Keepalive : silence toutes les 20ms pour éviter timeout ───────
        async def keepalive():
            silence_packet = build_packet(0x10, SILENCE_ULAW)
            while True:
                try:
                    writer.write(silence_packet)
                    await writer.drain()
                    await asyncio.sleep(0.02)
                except Exception:
                    break

        # ── Asterisk → WebSocket ──────────────────────────────────────────
        async def receive_from_asterisk():
            while True:
                kind, payload = await read_packet(reader)
                if kind != 0x10:
                    continue
                pcm8k     = audioop.ulaw2lin(payload, 2)
                pcm16k, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)
                await ws.send(pcm16k)

        # ── WebSocket → Asterisk ──────────────────────────────────────────
        async def send_to_asterisk():
            while True:
                pcm16k = await ws.recv()
                if not isinstance(pcm16k, bytes):
                    continue
                pcm8k, _ = audioop.ratecv(pcm16k, 2, 1, 16000, 8000, None)
                ulaw      = audioop.lin2ulaw(pcm8k, 2)
                packet    = build_packet(0x10, ulaw)
                writer.write(packet)
                await writer.drain()

        # ── Lancer keepalive AVANT gather ─────────────────────────────────
        keepalive_task = asyncio.create_task(keepalive())

        try:
            await asyncio.gather(
                receive_from_asterisk(),
                send_to_asterisk()
            )
        finally:
            keepalive_task.cancel()

    except Exception as e:
        logger.exception(e)

    finally:
        if keepalive_task:
            keepalive_task.cancel()
        try:
            if ws:
                await ws.close()
        except Exception:
            pass
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        logger.info("CALL CLOSED")


async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    logger.info(f"AudioSocket bridge listening on {PORT}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())