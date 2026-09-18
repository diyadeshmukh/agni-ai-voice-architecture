"""
Agni AI - Streaming STT WebSocket Endpoint

Receives continuous raw audio from a client such as the LiveKit
audio pipeline and forwards it to the streaming Deepgram STT service.

Expected incoming audio:
    - PCM16 / linear16
    - 16 kHz
    - mono
    - binary WebSocket messages

The endpoint sends JSON events back to the client:

    {"type": "partial", ...}
    {"type": "final", ...}
    {"type": "speech_started"}
    {"type": "utterance_end"}
    {"type": "silence"}
    {"type": "incomplete_speech"}
    {"type": "error", ...}
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.stt_service import DeepgramSTTService


logger = logging.getLogger("stt_api")


router = APIRouter(
    prefix="/stt",
    tags=["stt"],
)


# Sentinel object used to tell the audio generator that the
# WebSocket client has stopped sending audio.
_SENTINEL = object()


@router.websocket("/stream")
async def stt_stream(
    websocket: WebSocket,
) -> None:
    """
    Stream audio from a WebSocket client into Deepgram STT.

    The client sends raw PCM16 audio as binary messages.

    Audio format expected by this endpoint:

        Sample rate : 16 kHz
        Channels    : 1
        Encoding    : linear16 / PCM16
    """

    await websocket.accept()

    logger.info(
        "Streaming STT WebSocket connected."
    )

    audio_queue: asyncio.Queue[bytes | object] = (
        asyncio.Queue()
    )

    async def audio_generator() -> AsyncIterator[bytes]:
        """
        Convert queued WebSocket audio messages into an async
        audio stream for Deepgram.
        """

        while True:
            chunk = await audio_queue.get()

            if chunk is _SENTINEL:
                return

            if not chunk:
                continue

            yield chunk  # type: ignore[misc]

    async def on_transcript(
        payload: dict,
    ) -> None:
        """
        Send Deepgram transcript/VAD events back to the client.
        """

        try:
            await websocket.send_json(
                payload
            )

        except Exception:
            logger.exception(
                "Failed to send STT event to WebSocket client."
            )

    try:
        # Create one continuous Deepgram streaming session.
        stt = DeepgramSTTService(
            samplerate=16_000
        )

        stt_task = asyncio.create_task(
            stt.stream_audio_chunks(
                audio_gen=audio_generator(),
                on_transcript=on_transcript,
                trailing_wait_s=2.0,
                encoding="linear16",
                sample_rate=16_000,
                channels=1,
            )
        )

        # ---------------------------------------------------------------
        # Receive audio continuously from the WebSocket client.
        # ---------------------------------------------------------------

        while True:
            try:
                audio_chunk = (
                    await websocket.receive_bytes()
                )

            except WebSocketDisconnect:
                logger.info(
                    "Streaming STT WebSocket disconnected."
                )
                break

            if audio_chunk:
                await audio_queue.put(
                    audio_chunk
                )

    except RuntimeError as exc:
        logger.error(
            "STT configuration error: %s",
            exc,
        )

        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )

        except Exception:
            pass

    except Exception as exc:
        logger.exception(
            "Unhandled streaming STT error."
        )

        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )

        except Exception:
            pass

    finally:
        # ---------------------------------------------------------------
        # Tell the audio generator that no more audio will arrive.
        # ---------------------------------------------------------------

        await audio_queue.put(
            _SENTINEL
        )

        # ---------------------------------------------------------------
        # Allow the Deepgram streaming session to finalize and shut down
        # cleanly.
        # ---------------------------------------------------------------

        if "stt_task" in locals():
            try:
                await stt_task

            except Exception:
                logger.exception(
                    "Streaming STT task stopped with an error."
                )

        try:
            await websocket.close()

        except Exception:
            pass

        logger.info(
            "Streaming STT WebSocket closed."
        )