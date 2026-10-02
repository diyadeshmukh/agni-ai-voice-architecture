"""
Agni AI - Streaming STT WebSocket Endpoint

Receives continuous raw audio from a client such as the LiveKit
audio pipeline and forwards it to the streaming Deepgram STT service.

Expected incoming audio:
    - PCM16 / linear16
    - 16 kHz
    - mono
    - binary WebSocket messages

User-facing language routing:

    English
        -> Flux
        -> flux-general-en

    Hindi
        -> Flux Multilingual
        -> flux-general-multi
        -> language_hint=["hi"]

    Hinglish
        -> Flux Multilingual
        -> flux-general-multi
        -> language_hint=["en", "hi"]

    Marathi
        -> Nova-3
        -> language="mr"

Optional manual/debug mode:

    multi
        -> Nova-3 legacy/debug mode

Examples:

    /api/v1/stt/stream?language=english

    /api/v1/stt/stream?language=hindi

    /api/v1/stt/stream?language=hinglish

    /api/v1/stt/stream?language=marathi

Language aliases are also accepted:

    /api/v1/stt/stream?language=en

    /api/v1/stt/stream?language=en-IN

    /api/v1/stt/stream?language=hi

    /api/v1/stt/stream?language=mr

    /api/v1/stt/stream?language=multi

If no language query parameter is supplied,
DeepgramSTTService uses DEEPGRAM_STT_LANGUAGE
or defaults to "multi".

The endpoint sends JSON events back to the client:

    {"type": "partial", ...}
    {"type": "final", ...}
    {"type": "speech_started", ...}
    {"type": "utterance_end", ...}
    {"type": "silence", ...}
    {"type": "incomplete_speech", ...}
    {"type": "error", ...}
"""

from __future__ import annotations

import asyncio
import logging

from typing import AsyncIterator

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)

from app.services.stt_service import (
    DeepgramSTTService,
)


# ===========================================================================
# Logging
# ===========================================================================

logger = logging.getLogger(
    "stt_api"
)


# ===========================================================================
# Router
# ===========================================================================

router = APIRouter(
    prefix="/stt",
    tags=["stt"],
)


# ===========================================================================
# Audio queue sentinel
# ===========================================================================

_SENTINEL = object()


# ===========================================================================
# Supported canonical Deepgram language modes
# ===========================================================================

SUPPORTED_LANGUAGE_MODES = {
    "english",
    "hindi",
    "hinglish",
    "marathi",
    "multi",
}


# ===========================================================================
# Streaming STT WebSocket
# ===========================================================================

@router.websocket(
    "/stream"
)
async def stt_stream(
    websocket: WebSocket,
) -> None:
    """
    Stream audio from a WebSocket client into Deepgram STT.

    Expected audio:

        Sample rate : 16 kHz
        Channels    : 1
        Encoding    : linear16 / PCM16

    Recommended user-facing query values:

        ?language=english

        ?language=hindi

        ?language=hinglish

        ?language=marathi

    Canonical values also work:

        ?language=en-IN

        ?language=hi

        ?language=mr

        ?language=multi
    """

    # -----------------------------------------------------------------------
    # Read optional language from WebSocket query string
    # -----------------------------------------------------------------------

    requested_language = (
        websocket.query_params.get(
            "language"
        )
    )

    language_aliases = {
        # English
        "en": "english",
        "en-in": "english",
        "english": "english",

        # Hindi
        "hi": "hindi",
        "hindi": "hindi",

        # Hinglish
        "hinglish": "hinglish",

        # Marathi
        "mr": "marathi",
        "marathi": "marathi",

        # Optional legacy/debug mode
        "multi": "multi",
    }

    if requested_language is not None:

        requested_language = (
            requested_language.strip()
        )

        if requested_language:

            requested_language = (
                language_aliases.get(
                    requested_language.lower(),
                    requested_language,
                )
            )

        else:

            requested_language = None

    # -----------------------------------------------------------------------
    # Accept WebSocket
    # -----------------------------------------------------------------------

    await websocket.accept()

    client_connected = True

    # -----------------------------------------------------------------------
    # Validate requested language
    # -----------------------------------------------------------------------

    if (
        requested_language is not None
        and requested_language
        not in SUPPORTED_LANGUAGE_MODES
    ):

        supported = ", ".join(
            sorted(
                SUPPORTED_LANGUAGE_MODES
            )
        )

        logger.warning(
            "Unsupported STT language "
            "mode requested: %s",
            requested_language,
        )

        try:

            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        "Unsupported STT language "
                        f"mode: {requested_language}. "
                        f"Supported modes: {supported}."
                    ),
                }
            )

        except Exception:

            pass

        try:

            await websocket.close(
                code=1008
            )

        except Exception:

            pass

        return

    # -----------------------------------------------------------------------
    # Connection logging
    # -----------------------------------------------------------------------

    logger.info(
        "Streaming STT WebSocket connected."
    )

    if requested_language:

        logger.info(
            "Requested STT language mode: %s",
            requested_language,
        )

    else:

        logger.info(
            "No STT language query supplied; "
            "using configured default."
        )

    # -----------------------------------------------------------------------
    # Audio queue
    # -----------------------------------------------------------------------

    audio_queue: asyncio.Queue[
        bytes | object
    ] = asyncio.Queue()

    # -----------------------------------------------------------------------
    # Audio generator
    # -----------------------------------------------------------------------

    async def audio_generator(
    ) -> AsyncIterator[bytes]:
        """
        Convert queued WebSocket audio into
        an asynchronous Deepgram audio stream.
        """

        while True:

            chunk = (
                await audio_queue.get()
            )

            if chunk is _SENTINEL:
                return

            if not chunk:
                continue

            yield chunk  # type: ignore[misc]

    # -----------------------------------------------------------------------
    # Deepgram -> WebSocket callback
    # -----------------------------------------------------------------------

    async def on_transcript(
        payload: dict,
    ) -> None:
        """
        Forward Deepgram transcript and VAD
        events to the WebSocket client.
        """

        nonlocal client_connected

        if not client_connected:
            return

        try:

            await websocket.send_json(
                payload
            )

        except WebSocketDisconnect:

            client_connected = False

            logger.info(
                "STT WebSocket client disconnected "
                "before event could be sent."
            )

        except RuntimeError:

            client_connected = False

            logger.debug(
                "STT WebSocket already closed; "
                "dropping event."
            )

        except Exception:

            client_connected = False

            logger.debug(
                "Unable to send STT event because "
                "the WebSocket client disconnected.",
                exc_info=True,
            )

    # -----------------------------------------------------------------------
    # Deepgram streaming task
    # -----------------------------------------------------------------------

    stt_task: asyncio.Task | None = None

    try:

        stt = DeepgramSTTService(
            samplerate=16_000,
            language=requested_language,
        )

        logger.info(
            "Deepgram STT language mode: %s",
            stt.language,
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
        # Receive continuous audio from WebSocket client
        # ---------------------------------------------------------------

        while True:

            try:

                audio_chunk = (
                    await websocket.receive_bytes()
                )

            except WebSocketDisconnect:

                client_connected = False

                logger.info(
                    "Streaming STT "
                    "WebSocket disconnected."
                )

                break

            if audio_chunk:

                await audio_queue.put(
                    audio_chunk
                )

    # -----------------------------------------------------------------------
    # Configuration errors
    # -----------------------------------------------------------------------

    except RuntimeError as exc:

        logger.error(
            "STT configuration error: %s",
            exc,
        )

        if client_connected:

            try:

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": str(
                            exc
                        ),
                    }
                )

            except Exception:

                client_connected = False

    # -----------------------------------------------------------------------
    # Unexpected errors
    # -----------------------------------------------------------------------

    except Exception as exc:

        logger.exception(
            "Unhandled streaming STT error."
        )

        if client_connected:

            try:

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": str(
                            exc
                        ),
                    }
                )

            except Exception:

                client_connected = False

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------

    finally:

        # Prevent Deepgram trailing events from attempting
        # to send to a disconnected WebSocket client.
        client_connected = False

        # Tell audio generator that input has ended.
        await audio_queue.put(
            _SENTINEL
        )

        # Allow Deepgram session to finalize.
        if stt_task is not None:

            try:

                await stt_task

            except asyncio.CancelledError:

                pass

            except Exception:

                logger.exception(
                    "Streaming STT task "
                    "stopped with an error."
                )

        # Close WebSocket if necessary.
        try:

            await websocket.close()

        except Exception:

            pass

        logger.info(
            "Streaming STT WebSocket closed."
        )