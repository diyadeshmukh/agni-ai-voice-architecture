"""
Agni AI - Deepgram Streaming STT Service

Streaming speech-to-text service for the Agni AI voice pipeline.

Audio flow:

    Audio source
        ↓
    Persistent Deepgram WebSocket
        ↓
    Partial / Final transcript events
        ↓
    VAD / utterance events

Current Deepgram SDK:
    7.x

Current model:
    Nova-3

Language modes used by Agni AI:

    multi
        English + Hinglish / English-Hindi code-switching

    hi
        Hindi

    mr
        Marathi

Expected primary audio format:
    linear16 / PCM16
    16 kHz
    mono
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import queue
import time
import wave

from typing import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Optional,
)

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from app.core.config import settings


# ===========================================================================
# Logging
# ===========================================================================

logger = logging.getLogger("stt_service")


# ===========================================================================
# Latency logging
# ===========================================================================

LATENCY_LOG_PATH = "latency_log.csv"


if not os.path.exists(LATENCY_LOG_PATH):
    with open(
        LATENCY_LOG_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "event_type,chunk_sent_ts,response_ts,"
            "latency_ms,is_final,transcript\n"
        )


def _log_latency(
    audio_sent_ts: float,
    response_ts: float,
    is_final: bool,
    transcript: str,
) -> float:
    """
    Record streaming response latency.

    This measures response latency from the most recently
    sent audio chunk. It is not total utterance latency.
    """

    latency_ms = round(
        (response_ts - audio_sent_ts) * 1000,
        2,
    )

    event_type = (
        "final"
        if is_final
        else "partial"
    )

    with open(
        LATENCY_LOG_PATH,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            f'{event_type},{audio_sent_ts},{response_ts},'
            f'{latency_ms},{is_final},"{transcript}"\n'
        )

    logger.info(
        "[%s] %r (latency: %.2f ms)",
        event_type.upper(),
        transcript,
        latency_ms,
    )

    return latency_ms


# ===========================================================================
# Standalone microphone source
# ===========================================================================

def _mic_audio_generator(
    samplerate: int = 16_000,
    blocksize: int = 4_000,
) -> Iterator[bytes]:
    """
    Generate raw PCM16 mono audio from the system microphone.

    Used only for standalone STT testing.
    """

    import sounddevice as sd

    audio_queue: queue.Queue[bytes] = (
        queue.Queue()
    )

    def callback(
        indata,
        frames,
        time_info,
        status,
    ) -> None:

        if status:
            logger.warning(
                "%s",
                status,
            )

        audio_queue.put(
            bytes(indata)
        )

    stream = sd.RawInputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        dtype="int16",
        channels=1,
        callback=callback,
    )

    with stream:

        logger.info(
            "Microphone stream started."
        )

        while True:
            yield audio_queue.get()


# ===========================================================================
# Standalone WAV source
# ===========================================================================

def _file_audio_generator(
    path: str,
    chunk_ms: int = 100,
) -> Iterator[bytes]:
    """
    Read a WAV file and emit audio chunks at approximately
    real-time pace.
    """

    with wave.open(
        path,
        "rb",
    ) as wav_file:

        samplerate = (
            wav_file.getframerate()
        )

        frames_per_chunk = int(
            samplerate
            * (chunk_ms / 1000.0)
        )

        data = wav_file.readframes(
            frames_per_chunk
        )

        while data:

            yield data

            time.sleep(
                chunk_ms / 1000.0
            )

            data = wav_file.readframes(
                frames_per_chunk
            )


# ===========================================================================
# Sync -> Async adapter
# ===========================================================================

async def _as_async_iter(
    sync_gen: Iterator[bytes],
) -> AsyncIterator[bytes]:
    """
    Convert a blocking synchronous generator
    into an asynchronous iterator.
    """

    iterator = iter(
        sync_gen
    )

    while True:

        chunk = await asyncio.to_thread(
            next,
            iterator,
            None,
        )

        if chunk is None:
            return

        yield chunk


# ===========================================================================
# Callback type
# ===========================================================================

TranscriptCallback = Callable[
    [dict],
    Awaitable[None],
]


# ===========================================================================
# Deepgram Streaming STT
# ===========================================================================

class DeepgramSTTService:
    """
    Persistent Deepgram streaming STT service.

    Main integration point:

        await stt.stream_audio_chunks(...)

    Language configuration priority:

        1. language passed to constructor
        2. DEEPGRAM_STT_LANGUAGE environment variable
        3. "multi"
    """

    def __init__(
        self,
        samplerate: int = 16_000,
        language: str | None = None,
    ) -> None:

        self.samplerate = samplerate

        self.api_key = (
            settings.DEEPGRAM_API_KEY
            or os.environ.get(
                "DEEPGRAM_API_KEY"
            )
        )

        if not self.api_key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY is not set. "
                "Add it to .env.local or the environment."
            )

        self.language = (
            language
            or os.getenv(
                "DEEPGRAM_STT_LANGUAGE",
                "multi",
            )
        ).strip()

        if not self.language:
            self.language = "multi"

        self.language = (
            self.language.lower()
        )

    async def _run(
        self,
        audio_gen: AsyncIterator[bytes],
        on_transcript: Optional[
            TranscriptCallback
        ] = None,
        trailing_wait_s: float = 2.0,
        encoding: str = "linear16",
        sample_rate: Optional[int] = None,
        channels: int = 1,
    ) -> None:

        sample_rate = (
            sample_rate
            or self.samplerate
        )

        logger.info(
            "Deepgram STT language mode: %s",
            self.language,
        )

        # -------------------------------------------------------------------
        # Deepgram client
        # -------------------------------------------------------------------

        client = AsyncDeepgramClient(
            api_key=self.api_key,
        )

        # -------------------------------------------------------------------
        # Project terminology
        # -------------------------------------------------------------------

        keyterms = [
            "Agni AI",
            "LiveKit",
            "AudioFrame",
            "Deepgram",
            "speech-to-text",
            "streaming STT",
            "voice pipeline",
        ]

        # -------------------------------------------------------------------
        # Persistent Deepgram connection
        # -------------------------------------------------------------------

        async with client.listen.v1.connect(
            model="nova-3",
            language=self.language,
            encoding=encoding,
            sample_rate=sample_rate,
            channels=channels,

            interim_results=True,

            # Backstop utterance boundary.
            utterance_end_ms="1500",

            # Enable VAD events.
            vad_events=True,

            # Keep our latency-tested endpointing value.
            endpointing=500,

            smart_format=True,

            keyterm=keyterms,

        ) as connection:

            # ---------------------------------------------------------------
            # Runtime state
            # ---------------------------------------------------------------

            last_audio_sent_ts: (
                float | None
            ) = None

            state = {
                "in_speech": False,
                "final_since_boundary": False,
                "last_partial": None,
                "last_partial_finalized": True,
            }

            # ---------------------------------------------------------------
            # Sequential Deepgram message queue
            # ---------------------------------------------------------------

            message_queue: asyncio.Queue = (
                asyncio.Queue()
            )

            # ---------------------------------------------------------------
            # Event forwarding helper
            # ---------------------------------------------------------------

            async def emit(
                payload: dict,
            ) -> None:

                if on_transcript is None:
                    return

                try:
                    await on_transcript(
                        payload
                    )

                except Exception:
                    logger.exception(
                        "on_transcript callback "
                        "raised an exception."
                    )

            # ---------------------------------------------------------------
            # Message processor
            # ---------------------------------------------------------------

            async def handle_message(
                message,
            ) -> None:

                nonlocal last_audio_sent_ts

                # -----------------------------------------------------------
                # Internal error event
                # -----------------------------------------------------------

                if isinstance(
                    message,
                    dict,
                ):

                    error_message = (
                        message.get(
                            "__agni_error__"
                        )
                    )

                    if error_message:

                        await emit(
                            {
                                "type": "error",
                                "message": error_message,
                            }
                        )

                    return

                message_type = getattr(
                    message,
                    "type",
                    "",
                )

                # ===========================================================
                # Transcript result
                # ===========================================================

                if message_type == "Results":

                    channel = getattr(
                        message,
                        "channel",
                        None,
                    )

                    if channel is None:
                        return

                    alternatives = getattr(
                        channel,
                        "alternatives",
                        [],
                    )

                    if not alternatives:
                        return

                    transcript = getattr(
                        alternatives[0],
                        "transcript",
                        "",
                    )

                    if not transcript:
                        return

                    response_ts = time.time()

                    sent_ts = (
                        last_audio_sent_ts
                        if last_audio_sent_ts
                        is not None
                        else response_ts
                    )

                    is_final = bool(
                        getattr(
                            message,
                            "is_final",
                            False,
                        )
                    )

                    speech_final = bool(
                        getattr(
                            message,
                            "speech_final",
                            False,
                        )
                    )

                    latency_ms = _log_latency(
                        sent_ts,
                        response_ts,
                        is_final,
                        transcript,
                    )

                    if is_final:

                        state[
                            "final_since_boundary"
                        ] = True

                        state[
                            "last_partial_finalized"
                        ] = True

                    else:

                        state[
                            "last_partial"
                        ] = transcript

                        state[
                            "last_partial_finalized"
                        ] = False

                    await emit(
                        {
                            "type": (
                                "final"
                                if is_final
                                else "partial"
                            ),
                            "transcript": transcript,
                            "is_final": is_final,
                            "speech_final": speech_final,
                            "latency_ms": latency_ms,
                            "language_mode": self.language,
                        }
                    )

                    return

                # ===========================================================
                # Speech started
                # ===========================================================

                if (
                    message_type
                    == "SpeechStarted"
                ):

                    if not state[
                        "in_speech"
                    ]:

                        state[
                            "in_speech"
                        ] = True

                        logger.info(
                            "VAD: speech started"
                        )

                        await emit(
                            {
                                "type": (
                                    "speech_started"
                                ),
                                "language_mode": (
                                    self.language
                                ),
                            }
                        )

                    return

                # ===========================================================
                # Utterance end
                # ===========================================================

                if (
                    message_type
                    == "UtteranceEnd"
                ):

                    logger.info(
                        "VAD/Endpointing: "
                        "utterance end"
                    )

                    if not state[
                        "final_since_boundary"
                    ]:

                        logger.info(
                            "Silence detected: "
                            "utterance ended without "
                            "a final transcript."
                        )

                        await emit(
                            {
                                "type": "silence",
                                "language_mode": (
                                    self.language
                                ),
                            }
                        )

                    state[
                        "final_since_boundary"
                    ] = False

                    state[
                        "in_speech"
                    ] = False

                    await emit(
                        {
                            "type": "utterance_end",
                            "language_mode": (
                                self.language
                            ),
                        }
                    )

                    return

                # ===========================================================
                # Metadata
                # ===========================================================

                if message_type == "Metadata":

                    logger.debug(
                        "Deepgram metadata: %s",
                        message,
                    )

            # ---------------------------------------------------------------
            # Sequential queue processor
            # ---------------------------------------------------------------

            async def process_message_queue(
            ) -> None:

                while True:

                    message = (
                        await message_queue.get()
                    )

                    try:

                        if message is None:
                            return

                        await handle_message(
                            message
                        )

                    finally:

                        message_queue.task_done()

            processor_task = (
                asyncio.create_task(
                    process_message_queue()
                )
            )

            # ---------------------------------------------------------------
            # Deepgram SDK callbacks
            # ---------------------------------------------------------------

            def on_message(
                message,
            ) -> None:

                message_queue.put_nowait(
                    message
                )

            def on_open(
                _message,
            ) -> None:

                logger.info(
                    "Deepgram streaming "
                    "connection opened."
                )

            def on_close(
                message,
            ) -> None:

                logger.info(
                    "Deepgram streaming "
                    "connection closed: %s",
                    message,
                )

            def on_error(
                error,
            ) -> None:

                logger.error(
                    "Deepgram streaming error: %s",
                    error,
                )

                message_queue.put_nowait(
                    {
                        "__agni_error__": str(
                            error
                        )
                    }
                )

            # ---------------------------------------------------------------
            # Register Deepgram callbacks
            # ---------------------------------------------------------------

            connection.on(
                EventType.OPEN,
                on_open,
            )

            connection.on(
                EventType.MESSAGE,
                on_message,
            )

            connection.on(
                EventType.CLOSE,
                on_close,
            )

            connection.on(
                EventType.ERROR,
                on_error,
            )

            # ---------------------------------------------------------------
            # Start Deepgram listener
            # ---------------------------------------------------------------

            listener_task = (
                asyncio.create_task(
                    connection.start_listening()
                )
            )

            try:

                # ===========================================================
                # Continuous audio transmission
                # ===========================================================

                async for chunk in audio_gen:

                    if not chunk:
                        continue

                    last_audio_sent_ts = (
                        time.time()
                    )

                    await connection.send_media(
                        chunk
                    )

            finally:

                # ===========================================================
                # Finalize Deepgram stream
                # ===========================================================

                try:

                    await connection.send_finalize()

                except Exception:

                    logger.debug(
                        "Deepgram finalize failed.",
                        exc_info=True,
                    )

                # ===========================================================
                # Allow trailing results
                # ===========================================================

                try:

                    await asyncio.sleep(
                        trailing_wait_s
                    )

                except asyncio.CancelledError:

                    pass

                # ===========================================================
                # Process anything already received
                # ===========================================================

                try:

                    await message_queue.join()

                except asyncio.CancelledError:

                    pass

                # ===========================================================
                # Incomplete speech
                # ===========================================================

                if (
                    state["last_partial"]
                    and not state[
                        "last_partial_finalized"
                    ]
                ):

                    logger.info(
                        "Incomplete speech "
                        "at stream end: %r",
                        state[
                            "last_partial"
                        ],
                    )

                    await emit(
                        {
                            "type": (
                                "incomplete_speech"
                            ),
                            "transcript": (
                                state[
                                    "last_partial"
                                ]
                            ),
                            "language_mode": (
                                self.language
                            ),
                        }
                    )

                # ===========================================================
                # Close Deepgram stream
                # ===========================================================

                try:

                    await (
                        connection
                        .send_close_stream()
                    )

                except Exception:

                    logger.debug(
                        "Deepgram close-stream failed.",
                        exc_info=True,
                    )

                # ===========================================================
                # Stop message processor
                # ===========================================================

                message_queue.put_nowait(
                    None
                )

                try:

                    await message_queue.join()

                except asyncio.CancelledError:

                    pass

                if not processor_task.done():

                    try:

                        await processor_task

                    except asyncio.CancelledError:

                        pass

                # ===========================================================
                # Wait for Deepgram listener
                # ===========================================================

                if not listener_task.done():

                    try:

                        await asyncio.wait_for(
                            listener_task,
                            timeout=2.0,
                        )

                    except asyncio.TimeoutError:

                        listener_task.cancel()

                        try:

                            await listener_task

                        except asyncio.CancelledError:

                            pass

                    except asyncio.CancelledError:

                        pass

                    except Exception:

                        logger.debug(
                            "Deepgram listener stopped "
                            "with an exception.",
                            exc_info=True,
                        )

                else:

                    try:

                        listener_task.result()

                    except asyncio.CancelledError:

                        pass

                    except Exception:

                        logger.debug(
                            "Deepgram listener stopped "
                            "with an exception.",
                            exc_info=True,
                        )


    # =========================================================================
    # Standalone microphone
    # =========================================================================

    async def stream_from_mic(
        self,
    ) -> None:

        await self._run(
            _as_async_iter(
                _mic_audio_generator(
                    samplerate=(
                        self.samplerate
                    ),
                )
            )
        )


    # =========================================================================
    # Standalone WAV
    # =========================================================================

    async def stream_from_file(
        self,
        path: str,
    ) -> None:

        await self._run(
            _as_async_iter(
                _file_audio_generator(
                    path
                )
            )
        )


    # =========================================================================
    # External audio source
    # =========================================================================

    async def stream_audio_chunks(
        self,
        audio_gen: AsyncIterator[bytes],
        on_transcript: Optional[
            TranscriptCallback
        ] = None,
        trailing_wait_s: float = 2.0,
        encoding: str = "linear16",
        sample_rate: Optional[int] = None,
        channels: int = 1,
    ) -> None:

        await self._run(
            audio_gen=audio_gen,
            on_transcript=on_transcript,
            trailing_wait_s=(
                trailing_wait_s
            ),
            encoding=encoding,
            sample_rate=sample_rate,
            channels=channels,
        )


# ===========================================================================
# Standalone command-line entry point
# ===========================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(message)s"
        ),
    )

    parser = argparse.ArgumentParser(
        description=(
            "Agni AI streaming "
            "Deepgram STT service."
        )
    )

    parser.add_argument(
        "--source",
        choices=[
            "mic",
            "file",
        ],
        required=True,
        help=(
            "Audio source for "
            "standalone testing."
        ),
    )

    parser.add_argument(
        "--path",
        help=(
            "WAV file path when "
            "using --source file."
        ),
    )

    parser.add_argument(
        "--language",
        default=None,
        help=(
            "Deepgram language mode. "
            "Agni AI currently uses "
            "multi, hi, or mr."
        ),
    )

    args = parser.parse_args()

    if (
        args.source == "file"
        and not args.path
    ):
        parser.error(
            "--path is required "
            "for --source file."
        )

    stt = DeepgramSTTService(
        language=args.language,
    )

    try:

        if args.source == "mic":

            asyncio.run(
                stt.stream_from_mic()
            )

        else:

            asyncio.run(
                stt.stream_from_file(
                    args.path
                )
            )

    except KeyboardInterrupt:

        logger.info(
            "STT stopped by user."
        )