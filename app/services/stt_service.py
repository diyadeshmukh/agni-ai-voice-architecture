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

    This is a response-latency measurement from the most recently
    sent audio chunk. It is not total utterance latency.
    """

    latency_ms = round(
        (response_ts - audio_sent_ts) * 1000,
        2,
    )

    event_type = "final" if is_final else "partial"

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

    audio_queue: queue.Queue[bytes] = queue.Queue()

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
    Read a WAV file and emit small chunks at approximately real-time pace.
    """

    with wave.open(
        path,
        "rb",
    ) as wav_file:

        samplerate = wav_file.getframerate()

        frames_per_chunk = int(
            samplerate * (chunk_ms / 1000.0)
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
# Sync → Async adapter
# ===========================================================================

async def _as_async_iter(
    sync_gen: Iterator[bytes],
) -> AsyncIterator[bytes]:
    """
    Convert a blocking synchronous generator into an async iterator.
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

    Main external integration point:

        await stt.stream_audio_chunks(...)
    """

    def __init__(
        self,
        samplerate: int = 16_000,
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

        # -------------------------------------------------------------------
        # Deepgram client
        # -------------------------------------------------------------------

        client = AsyncDeepgramClient(
            api_key=self.api_key,
        )

        # -------------------------------------------------------------------
        # Project terminology
        #
        # Keep this list focused on genuine project-specific words/phrases.
        # Deepgram Nova-3 supports keyterm prompting.
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
            language="en-US",
            encoding=encoding,
            sample_rate=sample_rate,
            channels=channels,

            # Return interim hypotheses while the speaker is talking.
            interim_results=True,

            # Backstop signal for a completed utterance.
            utterance_end_ms="1500",

            # Enable VAD events.
            vad_events=True,

            # Slightly less aggressive endpointing than our previous
            # 300 ms configuration.
            endpointing=500,

            # Improve transcript formatting.
            smart_format=True,

            # Project-specific terminology.
            keyterm=keyterms,

        ) as connection:

            # ---------------------------------------------------------------
            # Runtime state
            # ---------------------------------------------------------------

            last_audio_sent_ts: float | None = None

            state = {
                # Whether Deepgram currently considers the speaker to be
                # inside a speech interval.
                "in_speech": False,

                # Whether at least one final result was produced since the
                # most recent utterance boundary.
                "final_since_boundary": False,

                # Most recent interim transcript.
                "last_partial": None,

                # Whether the most recent partial has already been finalized.
                "last_partial_finalized": True,
            }

            # ---------------------------------------------------------------
            # Deepgram message queue
            #
            # The SDK callback itself is synchronous. We place messages into
            # one queue and process them sequentially. This prevents our
            # application-level transcript state from being updated by
            # multiple concurrent callback tasks.
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
                        "on_transcript callback raised an exception."
                    )

            # ---------------------------------------------------------------
            # Message processing
            # ---------------------------------------------------------------

            async def handle_message(
                message,
            ) -> None:

                nonlocal last_audio_sent_ts

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
                        if last_audio_sent_ts is not None
                        else response_ts
                    )

                    is_final = bool(
                        getattr(
                            message,
                            "is_final",
                            False,
                        )
                    )

                    # Deepgram's speech_final indicates the end of a
                    # speech segment, while is_final indicates that this
                    # result itself will not be revised.
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

                    # Track final results.
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
                        }
                    )

                    return

                # ===========================================================
                # Speech started
                # ===========================================================

                if message_type == "SpeechStarted":

                    # Deepgram can occasionally produce more than one
                    # SpeechStarted signal during a continuous session.
                    # Only forward the transition into speech.
                    if not state["in_speech"]:

                        state["in_speech"] = True

                        logger.info(
                            "VAD: speech started"
                        )

                        await emit(
                            {
                                "type": "speech_started",
                            }
                        )

                    return

                # ===========================================================
                # Utterance end
                # ===========================================================

                if message_type == "UtteranceEnd":

                    logger.info(
                        "VAD/Endpointing: utterance end"
                    )

                    if not state[
                        "final_since_boundary"
                    ]:

                        logger.info(
                            "Silence detected: utterance ended "
                            "without a final transcript."
                        )

                        await emit(
                            {
                                "type": "silence",
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
            # Sequential message processor
            # ---------------------------------------------------------------

            async def process_message_queue() -> None:

                while True:

                    message = await message_queue.get()

                    try:

                        if message is None:
                            return

                        await handle_message(
                            message
                        )

                    finally:
                        message_queue.task_done()

            processor_task = asyncio.create_task(
                process_message_queue()
            )

            # ---------------------------------------------------------------
            # Synchronous SDK callbacks
            # ---------------------------------------------------------------

            def on_message(
                message,
            ) -> None:
                """
                Queue Deepgram events for sequential async processing.
                """

                message_queue.put_nowait(
                    message
                )

            def on_open(
                _message,
            ) -> None:

                logger.info(
                    "Deepgram streaming connection opened."
                )

            def on_close(
                message,
            ) -> None:

                logger.info(
                    "Deepgram streaming connection closed: %s",
                    message,
                )

            def on_error(
                error,
            ) -> None:

                logger.error(
                    "Deepgram streaming error: %s",
                    error,
                )

                # Forward errors through the same message queue so
                # output ordering remains deterministic.
                message_queue.put_nowait(
                    {
                        "__agni_error__": str(
                            error
                        )
                    }
                )

            # ---------------------------------------------------------------
            # Register event handlers
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
            # Start Deepgram receive loop
            # ---------------------------------------------------------------

            listener_task = asyncio.create_task(
                connection.start_listening()
            )

            try:

                # ===========================================================
                # Continuous audio transmission
                # ===========================================================

                async for chunk in audio_gen:

                    if not chunk:
                        continue

                    last_audio_sent_ts = time.time()

                    await connection.send_media(
                        chunk
                    )

            finally:

                # ===========================================================
                # Finalize the Deepgram stream FIRST
                # ===========================================================

                try:

                    await connection.send_finalize()

                except Exception:

                    logger.debug(
                        "Deepgram finalize failed.",
                        exc_info=True,
                    )

                # ===========================================================
                # Allow trailing results to arrive
                # ===========================================================

                await asyncio.sleep(
                    trailing_wait_s
                )

                # Make sure everything already received from Deepgram has
                # been processed before checking for incomplete speech.
                await message_queue.join()

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
                        "Incomplete speech at stream end: %r",
                        state[
                            "last_partial"
                        ],
                    )

                    await emit(
                        {
                            "type": "incomplete_speech",
                            "transcript": state[
                                "last_partial"
                            ],
                        }
                    )

                # ===========================================================
                # Tell Deepgram no more audio is coming
                # ===========================================================

                try:

                    await connection.send_close_stream()

                except Exception:

                    logger.debug(
                        "Deepgram close-stream failed.",
                        exc_info=True,
                    )

                # ===========================================================
                # Stop the message processor cleanly
                # ===========================================================

                message_queue.put_nowait(
                    None
                )

                await message_queue.join()

                # The processor returns after consuming None.
                await processor_task

                # ===========================================================
                # Wait for the Deepgram listener
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

                    except Exception:

                        logger.debug(
                            "Deepgram listener stopped with an exception.",
                            exc_info=True,
                        )

                else:

                    try:

                        listener_task.result()

                    except asyncio.CancelledError:
                        pass

                    except Exception:

                        logger.debug(
                            "Deepgram listener stopped with an exception.",
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
                    samplerate=self.samplerate,
                )
            )
        )


    # =========================================================================
    # Standalone WAV file
    # =========================================================================

    async def stream_from_file(
        self,
        path: str,
    ) -> None:

        await self._run(
            _as_async_iter(
                _file_audio_generator(
                    path,
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
            trailing_wait_s=trailing_wait_s,
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
            "Agni AI streaming Deepgram STT service."
        )
    )

    parser.add_argument(
        "--source",
        choices=[
            "mic",
            "file",
        ],
        required=True,
        help="Audio source for standalone testing.",
    )

    parser.add_argument(
        "--path",
        help=(
            "WAV file path when using --source file."
        ),
    )

    args = parser.parse_args()

    if (
        args.source == "file"
        and not args.path
    ):
        parser.error(
            "--path is required for --source file."
        )

    stt = DeepgramSTTService()

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