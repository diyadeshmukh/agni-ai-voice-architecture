"""
Agni AI - Deepgram Streaming STT Service

Streaming speech-to-text service for the Agni AI voice pipeline.

Production routing:

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

    multi
        -> Nova-3 legacy/debug mode

Expected primary audio format:
    linear16 / PCM16
    16 kHz
    mono
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from deepgram.listen.v2.types import ListenV2TurnInfo

from app.core.config import settings


# ===========================================================================
# Logging
# ===========================================================================

logger = logging.getLogger(
    "stt_service"
)


# ===========================================================================
# Latency logging
# ===========================================================================

LATENCY_LOG_PATH = "latency_log.csv"


if not os.path.exists(
    LATENCY_LOG_PATH
):
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
        (
            response_ts
            - audio_sent_ts
        )
        * 1000,
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

        safe_transcript = (
            transcript.replace(
                '"',
                '""',
            )
        )

        file.write(
            f"{event_type},"
            f"{audio_sent_ts},"
            f"{response_ts},"
            f"{latency_ms},"
            f"{is_final},"
            f'"{safe_transcript}"\n'
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

    audio_queue: queue.Queue[
        bytes
    ] = queue.Queue()

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
            * (
                chunk_ms
                / 1000.0
            )
        )

        data = wav_file.readframes(
            frames_per_chunk
        )

        while data:

            yield data

            time.sleep(
                chunk_ms
                / 1000.0
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
# Language routing
# ===========================================================================

LANGUAGE_MODE_ALIASES = {
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

    # Legacy/debug
    "multi": "multi",
}


SUPPORTED_LANGUAGE_MODES = {
    "english",
    "hindi",
    "hinglish",
    "marathi",
    "multi",
}


# ===========================================================================
# Deepgram Streaming STT
# ===========================================================================

class DeepgramSTTService:
    """
    Persistent Deepgram streaming STT service.

    Backend routing:

        english
            -> Flux English

        hindi
            -> Flux Multilingual + hi hint

        hinglish
            -> Flux Multilingual + en/hi hints

        marathi
            -> Nova-3 Marathi

        multi
            -> Nova-3 legacy multilingual mode
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

        requested_language = (
            language
            or os.getenv(
                "DEEPGRAM_STT_LANGUAGE",
                "multi",
            )
        ).strip()

        if not requested_language:

            requested_language = "multi"

        normalized = (
            requested_language.lower()
        )

        self.language_mode = (
            LANGUAGE_MODE_ALIASES.get(
                normalized,
                normalized,
            )
        )

        if (
            self.language_mode
            not in SUPPORTED_LANGUAGE_MODES
        ):

            raise RuntimeError(
                "Unsupported STT language mode: "
                f"{requested_language}"
            )

        # ---------------------------------------------------------------
        # Backend/model routing
        # ---------------------------------------------------------------

        if self.language_mode == "english":

            self.backend = "flux"
            self.model = "flux-general-en"
            self.language = "en"
            self.language_hints: (
                list[str] | None
            ) = None

        elif self.language_mode == "hindi":

            self.backend = "flux"
            self.model = "flux-general-multi"
            self.language = "hi"
            self.language_hints = [
                "hi",
            ]

        elif self.language_mode == "hinglish":

            self.backend = "flux"
            self.model = "flux-general-multi"
            self.language = "en+hi"
            self.language_hints = [
                "en",
                "hi",
            ]

        elif self.language_mode == "marathi":

            self.backend = "nova"
            self.model = "nova-3"
            self.language = "mr"
            self.language_hints = None

        else:

            # Preserve the existing debug/manual mode.
            self.backend = "nova"
            self.model = "nova-3"
            self.language = "multi"
            self.language_hints = None

    # =======================================================================
    # Public runner
    # =======================================================================

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
            "STT routing mode: %s",
            self.language_mode,
        )

        logger.info(
            "Deepgram backend: %s",
            self.backend,
        )

        logger.info(
            "Deepgram model: %s",
            self.model,
        )

        if self.backend == "flux":

            await self._run_flux(
                audio_gen=audio_gen,
                on_transcript=on_transcript,
                trailing_wait_s=trailing_wait_s,
                encoding=encoding,
                sample_rate=sample_rate,
            )

        else:

            await self._run_nova(
                audio_gen=audio_gen,
                on_transcript=on_transcript,
                trailing_wait_s=trailing_wait_s,
                encoding=encoding,
                sample_rate=sample_rate,
                channels=channels,
            )

    # =======================================================================
    # Event helper
    # =======================================================================

    async def _emit(
        self,
        on_transcript: Optional[
            TranscriptCallback
        ],
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

    # =======================================================================
    # Flux
    # =======================================================================

    async def _run_flux(
        self,
        audio_gen: AsyncIterator[bytes],
        on_transcript: Optional[
            TranscriptCallback
        ],
        trailing_wait_s: float,
        encoding: str,
        sample_rate: int,
    ) -> None:
        """
        Run Deepgram Flux v2.

        Flux events are normalized back into Agni's existing
        internal STT event protocol:

            speech_started
            partial
            final
            utterance_end

        This allows the rest of Agni to remain provider-agnostic.
        """

        client = AsyncDeepgramClient(
            api_key=self.api_key,
        )

        keyterms = [
            "Agni AI",
            "Python",
            "list",
            "tuple",
        ]

        connect_kwargs = {
            "model": self.model,
            "encoding": encoding,
            "sample_rate": sample_rate,
            "keyterm": keyterms,
        }

        if self.language_hints:

            connect_kwargs[
                "language_hint"
            ] = self.language_hints

        logger.info(
            "Opening Flux connection."
        )

        if self.language_hints:

            logger.info(
                "Flux language hints: %s",
                self.language_hints,
            )

        async with client.listen.v2.connect(
            **connect_kwargs
        ) as connection:

            last_audio_sent_ts: (
                float | None
            ) = None

            state = {
                "turn_index": None,
                "speech_started_emitted": False,
                "last_partial": "",
                "latest_transcript": "",
                "turn_completed": True,
            }

            message_queue: asyncio.Queue = (
                asyncio.Queue()
            )

            # -----------------------------------------------------------
            # Normalize one Flux message
            # -----------------------------------------------------------

            async def handle_message(
                message,
            ) -> None:

                nonlocal last_audio_sent_ts

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

                        await self._emit(
                            on_transcript,
                            {
                                "type": "error",
                                "message": (
                                    error_message
                                ),
                                "stt_backend": "flux",
                                "model": self.model,
                            },
                        )

                    return

                if not isinstance(
                    message,
                    ListenV2TurnInfo,
                ):

                    message_type = getattr(
                        message,
                        "type",
                        "",
                    )

                    logger.debug(
                        "Flux message: %s",
                        message_type,
                    )

                    return

                event = getattr(
                    message,
                    "event",
                    "",
                )

                transcript = (
                    getattr(
                        message,
                        "transcript",
                        "",
                    )
                    or ""
                ).strip()

                turn_index = getattr(
                    message,
                    "turn_index",
                    None,
                )

                languages = list(
                    getattr(
                        message,
                        "languages",
                        [],
                    )
                    or []
                )

                # -------------------------------------------------------
                # New turn
                # -------------------------------------------------------

                if (
                    turn_index
                    != state["turn_index"]
                ):

                    state[
                        "turn_index"
                    ] = turn_index

                    state[
                        "speech_started_emitted"
                    ] = False

                    state[
                        "last_partial"
                    ] = ""

                    state[
                        "latest_transcript"
                    ] = ""

                    state[
                        "turn_completed"
                    ] = False

                # -------------------------------------------------------
                # Helper: normalized speech start
                #
                # Flux StartOfTurn is the semantic speech-start signal.
                # It may contain an empty transcript, so Agni emits
                # speech_started immediately on StartOfTurn.
                #
                # A non-empty Update remains the fallback if StartOfTurn
                # was not received.
                # -------------------------------------------------------

                async def ensure_speech_started(
                ) -> None:

                    if state[
                        "speech_started_emitted"
                    ]:

                        return

                    state[
                        "speech_started_emitted"
                    ] = True

                    await self._emit(
                        on_transcript,
                        {
                            "type": "speech_started",
                            "transcript": transcript,
                            "language_mode": (
                                self.language_mode
                            ),
                            "languages": languages,
                            "stt_backend": "flux",
                            "model": self.model,
                            "turn_index": turn_index,
                        },
                    )

                # -------------------------------------------------------
                # StartOfTurn / Update
                # -------------------------------------------------------

                if event in {
                    "StartOfTurn",
                    "Update",
                    "TurnResumed",
                }:

                    # Flux StartOfTurn itself is the semantic
                    # turn-start signal used for barge-in.
                    #
                    # StartOfTurn can contain an empty transcript,
                    # so do not discard it.
                    if event == "StartOfTurn":

                        await ensure_speech_started()

                    if not transcript:
                        return

                    state[
                        "latest_transcript"
                    ] = transcript

                    # Fallback in case a non-empty Update arrives
                    # without a prior StartOfTurn.
                    await ensure_speech_started()

                    # Flux sends frequent Update messages, sometimes
                    # with an unchanged transcript.
                    #
                    # Do not flood the Agni event stream with duplicates.
                    if (
                        transcript
                        == state[
                            "last_partial"
                        ]
                    ):

                        return

                    state[
                        "last_partial"
                    ] = transcript

                    response_ts = time.time()

                    sent_ts = (
                        last_audio_sent_ts
                        if last_audio_sent_ts
                        is not None
                        else response_ts
                    )

                    latency_ms = _log_latency(
                        sent_ts,
                        response_ts,
                        False,
                        transcript,
                    )

                    await self._emit(
                        on_transcript,
                        {
                            "type": "partial",
                            "transcript": transcript,
                            "is_final": False,
                            "speech_final": False,
                            "latency_ms": latency_ms,
                            "language_mode": (
                                self.language_mode
                            ),
                            "languages": languages,
                            "stt_backend": "flux",
                            "model": self.model,
                            "turn_index": turn_index,
                        },
                    )

                    return

                # -------------------------------------------------------
                # EndOfTurn
                # -------------------------------------------------------

                if event == "EndOfTurn":

                    state[
                        "turn_completed"
                    ] = True

                    if transcript:

                        state[
                            "latest_transcript"
                        ] = transcript

                        await ensure_speech_started()

                        response_ts = time.time()

                        sent_ts = (
                            last_audio_sent_ts
                            if last_audio_sent_ts
                            is not None
                            else response_ts
                        )

                        latency_ms = (
                            _log_latency(
                                sent_ts,
                                response_ts,
                                True,
                                transcript,
                            )
                        )

                        await self._emit(
                            on_transcript,
                            {
                                "type": "final",
                                "transcript": transcript,
                                "is_final": True,
                                "speech_final": True,
                                "latency_ms": latency_ms,
                                "language_mode": (
                                    self.language_mode
                                ),
                                "languages": languages,
                                "stt_backend": "flux",
                                "model": self.model,
                                "turn_index": turn_index,
                                "trigger": getattr(
                                    message,
                                    "trigger",
                                    None,
                                ),

                                "end_of_turn_confidence": getattr(
                                    message,
                                    "end_of_turn_confidence",
                                    None,
                                ),

                            },
                        )

                    else:

                        await self._emit(
                            on_transcript,
                            {
                                "type": "silence",
                                "language_mode": (
                                    self.language_mode
                                ),
                                "stt_backend": "flux",
                                "model": self.model,
                                "turn_index": turn_index,
                            },
                        )

                    await self._emit(
                        on_transcript,
                        {
                            "type": "utterance_end",
                            "language_mode": (
                                self.language_mode
                            ),
                            "stt_backend": "flux",
                            "model": self.model,
                            "turn_index": turn_index,
                        },
                    )

                    return

            # -----------------------------------------------------------
            # Sequential message processor
            # -----------------------------------------------------------

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

            processor_task = asyncio.create_task(
                process_message_queue()
            )

            # -----------------------------------------------------------
            # SDK callbacks
            # -----------------------------------------------------------

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
                    "Flux streaming "
                    "connection opened."
                )

            def on_close(
                message,
            ) -> None:

                logger.info(
                    "Flux streaming "
                    "connection closed: %s",
                    message,
                )

            def on_error(
                error,
            ) -> None:

                logger.error(
                    "Flux streaming error: %s",
                    error,
                )

                message_queue.put_nowait(
                    {
                        "__agni_error__": str(
                            error
                        )
                    }
                )

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

            listener_task = asyncio.create_task(
                connection.start_listening()
            )

            # -----------------------------------------------------------
            # Flux strongly recommends ~80 ms audio chunks.
            #
            # Incoming Agni audio is PCM16 mono.
            #
            # bytes_per_second:
            #     sample_rate * 2 bytes/sample
            # -----------------------------------------------------------

            flux_chunk_bytes = int(
                sample_rate
                * 0.080
                * 2
            )

            audio_buffer = bytearray()

            async def send_flux_chunk(
                chunk: bytes,
            ) -> None:

                nonlocal last_audio_sent_ts

                if not chunk:
                    return

                last_audio_sent_ts = (
                    time.time()
                )

                await connection.send_media(
                    chunk
                )

            try:

                async for chunk in audio_gen:

                    if not chunk:
                        continue

                    audio_buffer.extend(
                        chunk
                    )

                    while (
                        len(audio_buffer)
                        >= flux_chunk_bytes
                    ):

                        flux_chunk = bytes(
                            audio_buffer[
                                :flux_chunk_bytes
                            ]
                        )

                        del audio_buffer[
                            :flux_chunk_bytes
                        ]

                        await send_flux_chunk(
                            flux_chunk
                        )

            finally:

                # Send any final remainder.
                if audio_buffer:

                    try:

                        await send_flux_chunk(
                            bytes(
                                audio_buffer
                            )
                        )

                    except Exception:

                        logger.debug(
                            "Unable to send final "
                            "Flux audio remainder.",
                            exc_info=True,
                        )

                # Allow already-received messages to be processed.
                try:

                    await asyncio.sleep(
                        min(
                            trailing_wait_s,
                            0.5,
                        )
                    )

                except asyncio.CancelledError:

                    pass

                try:

                    await message_queue.join()

                except asyncio.CancelledError:

                    pass

                # CloseStream does not guarantee EndOfTurn.
                #
                # Preserve an unfinished transcript for diagnostics.
                if (
                    state["latest_transcript"]
                    and not state[
                        "turn_completed"
                    ]
                ):

                    await self._emit(
                        on_transcript,
                        {
                            "type": (
                                "incomplete_speech"
                            ),
                            "transcript": (
                                state[
                                    "latest_transcript"
                                ]
                            ),
                            "language_mode": (
                                self.language_mode
                            ),
                            "stt_backend": "flux",
                            "model": self.model,
                        },
                    )

                try:

                    await (
                        connection
                        .send_close_stream()
                    )

                except Exception:

                    logger.debug(
                        "Flux close-stream failed.",
                        exc_info=True,
                    )

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

                if not listener_task.done():

                    try:

                        await asyncio.wait_for(
                            listener_task,
                            timeout=2.0,
                        )

                    except asyncio.TimeoutError:

                        listener_task.cancel()

                        await asyncio.gather(
                            listener_task,
                            return_exceptions=True,
                        )

                    except asyncio.CancelledError:

                        pass

                    except Exception:

                        logger.debug(
                            "Flux listener stopped "
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
                            "Flux listener stopped "
                            "with an exception.",
                            exc_info=True,
                        )

    # =======================================================================
    # Nova-3
    # =======================================================================

    async def _run_nova(
        self,
        audio_gen: AsyncIterator[bytes],
        on_transcript: Optional[
            TranscriptCallback
        ],
        trailing_wait_s: float,
        encoding: str,
        sample_rate: int,
        channels: int,
    ) -> None:
        """
        Existing Nova-3 streaming implementation.

        Used for Marathi and legacy multi mode.
        """

        client = AsyncDeepgramClient(
            api_key=self.api_key,
        )

        keyterms = [
            "Agni AI",
            "Python",
            "list",
            "tuple",
        ]

        async with client.listen.v1.connect(
            model="nova-3",
            language=self.language,
            encoding=encoding,
            sample_rate=sample_rate,
            channels=channels,
            interim_results=True,
            utterance_end_ms="1500",
            vad_events=True,
            endpointing=500,
            smart_format=True,
            keyterm=keyterms,
        ) as connection:

            last_audio_sent_ts: (
                float | None
            ) = None

            state = {
                "in_speech": False,
                "final_since_boundary": False,
                "last_partial": None,
                "last_partial_finalized": True,
            }

            message_queue: asyncio.Queue = (
                asyncio.Queue()
            )

            async def handle_message(
                message,
            ) -> None:

                nonlocal last_audio_sent_ts

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

                        await self._emit(
                            on_transcript,
                            {
                                "type": "error",
                                "message": error_message,
                                "stt_backend": "nova",
                                "model": "nova-3",
                            },
                        )

                    return

                message_type = getattr(
                    message,
                    "type",
                    "",
                )

                # =======================================================
                # Transcript result
                # =======================================================

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

                    transcript = (
                        getattr(
                            alternatives[0],
                            "transcript",
                            "",
                        )
                        or ""
                    ).strip()

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

                    await self._emit(
                        on_transcript,
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
                            "language_mode": (
                                self.language_mode
                            ),
                            "stt_backend": "nova",
                            "model": "nova-3",
                        },
                    )

                    return

                # =======================================================
                # Speech started
                # =======================================================

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

                        await self._emit(
                            on_transcript,
                            {
                                "type": (
                                    "speech_started"
                                ),
                                "language_mode": (
                                    self.language_mode
                                ),
                                "stt_backend": "nova",
                                "model": "nova-3",
                            },
                        )

                    return

                # =======================================================
                # Utterance end
                # =======================================================

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

                        await self._emit(
                            on_transcript,
                            {
                                "type": "silence",
                                "language_mode": (
                                    self.language_mode
                                ),
                                "stt_backend": "nova",
                                "model": "nova-3",
                            },
                        )

                    state[
                        "final_since_boundary"
                    ] = False

                    state[
                        "in_speech"
                    ] = False

                    await self._emit(
                        on_transcript,
                        {
                            "type": "utterance_end",
                            "language_mode": (
                                self.language_mode
                            ),
                            "stt_backend": "nova",
                            "model": "nova-3",
                        },
                    )

                    return

                if message_type == "Metadata":

                    logger.debug(
                        "Deepgram metadata: %s",
                        message,
                    )

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

            processor_task = asyncio.create_task(
                process_message_queue()
            )

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
                    "Nova-3 streaming "
                    "connection opened."
                )

            def on_close(
                message,
            ) -> None:

                logger.info(
                    "Nova-3 streaming "
                    "connection closed: %s",
                    message,
                )

            def on_error(
                error,
            ) -> None:

                logger.error(
                    "Nova-3 streaming error: %s",
                    error,
                )

                message_queue.put_nowait(
                    {
                        "__agni_error__": str(
                            error
                        )
                    }
                )

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

            listener_task = asyncio.create_task(
                connection.start_listening()
            )

            try:

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

                try:

                    await connection.send_finalize()

                except Exception:

                    logger.debug(
                        "Deepgram finalize failed.",
                        exc_info=True,
                    )

                try:

                    await asyncio.sleep(
                        trailing_wait_s
                    )

                except asyncio.CancelledError:

                    pass

                try:

                    await message_queue.join()

                except asyncio.CancelledError:

                    pass

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

                    await self._emit(
                        on_transcript,
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
                                self.language_mode
                            ),
                            "stt_backend": "nova",
                            "model": "nova-3",
                        },
                    )

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

                if not listener_task.done():

                    try:

                        await asyncio.wait_for(
                            listener_task,
                            timeout=2.0,
                        )

                    except asyncio.TimeoutError:

                        listener_task.cancel()

                        await asyncio.gather(
                            listener_task,
                            return_exceptions=True,
                        )

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

    # =======================================================================
    # Standalone microphone
    # =======================================================================

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

    # =======================================================================
    # Standalone WAV
    # =======================================================================

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

    # =======================================================================
    # External audio source
    # =======================================================================

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
    )

    parser.add_argument(
        "--path",
    )

    parser.add_argument(
        "--language",
        default=None,
        help=(
            "english, hindi, hinglish, "
            "marathi, or multi"
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
