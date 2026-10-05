"""

Agni AI - Integrated LiveKit Voice Pipeline



Realtime flow:



    Microphone

        ↓

    LiveKit

        ↓

    Streaming STT

        ↓

    Deepgram

        ↓

    Completed User Utterance

        ↓

    OpenAI LLM

        ↓

    ElevenLabs TTS

        ↓

    LiveKit voice-output

        ↓

    Caller / Listener



Important behavior:



- Deepgram connects only after a microphone track exists.

- Only completed utterances are sent to OpenAI.

- Deepgram speech_final is used as the fast turn-completion signal.

- Deepgram utterance_end remains as a fallback.

- OpenAI text can be forwarded to TTS while the LLM is still generating.

- Microphone audio continues to reach STT while Agni is speaking.

- User speech can interrupt the current AI response.

- Response latency is measured without making extra API calls.

"""
from __future__ import annotations
import asyncio
import contextlib
import os
import time
import numpy as np
from dotenv import load_dotenv
from livekit import api, rtc
from scipy.signal import resample_poly
from app.voice.elevenlabs_tts_provider import (
    ElevenLabsTTSProvider,
)
from app.voice.livekit_audio_output import (
    LiveKitAudioOutput,
)
from app.voice.openai_llm_provider import (
    OpenAILLMProvider,
)
from app.voice.stt_stream_adapter import (
    STTStreamAdapter,
)

load_dotenv(
    ".env.local",
    override=True,
)

# ---------------------------------------------------------------------------
# LiveKit
# ---------------------------------------------------------------------------

ROOM_NAME = (
    os.getenv("AGNI_SESSION_ROOM_NAME")
    or os.getenv(
        "LIVEKIT_ROOM_NAME",
        "agni-ai-voice-poc",
    )
)

PARTICIPANT_IDENTITY = (
    os.getenv("AGNI_SESSION_PARTICIPANT_IDENTITY")
    or os.getenv(
        "LIVEKIT_SUBSCRIBER_IDENTITY",
        "agni-audio-subscriber",
    )
)

# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

STT_SAMPLE_RATE = 16_000

STT_STREAM_ENDPOINT = os.getenv(
    "STT_STREAM_ENDPOINT",
    "ws://127.0.0.1:8000/api/v1/stt/stream",
)
# User-facing language selection.
#
# Examples:
#
#   AGNI_STT_LANGUAGE=english
#   AGNI_STT_LANGUAGE=hindi
#   AGNI_STT_LANGUAGE=hinglish
#   AGNI_STT_LANGUAGE=marathi
#
# Canonical Deepgram values also remain supported:
#
#   en-IN
#   hi
#   mr
#   multi
#
# Agni STT routing:
#
#   English  -> Flux / flux-general-en
#   Hindi    -> Flux / flux-general-multi
#   Hinglish -> Flux / flux-general-multi
#   Marathi  -> Nova-3 / mr
#
# "multi" remains available as a legacy/debug mode.

REQUESTED_STT_LANGUAGE = (
    os.getenv("AGNI_SESSION_LANGUAGE")
    or os.getenv(
        "AGNI_STT_LANGUAGE",
        "multi",
    )
).strip()

SYSTEM_PROMPT = os.getenv("AGNI_SESSION_SYSTEM_PROMPT") or None

STT_LANGUAGE_ALIASES = {
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

STT_LANGUAGE = STT_LANGUAGE_ALIASES.get(
    REQUESTED_STT_LANGUAGE.lower(),
    REQUESTED_STT_LANGUAGE,
)

SUPPORTED_STT_LANGUAGES = {
    "english",
    "hindi",
    "hinglish",
    "marathi",
    "multi",
}
if STT_LANGUAGE not in SUPPORTED_STT_LANGUAGES:
    raise RuntimeError(
        "Unsupported AGNI_STT_LANGUAGE: "
        f"{REQUESTED_STT_LANGUAGE}. "
        "Use English, Hindi, Hinglish, Marathi, "
        "or en-IN, hi, mr, multi."
    )

separator = (
    "&"
    if "?" in STT_STREAM_ENDPOINT
    else "?"
)

STT_STREAM_ENDPOINT_WITH_LANGUAGE = (
    f"{STT_STREAM_ENDPOINT}"
    f"{separator}"
    f"language={STT_LANGUAGE}"
)

# ---------------------------------------------------------------------------
# TTS / barge-in
# ---------------------------------------------------------------------------

TTS_SAMPLE_RATE = 16_000

TTS_CHANNELS = 1

# ---------------------------------------------------------------------------
# LiveKit authentication
# ---------------------------------------------------------------------------


def create_access_token() -> str:
    api_key = os.getenv(
        "LIVEKIT_API_KEY"
    )
    api_secret = os.getenv(
        "LIVEKIT_API_SECRET"
    )

    if not api_key or not api_secret:
        raise RuntimeError(
            "LIVEKIT_API_KEY or LIVEKIT_API_SECRET "
            "is missing from .env.local"
        )
    return (
        api.AccessToken(
            api_key,
            api_secret,
        )
        .with_identity(
            PARTICIPANT_IDENTITY
        )
        .with_name(
            "Agni AI Voice Agent"
        )
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=ROOM_NAME,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------


def convert_audio_frame_to_stt_format(
    audio_frame: rtc.AudioFrame,
) -> bytes:
    """

    Convert LiveKit PCM16 audio to:



        PCM16

        16 kHz

        mono

    """
    input_sample_rate = int(
        audio_frame.sample_rate
    )
    input_channels = int(
        audio_frame.num_channels
    )
    pcm_bytes = bytes(
        audio_frame.data
    )

    if not pcm_bytes:
        return b""
    samples = np.frombuffer(
        pcm_bytes,
        dtype=np.int16,
    )

    # ---------------------------------------------------------------
    # Multi-channel -> mono
    # ---------------------------------------------------------------

    if input_channels > 1:
        samples = samples.reshape(
            -1,
            input_channels,
        ).astype(
            np.int32
        )
        samples = np.mean(
            samples,
            axis=1,
        )
        samples = np.clip(
            samples,
            -32768,
            32767,
        ).astype(
            np.int16
        )

    # ---------------------------------------------------------------
    # Resample -> 16 kHz
    # ---------------------------------------------------------------

    if input_sample_rate != STT_SAMPLE_RATE:
        samples_float = samples.astype(
            np.float32
        )
        resampled = resample_poly(
            samples_float,
            STT_SAMPLE_RATE,
            input_sample_rate,
        )
        samples = np.clip(
            resampled,
            -32768,
            32767,
        ).astype(
            np.int16
        )
    return samples.tobytes()

# ---------------------------------------------------------------------------
# AI response worker
# ---------------------------------------------------------------------------


async def process_ai_responses(
    response_queue: asyncio.Queue[
        tuple[
            str,
            float,
            float | None,
        ]
    ],
    llm_provider: OpenAILLMProvider,
    tts_provider: ElevenLabsTTSProvider,
    audio_output: LiveKitAudioOutput,
    ai_speaking: asyncio.Event,
    interrupt_event: asyncio.Event,
) -> None:
    """

    Process completed utterances sequentially.



    Optimized flow:



        completed utterance

                ↓

        OpenAI streaming text

                ↓

        asyncio text queue

                ↓

        ElevenLabs streaming-text TTS

                ↓

        LiveKit audio



    OpenAI and ElevenLabs can therefore work concurrently.



    Only one OpenAI request is made for each completed

    user utterance.



    A confirmed user interruption stops the current response.

    """

    while True:
        (
            transcript,
            utterance_ready_at,
            last_final_at,
        ) = await response_queue.get()
        llm_task: asyncio.Task | None = None

        try:
            transcript = transcript.strip()

            if not transcript:
                continue

            # -------------------------------------------------------
            # New active AI turn
            # -------------------------------------------------------

            interrupt_event.clear()
            # Reset any hard-interrupt state left by the
            # previous response.
            audio_output.begin_response()
            ai_speaking.set()
            worker_started_at = (
                time.perf_counter()
            )
            print()
            print("=" * 72)
            print("AGNI AI - VOICE RESPONSE")
            print("=" * 72)
            print()
            print("User:")
            print(transcript)
            print()
            print(
                "Generating AI response "
                "and streaming speech..."
            )

            # -------------------------------------------------------
            # LLM -> TTS streaming bridge
            # -------------------------------------------------------

            llm_started_at = (
                time.perf_counter()
            )
            first_llm_chunk_at: (
                float | None
            ) = None
            llm_completed_at: (
                float | None
            ) = None
            response_chunks: list[str] = []
            text_queue: asyncio.Queue[
                str | None
            ] = asyncio.Queue()

            # -------------------------------------------------------
            # OpenAI producer
            # -------------------------------------------------------

            async def produce_llm_text() -> None:
                nonlocal first_llm_chunk_at
                nonlocal llm_completed_at
                llm_stream = (
                    llm_provider.stream_response(
                        transcript
                    )
                )

                try:
                    async with contextlib.aclosing(
                        llm_stream
                    ):

                        async for text_chunk in llm_stream:

                            if interrupt_event.is_set():
                                return

                            if not text_chunk:
                                continue

                            if first_llm_chunk_at is None:
                                first_llm_chunk_at = (
                                    time.perf_counter()
                                )
                            response_chunks.append(
                                text_chunk
                            )
                            await text_queue.put(
                                text_chunk
                            )

                finally:
                    llm_completed_at = (
                        time.perf_counter()
                    )
                    # None tells the TTS-side iterator
                    # that OpenAI has finished.
                    await text_queue.put(
                        None
                    )

            # -------------------------------------------------------
            # Queue -> TTS async text stream
            # -------------------------------------------------------

            async def llm_text_stream():

                while True:
                    item = await text_queue.get()

                    try:
                        if (
                            item is None
                            or interrupt_event.is_set()
                        ):
                            return
                        yield item

                    finally:
                        text_queue.task_done()
            # Start OpenAI generation.
            llm_task = asyncio.create_task(
                produce_llm_text()
            )

            # -------------------------------------------------------
            # Streaming TTS
            # -------------------------------------------------------

            tts_started_at = (
                time.perf_counter()
            )
            first_tts_chunk_at: (
                float | None
            ) = None
            chunk_count = 0
            total_audio_bytes = 0
            tts_stream = (
                tts_provider
                .synthesize_streaming_text(
                    llm_text_stream()
                )
            )

            async for audio_chunk in tts_stream:

                if interrupt_event.is_set():
                    break

                if first_tts_chunk_at is None:
                    first_tts_chunk_at = (
                        time.perf_counter()
                    )
                chunk_count += 1
                total_audio_bytes += len(
                    audio_chunk.data
                )
                await audio_output.send_chunk(
                    audio_chunk
                )
            tts_generation_completed_at = (
                time.perf_counter()
            )

            # -------------------------------------------------------
            # Confirmed barge-in during generation
            # -------------------------------------------------------

            if interrupt_event.is_set():
                if (
                    llm_task is not None
                    and not llm_task.done()
                ):
                    llm_task.cancel()
                    await asyncio.gather(
                        llm_task,
                        return_exceptions=True,
                    )
                llm_task = None
                # The async-for loop has already stopped, so the
                # ElevenLabs generator can now be closed cleanly.
                await tts_stream.aclose()
                # Clear once more in case capture_frame was already
                # in progress when hard interruption happened.
                audio_output.interrupt()
                print()
                print(
                    "[BARGE-IN] "
                    "Current AI response cancelled."
                )
                continue

            # -------------------------------------------------------
            # Make sure OpenAI completed successfully
            # -------------------------------------------------------

            await llm_task
            llm_task = None

            if interrupt_event.is_set():
                print()
                print(
                    "[BARGE-IN] "
                    "Current AI response cancelled."
                )
                continue

            if llm_completed_at is None:
                llm_completed_at = (
                    time.perf_counter()
                )
            response_text = "".join(
                response_chunks
            ).strip()
            if not response_text:
                print(
                    "LLM returned an empty response."
                )
                continue
            print()
            print("Agni AI:")
            print(response_text)
            print()

            # -------------------------------------------------------
            # Wait for LiveKit playback
            # -------------------------------------------------------

            await audio_output.wait_for_playout()

            if interrupt_event.is_set():
                print()
                print(
                    "[BARGE-IN] "
                    "AI playback interrupted."
                )
                continue
            playback_completed_at = (
                time.perf_counter()
            )
            print(
                f"TTS chunks: "
                f"{chunk_count}"
            )
            print(
                f"TTS audio bytes: "
                f"{total_audio_bytes}"
            )
            print(
                "AI voice response completed."
            )

            # -------------------------------------------------------
            # Latency report
            # -------------------------------------------------------

            print()
            print("-" * 72)
            print("RESPONSE LATENCY")
            print("-" * 72)

            # -------------------------------------------------------
            # STT -> turn ready
            # -------------------------------------------------------

            if last_final_at is not None:
                final_to_turn_ready = (
                    utterance_ready_at
                    - last_final_at
                )
                print(
                    "Final transcript -> "
                    "turn ready: "
                    f"{final_to_turn_ready:.3f} s"
                )

            # -------------------------------------------------------
            # Queue delay
            # -------------------------------------------------------

            queue_delay = (
                worker_started_at
                - utterance_ready_at
            )
            print(
                "Pipeline queue delay: "
                f"{queue_delay:.3f} s"
            )

            # -------------------------------------------------------
            # LLM latency
            # -------------------------------------------------------

            if first_llm_chunk_at is not None:
                llm_first_chunk = (
                    first_llm_chunk_at
                    - llm_started_at
                )
                print(
                    "LLM first text chunk: "
                    f"{llm_first_chunk:.3f} s"
                )
            llm_total = (
                llm_completed_at
                - llm_started_at
            )
            print(
                "LLM complete response: "
                f"{llm_total:.3f} s"
            )

            # -------------------------------------------------------
            # First AI audio
            # -------------------------------------------------------

            if first_tts_chunk_at is not None:
                response_start_latency = (
                    first_tts_chunk_at
                    - utterance_ready_at
                )
                print(
                    "Turn ready -> "
                    "first AI audio: "
                    f"{response_start_latency:.3f} s"
                )

                if first_llm_chunk_at is not None:
                    llm_to_audio = (
                        first_tts_chunk_at
                        - first_llm_chunk_at
                    )
                    print(
                        "First LLM text -> "
                        "first AI audio: "
                        f"{llm_to_audio:.3f} s"
                    )
                if (
                    llm_completed_at is not None
                    and first_tts_chunk_at
                    < llm_completed_at
                ):
                    overlap = (
                        llm_completed_at
                        - first_tts_chunk_at
                    )
                    print(
                        "TTS started before LLM "
                        "completed by: "
                        f"{overlap:.3f} s"
                    )

                else:
                    print(
                        "TTS started after LLM "
                        "completed."
                    )

            # -------------------------------------------------------
            # TTS session duration
            # -------------------------------------------------------

            tts_generation_time = (
                tts_generation_completed_at
                - tts_started_at
            )
            print(
                "TTS streaming session: "
                f"{tts_generation_time:.3f} s"
            )

            # -------------------------------------------------------
            # Total turn response duration
            # -------------------------------------------------------

            total_response_time = (
                playback_completed_at
                - utterance_ready_at
            )
            print(
                "Turn ready -> "
                "playback complete: "
                f"{total_response_time:.3f} s"
            )
            print("-" * 72)
            print()

        # -----------------------------------------------------------
        # Cancellation
        # -----------------------------------------------------------

        except asyncio.CancelledError:
            if (
                llm_task is not None
                and not llm_task.done()
            ):
                llm_task.cancel()
                await asyncio.gather(
                    llm_task,
                    return_exceptions=True,
                )
            raise

        # -----------------------------------------------------------
        # Pipeline error
        # -----------------------------------------------------------

        except Exception as exc:
            if (
                llm_task is not None
                and not llm_task.done()
            ):
                llm_task.cancel()
                await asyncio.gather(
                    llm_task,
                    return_exceptions=True,
                )
            print()
            print(
                f"[VOICE PIPELINE ERROR] "
                f"{exc}"
            )

        finally:
            ai_speaking.clear()
            response_queue.task_done()

# ---------------------------------------------------------------------------
# STT event handling
# ---------------------------------------------------------------------------


async def receive_stt_events(
    stt_adapter: STTStreamAdapter,
    response_queue: asyncio.Queue[
        tuple[
            str,
            float,
            float | None,
        ]
    ],
    ai_speaking: asyncio.Event,
    interrupt_event: asyncio.Event,
    audio_output: LiveKitAudioOutput,
) -> None:
    """
    Receive normalized STT events and dispatch completed user turns.

    Flux:
        speech_started is emitted from Flux StartOfTurn.
        If Agni is currently speaking, this immediately triggers
        a hard barge-in.

    Nova:
        speech_started is raw VAD and does not interrupt by itself.
        A non-empty partial or final transcript confirms barge-in.

    Completed user turns are dispatched using speech_final as the
    fast path, with utterance_end retained as a fallback.
    """
    final_transcript_parts: list[str] = []
    last_final_at: float | None = None
    # Prevent utterance_end from sending the same user
    # turn to OpenAI after speech_final already dispatched it.
    turn_dispatched = False

    # ------------------------------------------------------------------
    # Confirmed barge-in
    # ------------------------------------------------------------------

    def request_barge_in(
        reason: str,
    ) -> None:

        if not ai_speaking.is_set():
            return

        if interrupt_event.is_set():
            return
        print()
        print(
            f"[BARGE-IN] User interrupted Agni "
            f"({reason})"
        )
        # Tell the current LLM/TTS response to stop.
        interrupt_event.set()
        # Hard interruption:
        #
        # - reject remaining old-response chunks
        # - clear already queued LiveKit audio
        audio_output.interrupt()

    # ------------------------------------------------------------------
    # Completed utterance dispatcher
    # ------------------------------------------------------------------

    async def dispatch_utterance(
        trigger: str,
    ) -> None:
        nonlocal last_final_at
        nonlocal turn_dispatched
        utterance_ready_at = (
            time.perf_counter()
        )
        final_utterance = " ".join(
            final_transcript_parts
        ).strip()
        final_transcript_parts.clear()

        if not final_utterance:
            last_final_at = None
            return
        print()
        print(
            f"[UTTERANCE - {trigger}] "
            f"{final_utterance}"
        )
        await response_queue.put(
            (
                final_utterance,
                utterance_ready_at,
                last_final_at,
            )
        )
        last_final_at = None
        turn_dispatched = True

    # ------------------------------------------------------------------
    # STT event loop
    # ------------------------------------------------------------------

    while True:
        event = await stt_adapter.receive_event()
        event_type = event.get(
            "type",
            "unknown",
        )

        # -------------------------------------------------------
        # Partial
        # -------------------------------------------------------

        if event_type == "partial":
            transcript = event.get(
                "transcript",
                "",
            ).strip()
            if (
                transcript
                and ai_speaking.is_set()
            ):
                request_barge_in(
                    "partial transcript"
                )
            latency = event.get(
                "latency_ms",
                0,
            )
            print(
                f"[PARTIAL] {transcript}"
                f"  ({latency:.2f} ms)"
            )

        # -------------------------------------------------------
        # Final
        # -------------------------------------------------------

        elif event_type == "final":
            transcript = event.get(
                "transcript",
                "",
            ).strip()
            if transcript:
                final_transcript_parts.append(
                    transcript
                )
                last_final_at = time.perf_counter()

            if (
                transcript
                and ai_speaking.is_set()
            ):
                request_barge_in(
                    "final transcript"
                )

            latency = event.get(
                "latency_ms",
                0,
            )

            speech_final = bool(
                event.get(
                    "speech_final",
                    False,
                )
            )

            trigger = event.get(
                "trigger"
            )

            end_of_turn_confidence = event.get(
                "end_of_turn_confidence"
            )

            speech_final_label = (
                " [SPEECH FINAL]"
                if speech_final
                else ""
            )

            print(
                f"[FINAL] {transcript}"
                f"  ({latency:.2f} ms)"
                f"{speech_final_label}"
            )

            if trigger is not None:
                print(
                    "[TURN END] "
                    f"trigger={trigger}, "
                    f"confidence={end_of_turn_confidence}"
                )

            # ---------------------------------------------------
            # FAST PATH
            # ---------------------------------------------------

            if (
                speech_final
                and final_transcript_parts
                and not turn_dispatched
            ):
                print(
                    "[VAD] Speech endpoint detected"
                )
                await dispatch_utterance(
                    "speech_final"
                )

        # -------------------------------------------------------
        # Speech started
        # -------------------------------------------------------

        elif event_type == "speech_started":

            turn_dispatched = False

            stt_backend = event.get(
                "stt_backend",
                "",
            )

            print(
                "[VAD] Speech started"
            )

            # Flux speech_started comes from the semantic
            # StartOfTurn event.
            #
            # StartOfTurn may contain an empty transcript,
            # so transcript text is intentionally not required
            # before interrupting the active AI response.
            if (
                ai_speaking.is_set()
                and stt_backend == "flux"
            ):

                request_barge_in(
                    "Flux StartOfTurn"
                )

            # Nova-3 speech_started is raw VAD.
            #
            # Do not interrupt here for Marathi/Nova.
            # A non-empty partial/final transcript confirms
            # the user's speech instead.

        # -------------------------------------------------------
        # Utterance end
        # -------------------------------------------------------

        elif event_type == "utterance_end":
            print(
                "[VAD] Utterance ended"
            )

            # ---------------------------------------------------
            # speech_final already dispatched this turn.
            # ---------------------------------------------------

            if turn_dispatched:
                final_transcript_parts.clear()
                last_final_at = None
                turn_dispatched = False
                print(
                    "[VAD] Utterance already "
                    "processed via speech_final"
                )
                continue

            # ---------------------------------------------------
            # FALLBACK
            # ---------------------------------------------------

            if final_transcript_parts:
                await dispatch_utterance(
                    "utterance_end fallback"
                )

            else:
                last_final_at = None

        # -------------------------------------------------------
        # Silence
        # -------------------------------------------------------

        elif event_type == "silence":
            final_transcript_parts.clear()
            last_final_at = None
            turn_dispatched = False
            print(
                "[VAD] Silence detected"
            )

        # -------------------------------------------------------
        # Incomplete speech
        # -------------------------------------------------------

        elif event_type == "incomplete_speech":
            transcript = event.get(
                "transcript",
                "",
            )
            print(
                f"[STT] Incomplete speech: "
                f"{transcript}"
            )

        # -------------------------------------------------------
        # Error
        # -------------------------------------------------------

        elif event_type == "error":
            print(
                f"[STT ERROR] "
                f"{event.get('message', '')}"
            )

        else:
            print(
                f"[STT EVENT] {event}"
            )

# ---------------------------------------------------------------------------
# LiveKit microphone handling
# ---------------------------------------------------------------------------


async def consume_audio_track(
    track: rtc.Track,
    participant_identity: str,
    stt_adapter: STTStreamAdapter,
) -> None:
    """
    Stream microphone audio continuously to STT.

    The microphone remains active while Agni is generating
    or speaking so Deepgram can detect user barge-in.
    """

    if track.kind != rtc.TrackKind.KIND_AUDIO:
        return
    print()
    print(
        f"Receiving audio from "
        f"'{participant_identity}'"
    )
    audio_stream = rtc.AudioStream(
        track
    )
    frame_count = 0
    total_audio_bytes = 0

    try:

        async for audio_frame_event in audio_stream:
            audio_frame = (
                audio_frame_event.frame
            )
            frame_count += 1
            stt_audio = (
                convert_audio_frame_to_stt_format(
                    audio_frame
                )
            )

            if not stt_audio:
                continue
            total_audio_bytes += len(
                stt_audio
            )
            # Always send the real microphone signal.
            #
            # Deepgram must continue hearing the user while
            # Agni speaks so barge-in can be detected.
            await stt_adapter.send_audio(
                stt_audio
            )

            if frame_count % 100 == 0:
                print(
                    f"Streaming audio frames: "
                    f"{frame_count}"
                )

    finally:
        await audio_stream.aclose()
        print()
        print(
            f"Audio stream ended. "
            f"Frames received: {frame_count}. "
            f"STT audio bytes sent: "
            f"{total_audio_bytes}"
        )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """
    Run the integrated Agni AI voice pipeline.

    LiveKit is connected first.

    Deepgram is not connected until a real microphone
    track has been received.
    """
    livekit_url = os.getenv(
        "LIVEKIT_URL"
    )

    if not livekit_url:
        raise RuntimeError(
            "LIVEKIT_URL is missing from "
            ".env.local"
        )
    print()
    print("=" * 72)
    print(
        "AGNI AI - INTEGRATED VOICE PIPELINE"
    )
    print("=" * 72)
    print()

    # ---------------------------------------------------------------
    # Providers
    # ---------------------------------------------------------------

    print(
        "Initializing OpenAI LLM provider..."
    )
    llm_provider = OpenAILLMProvider(
        instructions=SYSTEM_PROMPT,
        response_language=REQUESTED_STT_LANGUAGE,
    )
    print(
        f"LLM ready: "
        f"{llm_provider.model}"
    )
    print(
        "LLM response language: "
        f"{llm_provider.response_language or 'auto'}"
    )
    print(
        "Initializing ElevenLabs TTS provider..."
    )
    tts_provider = ElevenLabsTTSProvider()
    print(
        "TTS provider ready."
    )

    # ---------------------------------------------------------------
    # LiveKit
    # ---------------------------------------------------------------

    room = rtc.Room()
    audio_output = LiveKitAudioOutput(
        sample_rate=TTS_SAMPLE_RATE,
        channels=TTS_CHANNELS,
    )
    response_queue: asyncio.Queue[
        tuple[
            str,
            float,
            float | None,
        ]
    ] = asyncio.Queue()
    ai_speaking = asyncio.Event()
    # Set when the user confirms an interruption of the
    # current AI response.
    interrupt_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    microphone_track_future: (
        asyncio.Future[
            tuple[
                rtc.Track,
                str,
            ]
        ]
    ) = loop.create_future()
    stt_adapter: (
        STTStreamAdapter
        | None
    ) = None
    audio_task: (
        asyncio.Task
        | None
    ) = None
    stt_event_task: (
        asyncio.Task
        | None
    ) = None
    response_task: (
        asyncio.Task
        | None
    ) = None

    # ---------------------------------------------------------------
    # LiveKit track callback
    # ---------------------------------------------------------------

    @room.on(
        "track_subscribed"
    )
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if (
            track.kind
            != rtc.TrackKind.KIND_AUDIO
        ):
            return

        if publication.name == "voice-output":
            return

        if microphone_track_future.done():
            return
        print()
        print(
            f"Subscribed to track "
            f"'{publication.name}' "
            f"from "
            f"'{participant.identity}'"
        )
        microphone_track_future.set_result(
            (
                track,
                participant.identity,
            )
        )

    try:

        # -----------------------------------------------------------
        # Connect LiveKit
        # -----------------------------------------------------------

        print(
            "Connecting to LiveKit..."
        )
        await room.connect(
            livekit_url,
            create_access_token(),
        )
        print(
            "Connected to LiveKit."
        )
        print(
            f"Room: {room.name}"
        )
        print(
            "Participant: "
            f"{room.local_participant.identity}"
        )

        # -----------------------------------------------------------
        # Publish AI voice output
        # -----------------------------------------------------------

        publication = await audio_output.publish(
            room.local_participant
        )
        print(
            "Published AI voice track: "
            f"{publication.sid}"
        )
        print()
        print(
            "Waiting for microphone track..."
        )

        # -----------------------------------------------------------
        # Wait for microphone
        # -----------------------------------------------------------

        (
            microphone_track,
            microphone_identity,
        ) = await microphone_track_future
        print(
            "Microphone track ready."
        )

        # -----------------------------------------------------------
        # Connect STT only now
        # -----------------------------------------------------------

        stt_adapter = STTStreamAdapter(
            endpoint=STT_STREAM_ENDPOINT_WITH_LANGUAGE
        )
        print(
            "Connecting to streaming STT..."
        )
        await stt_adapter.connect()
        print(
            "Streaming STT connected."
        )

        # -----------------------------------------------------------
        # Workers
        # -----------------------------------------------------------

        stt_event_task = asyncio.create_task(
            receive_stt_events(
                stt_adapter,
                response_queue,
                ai_speaking,
                interrupt_event,
                audio_output,
            )
        )
        response_task = asyncio.create_task(
            process_ai_responses(
                response_queue,
                llm_provider,
                tts_provider,
                audio_output,
                ai_speaking,
                interrupt_event,
            )
        )
        audio_task = asyncio.create_task(
            consume_audio_track(
                microphone_track,
                microphone_identity,
                stt_adapter,
            )
        )
        print()
        print(
            f"Streaming STT endpoint: "
            f"{STT_STREAM_ENDPOINT_WITH_LANGUAGE}"
        )
        print(
            f"Requested STT language: "
            f"{REQUESTED_STT_LANGUAGE}"
        )
        print(
            f"STT routing mode: "
            f"{STT_LANGUAGE}"
        )
        print()
        print(
            "Voice pipeline ready."
        )
        print(
            "Only completed utterances are "
            "sent to OpenAI."
        )
        print(
            "OpenAI text is streamed "
            "toward ElevenLabs in real time."
        )
        await asyncio.Event().wait()

    finally:
        print()
        print(
            "Stopping Agni AI "
            "voice pipeline..."
        )

        # -----------------------------------------------------------
        # Stop current AI response first
        # -----------------------------------------------------------

        #
        # Stop any active AI playback and reject remaining
        # chunks from the interrupted response.
        interrupt_event.set()
        audio_output.interrupt()

        if response_task is not None:
            response_task.cancel()
            await asyncio.gather(
                response_task,
                return_exceptions=True,
            )

        # -----------------------------------------------------------
        # Stop STT event processing
        # -----------------------------------------------------------

        if stt_event_task is not None:
            stt_event_task.cancel()
            await asyncio.gather(
                stt_event_task,
                return_exceptions=True,
            )

        # -----------------------------------------------------------
        # Stop microphone streaming
        # -----------------------------------------------------------

        if audio_task is not None:
            audio_task.cancel()
            await asyncio.gather(
                audio_task,
                return_exceptions=True,
            )

        # -----------------------------------------------------------
        # Close external streams
        # -----------------------------------------------------------

        if stt_adapter is not None:
            await stt_adapter.close()
        await audio_output.close()
        await room.disconnect()
        print(
            "Disconnected from LiveKit."
        )
if __name__ == "__main__":

    try:
        asyncio.run(
            main()
        )

    except KeyboardInterrupt:
        print()
        print(
            "Agni AI voice pipeline "
            "stopped by user."
        )

    except Exception as exc:
        print()
        print(
            f"Voice pipeline error: "
            f"{exc}"
        )
