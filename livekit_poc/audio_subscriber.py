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
- While AI is generating/speaking, microphone audio is replaced
  with silence before being sent to STT.
- Response latency is measured without making extra API calls.
"""

from __future__ import annotations

import asyncio
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

ROOM_NAME = os.getenv(
    "LIVEKIT_ROOM_NAME",
    "agni-ai-voice-poc",
)

PARTICIPANT_IDENTITY = os.getenv(
    "LIVEKIT_SUBSCRIBER_IDENTITY",
    "agni-audio-subscriber",
)


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

STT_SAMPLE_RATE = 16_000
STT_CHANNELS = 1

STT_STREAM_ENDPOINT = os.getenv(
    "STT_STREAM_ENDPOINT",
    "ws://127.0.0.1:8000/api/v1/stt/stream",
)


# ---------------------------------------------------------------------------
# TTS
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
) -> None:
    """
    Process completed utterances sequentially.

    Timing begins when STT declares that the user's
    utterance has ended.

    Only one OpenAI request is made for each completed
    user utterance.
    """

    while True:
        (
            transcript,
            utterance_ready_at,
            last_final_at,
        ) = await response_queue.get()

        try:
            transcript = transcript.strip()

            if not transcript:
                continue

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

            # -------------------------------------------------------
            # LLM
            # -------------------------------------------------------

            print(
                "Generating AI response..."
            )

            llm_started_at = (
                time.perf_counter()
            )

            first_llm_chunk_at: (
                float | None
            ) = None

            response_chunks: list[str] = []

            # Use the provider's existing streaming method.
            #
            # This is still ONE OpenAI request.
            # We collect the chunks so ElevenLabs receives the
            # full response once.
            async for text_chunk in (
                llm_provider.stream_response(
                    transcript
                )
            ):
                if first_llm_chunk_at is None:
                    first_llm_chunk_at = (
                        time.perf_counter()
                    )

                response_chunks.append(
                    text_chunk
                )

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
            # TTS
            # -------------------------------------------------------

            print(
                "Generating and streaming speech..."
            )

            tts_started_at = (
                time.perf_counter()
            )

            first_tts_chunk_at: (
                float | None
            ) = None

            chunk_count = 0
            total_audio_bytes = 0

            async for audio_chunk in (
                tts_provider.synthesize(
                    response_text
                )
            ):
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

            # Keep microphone suppression enabled until
            # every generated audio frame has played out.
            await (
                audio_output
                .audio_source
                .wait_for_playout()
            )

            playback_completed_at = (
                time.perf_counter()
            )

            print(
                f"TTS chunks: {chunk_count}"
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

            if last_final_at is not None:
                final_to_utterance = (
                    utterance_ready_at
                    - last_final_at
                )

                print(
                    "Final transcript -> "
                    "utterance end: "
                    f"{final_to_utterance:.3f} s"
                )

            queue_delay = (
                worker_started_at
                - utterance_ready_at
            )

            print(
                "Pipeline queue delay: "
                f"{queue_delay:.3f} s"
            )

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

            if first_tts_chunk_at is not None:
                tts_first_chunk = (
                    first_tts_chunk_at
                    - tts_started_at
                )

                print(
                    "ElevenLabs first audio chunk: "
                    f"{tts_first_chunk:.3f} s"
                )

                response_start_latency = (
                    first_tts_chunk_at
                    - utterance_ready_at
                )

                print(
                    "Utterance end -> "
                    "first AI audio: "
                    f"{response_start_latency:.3f} s"
                )

            tts_generation_time = (
                tts_generation_completed_at
                - tts_started_at
            )

            print(
                "TTS generation/streaming: "
                f"{tts_generation_time:.3f} s"
            )

            total_response_time = (
                playback_completed_at
                - utterance_ready_at
            )

            print(
                "Utterance end -> "
                "playback complete: "
                f"{total_response_time:.3f} s"
            )

            print("-" * 72)
            print()

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            print()
            print(
                f"[VOICE PIPELINE ERROR] {exc}"
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
) -> None:
    """
    Receive STT transcript and VAD events.

    Final transcript pieces are collected until Deepgram
    sends utterance_end.

    Only the completed utterance is queued for OpenAI.
    """

    final_transcript_parts: list[str] = []

    last_final_at: float | None = None

    while True:
        event = await stt_adapter.receive_event()

        event_type = event.get(
            "type",
            "unknown",
        )

        # -----------------------------------------------------------
        # Ignore recognition while AI is active
        # -----------------------------------------------------------

        if ai_speaking.is_set():
            if event_type in (
                "utterance_end",
                "silence",
            ):
                final_transcript_parts.clear()
                last_final_at = None

            if event_type == "error":
                print(
                    f"[STT ERROR] "
                    f"{event.get('message', '')}"
                )

            continue

        # -----------------------------------------------------------
        # Partial
        # -----------------------------------------------------------

        if event_type == "partial":
            transcript = event.get(
                "transcript",
                "",
            )

            latency = event.get(
                "latency_ms",
                0,
            )

            print(
                f"[PARTIAL] {transcript}"
                f"  ({latency:.2f} ms)"
            )

        # -----------------------------------------------------------
        # Final
        # -----------------------------------------------------------

        elif event_type == "final":
            transcript = event.get(
                "transcript",
                "",
            ).strip()

            latency = event.get(
                "latency_ms",
                0,
            )

            print(
                f"[FINAL] {transcript}"
                f"  ({latency:.2f} ms)"
            )

            if transcript:
                final_transcript_parts.append(
                    transcript
                )

                last_final_at = (
                    time.perf_counter()
                )

        # -----------------------------------------------------------
        # Speech started
        # -----------------------------------------------------------

        elif event_type == "speech_started":
            print(
                "[VAD] Speech started"
            )

        # -----------------------------------------------------------
        # Utterance end
        # -----------------------------------------------------------

        elif event_type == "utterance_end":
            utterance_ready_at = (
                time.perf_counter()
            )

            print(
                "[VAD] Utterance ended"
            )

            final_utterance = " ".join(
                final_transcript_parts
            ).strip()

            final_transcript_parts.clear()

            if not final_utterance:
                last_final_at = None
                continue

            print()
            print(
                f"[UTTERANCE] "
                f"{final_utterance}"
            )

            # Suppress microphone immediately.
            ai_speaking.set()

            await response_queue.put(
                (
                    final_utterance,
                    utterance_ready_at,
                    last_final_at,
                )
            )

            last_final_at = None

        # -----------------------------------------------------------
        # Silence
        # -----------------------------------------------------------

        elif event_type == "silence":
            final_transcript_parts.clear()
            last_final_at = None

            print(
                "[VAD] Silence detected"
            )

        # -----------------------------------------------------------
        # Incomplete speech
        # -----------------------------------------------------------

        elif event_type == "incomplete_speech":
            transcript = event.get(
                "transcript",
                "",
            )

            print(
                f"[STT] Incomplete speech: "
                f"{transcript}"
            )

        # -----------------------------------------------------------
        # Error
        # -----------------------------------------------------------

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
    ai_speaking: asyncio.Event,
) -> None:
    """
    Stream microphone audio to STT.

    While the AI is generating/speaking, microphone samples
    are replaced with silence.

    Silence keeps the Deepgram connection active without
    allowing speaker feedback to create another utterance.
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
    suppressed_frames = 0

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

            if ai_speaking.is_set():
                suppressed_frames += 1

                # Same PCM frame length, but silence.
                stt_audio = bytes(
                    len(stt_audio)
                )

            total_audio_bytes += len(
                stt_audio
            )

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
            f"Suppressed frames: {suppressed_frames}. "
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
    print("AGNI AI - INTEGRATED VOICE PIPELINE")
    print("=" * 72)
    print()

    # ---------------------------------------------------------------
    # Providers
    # ---------------------------------------------------------------

    print(
        "Initializing OpenAI LLM provider..."
    )

    llm_provider = OpenAILLMProvider()

    print(
        f"LLM ready: {llm_provider.model}"
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

    loop = asyncio.get_running_loop()

    microphone_track_future: (
        asyncio.Future[
            tuple[
                rtc.Track,
                str,
            ]
        ]
    ) = loop.create_future()

    stt_adapter: STTStreamAdapter | None = None

    audio_task: asyncio.Task | None = None
    stt_event_task: asyncio.Task | None = None
    response_task: asyncio.Task | None = None

    # ---------------------------------------------------------------
    # LiveKit track callback
    # ---------------------------------------------------------------

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        if publication.name == "voice-output":
            return

        if microphone_track_future.done():
            return

        print()
        print(
            f"Subscribed to track "
            f"'{publication.name}' "
            f"from '{participant.identity}'"
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
            endpoint=STT_STREAM_ENDPOINT
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
            )
        )

        response_task = asyncio.create_task(
            process_ai_responses(
                response_queue,
                llm_provider,
                tts_provider,
                audio_output,
                ai_speaking,
            )
        )

        audio_task = asyncio.create_task(
            consume_audio_track(
                microphone_track,
                microphone_identity,
                stt_adapter,
                ai_speaking,
            )
        )

        print()
        print(
            f"Streaming STT endpoint: "
            f"{STT_STREAM_ENDPOINT}"
        )

        print()
        print(
            "Voice pipeline ready."
        )

        print(
            "Only completed utterances are "
            "sent to OpenAI."
        )

        await asyncio.Event().wait()

    finally:
        print()
        print(
            "Stopping Agni AI voice pipeline..."
        )

        if audio_task is not None:
            audio_task.cancel()

            await asyncio.gather(
                audio_task,
                return_exceptions=True,
            )

        if stt_event_task is not None:
            stt_event_task.cancel()

            await asyncio.gather(
                stt_event_task,
                return_exceptions=True,
            )

        if response_task is not None:
            response_task.cancel()

            await asyncio.gather(
                response_task,
                return_exceptions=True,
            )

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
            f"Voice pipeline error: {exc}"
        )