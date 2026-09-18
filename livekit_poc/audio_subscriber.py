"""
Agni AI - LiveKit Audio to Streaming STT

Receives realtime microphone audio from LiveKit and continuously sends
converted PCM16 audio to the Agni AI streaming STT WebSocket endpoint.

The terminal is intentionally optimized for transcript readability:

    [LIVE]    Current interim hypothesis
    [FINAL]   Confirmed transcript segment
    [VAD]     Speech activity

At shutdown:

    COMPLETE TRANSCRIPT

contains only finalized transcript segments.
"""

from __future__ import annotations

import asyncio
import os

import numpy as np
from dotenv import load_dotenv
from livekit import api, rtc
from scipy.signal import resample_poly

from app.voice.stt_stream_adapter import STTStreamAdapter


# ===========================================================================
# Environment
# ===========================================================================

load_dotenv(
    ".env.local",
    override=True,
)


# ===========================================================================
# LiveKit configuration
# ===========================================================================

ROOM_NAME = os.getenv(
    "LIVEKIT_ROOM_NAME",
    "agni-ai-voice-poc",
)

PARTICIPANT_IDENTITY = os.getenv(
    "LIVEKIT_SUBSCRIBER_IDENTITY",
    "agni-audio-subscriber",
)


# ===========================================================================
# Audio configuration
# ===========================================================================

LIVEKIT_SAMPLE_RATE = 48_000
STT_SAMPLE_RATE = 16_000
STT_CHANNELS = 1


# ===========================================================================
# Streaming STT
# ===========================================================================

STT_STREAM_ENDPOINT = os.getenv(
    "STT_STREAM_ENDPOINT",
    "ws://127.0.0.1:8000/api/v1/stt/stream",
)


# ===========================================================================
# Terminal configuration
# ===========================================================================

LIVE_LINE_WIDTH = 120


# ===========================================================================
# LiveKit authentication
# ===========================================================================

def create_access_token() -> str:
    """
    Create a subscriber-only LiveKit access token.
    """

    api_key = os.getenv(
        "LIVEKIT_API_KEY"
    )

    api_secret = os.getenv(
        "LIVEKIT_API_SECRET"
    )

    if not api_key or not api_secret:
        raise RuntimeError(
            "LIVEKIT_API_KEY or LIVEKIT_API_SECRET is missing "
            "from .env.local"
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
            "Agni AI STT Subscriber"
        )
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=ROOM_NAME,
                can_publish=False,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )


# ===========================================================================
# Audio conversion
# ===========================================================================

def convert_audio_frame_to_stt_format(
    audio_frame: rtc.AudioFrame,
) -> bytes:
    """
    Convert a LiveKit AudioFrame into:

        PCM16 / linear16
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

    # -----------------------------------------------------------------------
    # Multi-channel → mono
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Sample-rate conversion
    # -----------------------------------------------------------------------

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


# ===========================================================================
# Terminal helpers
# ===========================================================================

def clear_live_line() -> None:
    """
    Clear the current [LIVE] terminal line.
    """

    print(
        "\r" + (" " * LIVE_LINE_WIDTH) + "\r",
        end="",
        flush=True,
    )


# ===========================================================================
# STT event handling
# ===========================================================================

async def receive_stt_events(
    stt_adapter: STTStreamAdapter,
    completed_transcript: list[str],
) -> None:
    """
    Receive STT events and display them cleanly.

    Partial results replace the same terminal line.

    Final results are stored and printed as confirmed segments.
    """

    print()
    print("=" * 72)
    print("AGNI AI - LIVE TRANSCRIPTION")
    print("=" * 72)
    print()
    print("Listening...")
    print()

    live_line_active = False

    while True:

        event = await stt_adapter.receive_event()

        event_type = event.get(
            "type",
            "unknown",
        )

        # -------------------------------------------------------------------
        # Partial
        # -------------------------------------------------------------------

        if event_type == "partial":

            transcript = event.get(
                "transcript",
                "",
            ).strip()

            if not transcript:
                continue

            live_line_active = True

            live_text = (
                f"[LIVE] {transcript}"
            )

            # Clear the previous live hypothesis and replace it.
            print(
                "\r" + (" " * LIVE_LINE_WIDTH) + "\r",
                end="",
                flush=True,
            )

            print(
                live_text[:LIVE_LINE_WIDTH],
                end="",
                flush=True,
            )

            continue

        # -------------------------------------------------------------------
        # Final
        # -------------------------------------------------------------------

        if event_type == "final":

            transcript = event.get(
                "transcript",
                "",
            ).strip()

            if not transcript:
                continue

            # Remove the currently displayed interim result.
            if live_line_active:
                clear_live_line()
                live_line_active = False

            # Store ONLY finalized transcript segments.
            completed_transcript.append(
                transcript
            )

            latency = float(
                event.get(
                    "latency_ms",
                    0,
                )
            )

            speech_final = bool(
                event.get(
                    "speech_final",
                    False,
                )
            )

            print(
                f"[FINAL] {transcript}"
            )

            print(
                f"        Response latency: "
                f"{latency:.2f} ms"
            )

            if speech_final:
                print(
                    "        └─ end of speech"
                )

            print()

            continue

        # -------------------------------------------------------------------
        # Speech started
        # -------------------------------------------------------------------

        if event_type == "speech_started":

            if live_line_active:
                clear_live_line()
                live_line_active = False

            print(
                "[VAD] Speech started"
            )

            continue

        # -------------------------------------------------------------------
        # Utterance end
        # -------------------------------------------------------------------

        if event_type == "utterance_end":

            print(
                "[VAD] Utterance ended"
            )

            continue

        # -------------------------------------------------------------------
        # Silence
        # -------------------------------------------------------------------

        if event_type == "silence":

            print(
                "[VAD] Silence detected"
            )

            continue

        # -------------------------------------------------------------------
        # Incomplete speech
        # -------------------------------------------------------------------

        if event_type == "incomplete_speech":

            transcript = event.get(
                "transcript",
                "",
            ).strip()

            if live_line_active:
                clear_live_line()
                live_line_active = False

            print(
                f"[INCOMPLETE] {transcript}"
            )

            continue

        # -------------------------------------------------------------------
        # Error
        # -------------------------------------------------------------------

        if event_type == "error":

            if live_line_active:
                clear_live_line()
                live_line_active = False

            print(
                "[STT ERROR]"
            )

            print(
                event.get(
                    "message",
                    "",
                )
            )

            continue

        # -------------------------------------------------------------------
        # Unknown
        # -------------------------------------------------------------------

        if live_line_active:
            clear_live_line()
            live_line_active = False

        print(
            f"[STT EVENT] {event}"
        )


# ===========================================================================
# LiveKit audio stream
# ===========================================================================

async def consume_audio_track(
    track: rtc.Track,
    participant_identity: str,
    stt_adapter: STTStreamAdapter,
) -> None:
    """
    Consume the remote microphone track continuously.

    No 2-second buffering is performed.
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

            # Send immediately to the persistent STT connection.
            await stt_adapter.send_audio(
                stt_audio
            )

    finally:

        await audio_stream.aclose()

        print()
        print(
            f"Audio stream ended. "
            f"Frames received: {frame_count}. "
            f"STT audio bytes sent: {total_audio_bytes}"
        )


# ===========================================================================
# Main
# ===========================================================================

async def main() -> None:
    """
    Connect to LiveKit and route the microphone stream to STT.
    """

    livekit_url = os.getenv(
        "LIVEKIT_URL"
    )

    if not livekit_url:
        raise RuntimeError(
            "LIVEKIT_URL is missing from .env.local"
        )

    # -----------------------------------------------------------------------
    # Finalized transcript segments
    # -----------------------------------------------------------------------

    completed_transcript: list[str] = []

    # -----------------------------------------------------------------------
    # Streaming STT adapter
    # -----------------------------------------------------------------------

    stt_adapter = STTStreamAdapter(
        endpoint=STT_STREAM_ENDPOINT
    )

    print(
        "Connecting to streaming STT..."
    )

    await stt_adapter.connect()

    # -----------------------------------------------------------------------
    # STT event receiver
    # -----------------------------------------------------------------------

    stt_event_task = asyncio.create_task(
        receive_stt_events(
            stt_adapter,
            completed_transcript,
        )
    )

    # -----------------------------------------------------------------------
    # LiveKit room
    # -----------------------------------------------------------------------

    room = rtc.Room()

    audio_task: asyncio.Task | None = None

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        nonlocal audio_task

        print()
        print(
            f"Subscribed to track "
            f"'{publication.name}' "
            f"from '{participant.identity}'"
        )

        if (
            audio_task is not None
            and not audio_task.done()
        ):
            return

        audio_task = asyncio.create_task(
            consume_audio_track(
                track,
                participant.identity,
                stt_adapter,
            )
        )

    # -----------------------------------------------------------------------
    # Connect LiveKit
    # -----------------------------------------------------------------------

    print(
        "Connecting to LiveKit..."
    )

    await room.connect(
        livekit_url,
        create_access_token(),
    )

    print(
        "Connected to LiveKit"
    )

    print(
        f"Room: {room.name}"
    )

    print(
        "Participant: "
        f"{room.local_participant.identity}"
    )

    print(
        f"Streaming STT endpoint: "
        f"{STT_STREAM_ENDPOINT}"
    )

    print()
    print(
        "Waiting for incoming microphone audio..."
    )

    try:

        # Keep subscriber alive.
        await asyncio.Event().wait()

    finally:

        print()
        print(
            "Stopping LiveKit audio subscriber..."
        )

        # -------------------------------------------------------------------
        # Stop the LiveKit audio consumer.
        # -------------------------------------------------------------------

        if audio_task is not None:

            audio_task.cancel()

            await asyncio.gather(
                audio_task,
                return_exceptions=True,
            )

        # -------------------------------------------------------------------
        # Close STT connection.
        #
        # Any events already queued by the adapter are given a moment
        # to be consumed before the receiver task is cancelled.
        # -------------------------------------------------------------------

        await stt_adapter.close()

        await asyncio.sleep(
            0.25
        )

        stt_event_task.cancel()

        await asyncio.gather(
            stt_event_task,
            return_exceptions=True,
        )

        # -------------------------------------------------------------------
        # Disconnect LiveKit.
        # -------------------------------------------------------------------

        await room.disconnect()

        # -------------------------------------------------------------------
        # Final transcript
        # -------------------------------------------------------------------

        print()
        print("=" * 72)
        print("COMPLETE TRANSCRIPT")
        print("=" * 72)
        print()

        if completed_transcript:

            complete_text = " ".join(
                completed_transcript
            )

            print(
                complete_text
            )

        else:

            print(
                "No finalized transcript was received."
            )

        print()
        print("=" * 72)
        print(
            f"Final segments: "
            f"{len(completed_transcript)}"
        )
        print("=" * 72)

        print(
            "Disconnected from LiveKit"
        )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print()
        print(
            "Application stopped by user."
        )