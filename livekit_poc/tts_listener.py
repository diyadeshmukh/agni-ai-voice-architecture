"""
Agni AI - LiveKit TTS Listener

Subscribes to the synthesized "voice-output" track published by
the TTS publisher and plays the received PCM audio through
the laptop's default speaker.

Flow:

    LiveKit Room
         ↓
    voice-output track
         ↓
    AudioStream
         ↓
    PCM16 @ 16 kHz mono
         ↓
    Laptop Speaker
"""

from __future__ import annotations

import asyncio
import os

import sounddevice as sd
from dotenv import load_dotenv
from livekit import api, rtc


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

load_dotenv(
    ".env.local",
    override=True,
)


# ---------------------------------------------------------------------------
# LiveKit configuration
# ---------------------------------------------------------------------------

ROOM_NAME = os.getenv(
    "LIVEKIT_ROOM_NAME",
    "agni-ai-voice-poc",
)

PARTICIPANT_IDENTITY = os.getenv(
    "LIVEKIT_TTS_LISTENER_IDENTITY",
    "agni-tts-listener",
)


# ---------------------------------------------------------------------------
# Audio configuration
# ---------------------------------------------------------------------------

PLAYBACK_SAMPLE_RATE = 16_000
PLAYBACK_CHANNELS = 1
PLAYBACK_FRAME_SIZE_MS = 20


# ---------------------------------------------------------------------------
# LiveKit authentication
# ---------------------------------------------------------------------------

def create_access_token() -> str:
    """
    Create a LiveKit access token for the TTS listener.

    The listener:
        - can join the configured room
        - can subscribe to remote tracks
        - does not publish audio
    """

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
            "Agni AI TTS Listener"
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


# ---------------------------------------------------------------------------
# Audio playback
# ---------------------------------------------------------------------------

async def play_audio_track(
    track: rtc.Track,
    participant_identity: str,
) -> None:
    """
    Receive the TTS track from LiveKit and play it through
    the laptop's default output device.

    LiveKit resamples the incoming audio stream to the requested
    playback sample rate.
    """

    print()
    print(
        f"Receiving TTS audio from "
        f"'{participant_identity}'"
    )

    audio_stream = rtc.AudioStream.from_track(
        track=track,
        sample_rate=PLAYBACK_SAMPLE_RATE,
        num_channels=PLAYBACK_CHANNELS,
        frame_size_ms=PLAYBACK_FRAME_SIZE_MS,
    )

    output_stream = sd.RawOutputStream(
        samplerate=PLAYBACK_SAMPLE_RATE,
        channels=PLAYBACK_CHANNELS,
        dtype="int16",
    )

    output_stream.start()

    frame_count = 0
    total_audio_bytes = 0

    try:
        async for audio_frame_event in audio_stream:

            audio_frame = (
                audio_frame_event.frame
            )

            pcm_audio = bytes(
                audio_frame.data
            )

            if not pcm_audio:
                continue

            frame_count += 1
            total_audio_bytes += len(
                pcm_audio
            )

            await asyncio.to_thread(
                output_stream.write,
                pcm_audio,
            )

    finally:
        output_stream.stop()
        output_stream.close()

        await audio_stream.aclose()

        print()
        print(
            f"TTS playback ended. "
            f"Frames: {frame_count}. "
            f"Audio bytes: {total_audio_bytes}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Connect to LiveKit and wait for the TTS output track.
    """

    livekit_url = os.getenv(
        "LIVEKIT_URL"
    )

    if not livekit_url:
        raise RuntimeError(
            "LIVEKIT_URL is missing from "
            ".env.local"
        )

    room = rtc.Room()

    playback_task: asyncio.Task | None = None

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        nonlocal playback_task

        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        # Only listen to the audio track created by
        # LiveKitAudioOutput.
        if publication.name != "voice-output":
            return

        print()
        print(
            f"Subscribed to TTS track "
            f"'{publication.name}' "
            f"from '{participant.identity}'"
        )

        if (
            playback_task is None
            or playback_task.done()
        ):
            playback_task = asyncio.create_task(
                play_audio_track(
                    track,
                    participant.identity,
                )
            )

    print()
    print("=" * 72)
    print("AGNI AI - TTS LISTENER")
    print("=" * 72)
    print()

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
        f"Participant: "
        f"{room.local_participant.identity}"
    )

    print()
    print(
        "Waiting for 'voice-output'..."
    )
    print(
        "Keep this terminal running while "
        "the TTS publisher is active."
    )

    try:
        await asyncio.Event().wait()

    finally:

        if playback_task is not None:
            playback_task.cancel()

            await asyncio.gather(
                playback_task,
                return_exceptions=True,
            )

        await room.disconnect()

        print()
        print(
            "Disconnected from LiveKit."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        print(
            "TTS listener stopped by user."
        )