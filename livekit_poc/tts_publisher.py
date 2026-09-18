"""
Agni AI - LiveKit TTS Publisher

Generates speech using the configured TTS provider and publishes
the synthesized PCM audio into a LiveKit room.

Flow:

    Text
      ↓
    ElevenLabsTTSProvider
      ↓
    Streaming PCM16 audio
      ↓
    TTSAudioChunk
      ↓
    LiveKitAudioOutput
      ↓
    LiveKit AudioSource
      ↓
    LocalAudioTrack
      ↓
    LiveKit Room
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from livekit import api, rtc

from app.voice.elevenlabs_tts_provider import (
    ElevenLabsTTSProvider,
)
from app.voice.livekit_audio_output import (
    LiveKitAudioOutput,
)


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
    "LIVEKIT_TTS_PUBLISHER_IDENTITY",
    "agni-tts-publisher",
)


# ---------------------------------------------------------------------------
# TTS audio configuration
# ---------------------------------------------------------------------------

TTS_SAMPLE_RATE = 16_000
TTS_CHANNELS = 1


# ---------------------------------------------------------------------------
# Default text
# ---------------------------------------------------------------------------

DEFAULT_TEXT = (
    "Ganesh utsavachya haardik shubhechha !"
    "Ganpati bappa morya !"
    "Mangal murti morya ! "
)


# ---------------------------------------------------------------------------
# LiveKit authentication
# ---------------------------------------------------------------------------

def create_access_token() -> str:
    """
    Create a LiveKit access token for the TTS publisher.

    The publisher:
        - can join the configured room
        - can publish audio
        - does not need to subscribe to remote tracks
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
            "Agni AI TTS Publisher"
        )
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=ROOM_NAME,
                can_publish=True,
                can_subscribe=False,
            )
        )
        .to_jwt()
    )


# ---------------------------------------------------------------------------
# Text selection
# ---------------------------------------------------------------------------

def get_text() -> str:
    """
    Return custom command-line text when supplied.
    Otherwise use the default test sentence.
    """

    if len(sys.argv) > 1:
        text = " ".join(
            sys.argv[1:]
        ).strip()

        if text:
            return text

    return DEFAULT_TEXT


# ---------------------------------------------------------------------------
# Main TTS publisher
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Generate speech with ElevenLabs and publish it through LiveKit.
    """

    livekit_url = os.getenv(
        "LIVEKIT_URL"
    )

    if not livekit_url:
        raise RuntimeError(
            "LIVEKIT_URL is missing from .env.local"
        )

    text = get_text()

    print()
    print("=" * 72)
    print("AGNI AI - TTS PUBLISHER")
    print("=" * 72)
    print()

    print("Text:")
    print(text)
    print()

    print(
        "Initializing ElevenLabs TTS provider..."
    )

    tts_provider = ElevenLabsTTSProvider()

    print(
        "ElevenLabs TTS provider initialized."
    )

    room = rtc.Room()

    audio_output = LiveKitAudioOutput(
        sample_rate=TTS_SAMPLE_RATE,
        channels=TTS_CHANNELS,
    )

    try:
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

        publication = (
            await audio_output.publish(
                room.local_participant
            )
        )

        print(
            f"Published audio track: "
            f"{publication.sid}"
        )

        print()
        print(
            "Generating and streaming speech..."
        )
        print()

        chunk_count = 0
        total_audio_bytes = 0

        async for audio_chunk in (
            tts_provider.synthesize(text)
        ):
            chunk_count += 1
            total_audio_bytes += len(
                audio_chunk.data
            )

            await audio_output.send_chunk(
                audio_chunk
            )

            print(
                f"\rStreaming audio chunks: "
                f"{chunk_count}",
                end="",
                flush=True,
            )

        print()
        print()
        print(
            "TTS generation finished."
        )

        print(
            f"Audio chunks: {chunk_count}"
        )

        print(
            f"Audio bytes: "
            f"{total_audio_bytes}"
        )

        print()
        print(
            "Waiting for audio playout..."
        )

        await (
            audio_output
            .audio_source
            .wait_for_playout()
        )

        print(
            "Audio playout completed."
        )

        print()
        print(
            "TTS audio was published successfully."
        )

    finally:
        await audio_output.close()
        await room.disconnect()

        print(
            "Disconnected from LiveKit."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(
            main()
        )

    except KeyboardInterrupt:
        print()
        print(
            "TTS publisher stopped by user."
        )

    except Exception as exc:
        print()
        print(
            f"TTS publisher error: {exc}"
        )