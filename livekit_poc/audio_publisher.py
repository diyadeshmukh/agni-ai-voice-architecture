"""
Agni AI - LiveKit Microphone Audio Publisher

Captures real microphone audio and publishes it to a LiveKit room.

Flow:

    Microphone
        ↓
    PlatformAudio
        ↓
    LocalAudioTrack
        ↓
    LiveKit Room
        ↓
    Remote Subscriber
"""

import asyncio
import os

from dotenv import load_dotenv
from livekit import api, rtc


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

# LiveKit CLI generated the project credentials in .env.local.
load_dotenv(".env.local", override=True)


# ---------------------------------------------------------------------------
# Room / participant configuration
# ---------------------------------------------------------------------------

ROOM_NAME = os.getenv(
    "LIVEKIT_ROOM_NAME",
    "agni-ai-voice-poc",
)

PARTICIPANT_IDENTITY = os.getenv(
    "LIVEKIT_PARTICIPANT_IDENTITY",
    "agni-audio-publisher",
)


# ---------------------------------------------------------------------------
# LiveKit authentication
# ---------------------------------------------------------------------------

def create_access_token() -> str:
    """
    Create a LiveKit access token for the audio publisher.

    The participant can:
        - join the configured room
        - publish audio
        - subscribe to other tracks
    """

    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        raise RuntimeError(
            "LIVEKIT_API_KEY or LIVEKIT_API_SECRET is missing "
            "from .env.local"
        )

    return (
        api.AccessToken(api_key, api_secret)
        .with_identity(PARTICIPANT_IDENTITY)
        .with_name("Agni AI Audio Publisher")
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
# Main microphone publishing flow
# ---------------------------------------------------------------------------

async def main():
    """
    Connect to LiveKit and publish real microphone audio.
    """

    livekit_url = os.getenv("LIVEKIT_URL")

    if not livekit_url:
        raise RuntimeError(
            "LIVEKIT_URL is missing from .env.local"
        )

    room = rtc.Room()

    print("Connecting to LiveKit...")

    # Connect the participant to the LiveKit room.
    await room.connect(
        livekit_url,
        create_access_token(),
    )

    print("Connected to LiveKit")
    print(f"Room: {room.name}")
    print(
        f"Participant: "
        f"{room.local_participant.identity}"
    )

    # -----------------------------------------------------------------------
    # Initialize platform audio
    # -----------------------------------------------------------------------
    #
    # PlatformAudio provides access to the operating system's
    # microphone/audio device.
    #
    platform_audio = rtc.PlatformAudio()

    # Display available recording devices so we can verify that
    # the expected microphone is visible to LiveKit.
    recording_devices = platform_audio.recording_devices()

    print("\nAvailable microphones:")

    if not recording_devices:
        print("No recording devices found.")

    for device in recording_devices:
        print(
            f"  [{device.index}] {device.name}"
        )

    # -----------------------------------------------------------------------
    # Create microphone audio source
    # -----------------------------------------------------------------------
    #
    # The platform audio interface captures microphone input and
    # provides it as an audio source that can be published.
    #
    audio_source = platform_audio.create_audio_source()

    # Create a LocalAudioTrack backed by the microphone source.
    microphone_track = rtc.LocalAudioTrack.create_audio_track(
        "microphone",
        audio_source,
    )

    # -----------------------------------------------------------------------
    # Publish microphone audio to LiveKit
    # -----------------------------------------------------------------------

    publication = await room.local_participant.publish_track(
        microphone_track
    )

    print("\nMicrophone audio published")
    print(f"Track: {publication.name}")
    print("\nSpeak into the microphone...")
    print("Press Ctrl+C to stop.")

    try:
        # Keep the participant connected while microphone audio
        # is being captured and published.
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping microphone publisher...")

    finally:
        # Clean up audio resources.
        audio_source.close()
        platform_audio.close()

        await room.disconnect()

        print("Disconnected from LiveKit")


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")