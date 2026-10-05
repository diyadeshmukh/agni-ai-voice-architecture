"""
Agni AI - LiveKit Microphone Audio Publisher

Captures real microphone audio and publishes it to a LiveKit room.

This file is a local development/test client.

For the production Exotel integration, the caller's phone handles
its own microphone, speaker, headset, and Bluetooth routing.

Local test flow:

    Laptop Microphone
        ↓
    MediaDevices Input
    AEC enabled
        ↓
    LocalAudioTrack
        ↓
    LiveKit Room
        ↓
    Agni AI
        ↓
    voice-output
        ↓
    MediaDevices Output
        ↓
    Laptop Speaker

Using the same MediaDevices instance for input and output allows
LiveKit's WebRTC audio processing module to use speaker playback
as the AEC reverse/reference stream.
"""

import asyncio
import logging
import os

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
# Logging
# ---------------------------------------------------------------------------

class _AudioMixerTimeoutFilter(logging.Filter):
    """
    Hide the expected LiveKit AudioMixer warning that occurs
    while the remote voice-output track is temporarily silent.
    """

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:
        message = record.getMessage()

        return not (
            message.startswith(
                "AudioMixer: stream "
            )
            and message.endswith(
                " timeout, ignoring"
            )
        )


logging.getLogger(
    "livekit"
).addFilter(
    _AudioMixerTimeoutFilter()
)


# ---------------------------------------------------------------------------
# Room / participant configuration
# ---------------------------------------------------------------------------

ROOM_NAME = (
    os.getenv("AGNI_SESSION_ROOM_NAME")
    or os.getenv(
        "LIVEKIT_ROOM_NAME",
        "agni-ai-voice-poc",
    )
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
            "Agni AI Audio Publisher"
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
# Local laptop device selection
# ---------------------------------------------------------------------------

def select_laptop_microphone(
    media_devices: rtc.MediaDevices,
) -> int:
    """
    Select the laptop's built-in microphone array.

    Bluetooth/headset microphones are intentionally ignored because
    this module is only used as the local Agni development harness.
    """

    input_devices = media_devices.list_input_devices()

    # Prefer the built-in Intel microphone array.
    for device in input_devices:
        name = str(
            device.get(
                "name",
                "",
            )
        ).lower()

        if (
            "microphone array" in name
            and "intel" in name
        ):
            print(
                "Laptop microphone selected:"
            )

            print(
                f"  Input:  "
                f"[{device['index']}] "
                f"{device['name']}"
            )

            return int(
                device["index"]
            )

    # Some audio backends expose the same microphone without
    # including "Intel" in the name.
    for device in input_devices:
        name = str(
            device.get(
                "name",
                "",
            )
        ).lower()

        if "microphone array" in name:
            print(
                "Laptop microphone selected:"
            )

            print(
                f"  Input:  "
                f"[{device['index']}] "
                f"{device['name']}"
            )

            return int(
                device["index"]
            )

    raise RuntimeError(
        "Laptop microphone array could not be found."
    )


def select_laptop_speaker(
    media_devices: rtc.MediaDevices,
) -> int:
    """
    Select the laptop's built-in Realtek speaker output.

    Bluetooth/headphone outputs are intentionally ignored because
    this module is only used as the local Agni development harness.
    """

    output_devices = media_devices.list_output_devices()

    # Prefer the normal Realtek laptop speaker endpoint.
    for device in output_devices:
        name = str(
            device.get(
                "name",
                "",
            )
        ).lower()

        if (
            "speakers" in name
            and "realtek" in name
        ):
            print(
                "Laptop speaker selected:"
            )

            print(
                f"  Output: "
                f"[{device['index']}] "
                f"{device['name']}"
            )

            return int(
                device["index"]
            )

    # Fallback for systems where the Realtek name differs slightly.
    for device in output_devices:
        name = str(
            device.get(
                "name",
                "",
            )
        ).lower()

        if (
            "speaker" in name
            and "headphone" not in name
            and "headset" not in name
        ):
            print(
                "Laptop speaker selected:"
            )

            print(
                f"  Output: "
                f"[{device['index']}] "
                f"{device['name']}"
            )

            return int(
                device["index"]
            )

    raise RuntimeError(
        "Laptop speaker output could not be found."
    )


# ---------------------------------------------------------------------------
# Main microphone publishing flow
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Connect to LiveKit, publish laptop microphone audio,
    and play Agni's voice through the laptop speaker.
    """

    livekit_url = os.getenv(
        "LIVEKIT_URL"
    )

    if not livekit_url:
        raise RuntimeError(
            "LIVEKIT_URL is missing from .env.local"
        )

    room = rtc.Room()

    # -----------------------------------------------------------------------
    # Initialize MediaDevices
    # -----------------------------------------------------------------------
    #
    # The same MediaDevices instance handles both microphone capture
    # and Agni playback.
    #
    # This allows the WebRTC audio-processing module to receive
    # speaker playback as the AEC reverse/reference stream.
    #
    # Local laptop speaker/microphone configuration.
    #
    # AEC is enabled to reduce speaker echo reaching
    # the microphone during full-duplex testing.
    #
    # Noise suppression and automatic gain control
    # remain disabled for the current local test setup.

    media_devices = rtc.MediaDevices()

    input_device = select_laptop_microphone(
        media_devices
    )

    output_device = select_laptop_speaker(
        media_devices
    )

    # -----------------------------------------------------------------------
    # Laptop microphone input
    # -----------------------------------------------------------------------

    input_capture = media_devices.open_input(
        enable_aec=True,
        noise_suppression=False,
        high_pass_filter=True,
        auto_gain_control=False,
        input_device=input_device,
    )

    # -----------------------------------------------------------------------
    # Laptop speaker output
    # -----------------------------------------------------------------------

    output_player = media_devices.open_output(
        output_device=output_device,
    )

    playback_task = None

    # -----------------------------------------------------------------------
    # Play Agni voice
    # -----------------------------------------------------------------------

    async def play_voice_output(
        track: rtc.Track,
    ) -> None:
        await output_player.add_track(
            track
        )

        await output_player.start()

        print(
            "\nAgni voice playback started "
            "through laptop speaker"
        )

    # -----------------------------------------------------------------------
    # LiveKit remote track subscription
    # -----------------------------------------------------------------------

    @room.on(
        "track_subscribed"
    )
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        nonlocal playback_task

        if (
            track.kind
            != rtc.TrackKind.KIND_AUDIO
        ):
            return

        if (
            publication.name
            != "voice-output"
        ):
            return

        print(
            "\nSubscribed to Agni voice track "
            f"'{publication.name}' "
            f"from '{participant.identity}'"
        )

        if (
            playback_task is None
            or playback_task.done()
        ):
            playback_task = asyncio.create_task(
                play_voice_output(
                    track
                )
            )

    # -----------------------------------------------------------------------
    # Connect to LiveKit
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

    # -----------------------------------------------------------------------
    # Create microphone track
    # -----------------------------------------------------------------------

    microphone_track = (
        rtc.LocalAudioTrack.create_audio_track(
            "microphone",
            input_capture.source,
        )
    )

    # -----------------------------------------------------------------------
    # Publish microphone audio
    # -----------------------------------------------------------------------

    publication = (
        await room.local_participant.publish_track(
            microphone_track
        )
    )

    print(
        "\nMicrophone audio published"
    )

    print(
        f"Track: {publication.name}"
    )

    print(
        "Audio processing: "
        "AEC enabled, "
        "NS disabled, "
        "AGC disabled"
    )

    print(
        "\nSpeak into the laptop microphone..."
    )

    print(
        "Press Ctrl+C to stop."
    )

    try:
        while True:
            await asyncio.sleep(
                1
            )

    except KeyboardInterrupt:
        print(
            "\nStopping microphone publisher..."
        )

    finally:
        # -------------------------------------------------------------------
        # Stop playback task
        # -------------------------------------------------------------------

        if (
            playback_task is not None
            and not playback_task.done()
        ):
            playback_task.cancel()

            await asyncio.gather(
                playback_task,
                return_exceptions=True,
            )

        # -------------------------------------------------------------------
        # Close output
        # -------------------------------------------------------------------

        await output_player.aclose()

        # -------------------------------------------------------------------
        # Close microphone capture
        # -------------------------------------------------------------------

        await input_capture.aclose()

        # -------------------------------------------------------------------
        # Disconnect from LiveKit
        # -------------------------------------------------------------------

        await room.disconnect()

        print(
            "Disconnected from LiveKit"
        )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(
            main()
        )

    except KeyboardInterrupt:
        print(
            "\nApplication stopped by user."
        )