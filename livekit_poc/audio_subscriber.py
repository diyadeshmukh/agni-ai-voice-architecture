"""
Agni AI - LiveKit Audio to STT Integration

Receives realtime microphone audio from LiveKit, buffers it into
2-second chunks, converts the audio into the format expected by
the STT service, and sends each chunk for transcription.

Audio flow:

    LiveKit Audio Track
            ↓
        AudioStream
            ↓
        AudioFrame
            ↓
    AudioChunkBuffer
            ↓
      2-second PCM16
            ↓
       STTAdapter
            ↓
     48 kHz → 16 kHz
            ↓
       WAV / PCM16
            ↓
        POST /audio
            ↓
      STT Service
            ↓
      Deepgram Nova-3
            ↓
        Transcript
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from livekit import api, rtc

from app.voice.audio_buffer import AudioChunkBuffer
from app.voice.stt_adapter import STTAdapter


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

load_dotenv(".env.local", override=True)


# ---------------------------------------------------------------------------
# LiveKit configuration
# ---------------------------------------------------------------------------

ROOM_NAME = os.getenv(
    "LIVEKIT_ROOM_NAME",
    "agni-ai-voice-poc",
)

PARTICIPANT_IDENTITY = os.getenv(
    "LIVEKIT_SUBSCRIBER_IDENTITY",
    "agni-audio-subscriber",
)

LIVEKIT_SAMPLE_RATE = 48_000
AUDIO_CHANNELS = 1


# ---------------------------------------------------------------------------
# STT configuration
# ---------------------------------------------------------------------------

STT_ENDPOINT = os.getenv(
    "STT_ENDPOINT",
    "http://127.0.0.1:8000/audio",
)

STT_CHUNK_DURATION = float(
    os.getenv(
        "STT_CHUNK_DURATION_SECONDS",
        "2",
    )
)


# ---------------------------------------------------------------------------
# LiveKit authentication
# ---------------------------------------------------------------------------

def create_access_token() -> str:
    """
    Create an access token for the LiveKit audio subscriber.

    The participant is allowed to:
        - join the configured room
        - subscribe to remote audio tracks

    It does not need to publish audio.
    """

    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

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
        .with_identity(PARTICIPANT_IDENTITY)
        .with_name("Agni AI STT Subscriber")
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
# STT processing
# ---------------------------------------------------------------------------

async def send_chunk_to_stt(
    stt_adapter: STTAdapter,
    audio_chunk: bytes,
    chunk_number: int,
) -> None:
    """
    Send one completed audio chunk to the STT service.

    The STT adapter performs:
        PCM16 → 16 kHz → WAV → POST /audio

    The HTTP request is executed in a worker thread so that
    the LiveKit audio-receiving loop is not blocked by the
    synchronous requests library.
    """

    filename = (
        f"livekit_chunk_{chunk_number:04d}.wav"
    )

    print(
        f"\nSending audio chunk "
        f"{chunk_number} to STT..."
    )

    try:
        transcript = await asyncio.to_thread(
            stt_adapter.send_audio,
            audio_chunk,
            filename,
        )

        print(
            f"Transcript [{chunk_number}]: "
            f"{transcript}"
        )

    except Exception as exc:
        print(
            f"STT processing failed for chunk "
            f"{chunk_number}: {exc}"
        )


# ---------------------------------------------------------------------------
# LiveKit audio stream handling
# ---------------------------------------------------------------------------

async def consume_audio_track(
    track: rtc.Track,
    participant_identity: str,
    stt_adapter: STTAdapter,
    processing_tasks: set[asyncio.Task],
) -> None:
    """
    Consume a remote LiveKit audio track.

    Incoming AudioFrames are continuously added to a 2-second
    buffer. Whenever a complete chunk is available, it is sent
    to the STT service in a background task.
    """

    if track.kind != rtc.TrackKind.KIND_AUDIO:
        return

    print(
        f"\nReceiving audio from "
        f"'{participant_identity}'"
    )

    # Buffer incoming 48 kHz PCM16 mono audio into 2-second chunks.
    audio_buffer = AudioChunkBuffer(
        sample_rate=LIVEKIT_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        chunk_duration_seconds=STT_CHUNK_DURATION,
    )

    # AudioStream exposes incoming audio as AudioFrame objects.
    audio_stream = rtc.AudioStream(track)

    chunk_number = 0

    try:
        async for audio_frame_event in audio_stream:

            audio_frame = audio_frame_event.frame

            # LiveKit provides the raw PCM audio data for this frame.
            pcm_bytes = bytes(audio_frame.data)

            # Add this frame to our 2-second buffer.
            complete_chunk = audio_buffer.add(
                pcm_bytes
            )

            # A complete chunk becomes available once 2 seconds
            # of audio have been collected.
            if complete_chunk is not None:

                chunk_number += 1

                print(
                    f"\n2-second audio chunk ready: "
                    f"{chunk_number}"
                )

                # Process the chunk asynchronously so that receiving
                # the next LiveKit frames can continue immediately.
                task = asyncio.create_task(
                    send_chunk_to_stt(
                        stt_adapter,
                        complete_chunk,
                        chunk_number,
                    )
                )

                processing_tasks.add(task)

                task.add_done_callback(
                    processing_tasks.discard
                )

    finally:
        # If the track ends with a partial chunk, process it as well.
        remaining_audio = audio_buffer.flush()

        if remaining_audio:
            chunk_number += 1

            print(
                f"\nProcessing final audio chunk: "
                f"{chunk_number}"
            )

            task = asyncio.create_task(
                send_chunk_to_stt(
                    stt_adapter,
                    remaining_audio,
                    chunk_number,
                )
            )

            processing_tasks.add(task)

            task.add_done_callback(
                processing_tasks.discard
            )

        await audio_stream.aclose()

        print(
            f"Audio stream ended. "
            f"Chunks created: {chunk_number}"
        )


# ---------------------------------------------------------------------------
# Main LiveKit connection
# ---------------------------------------------------------------------------

async def main():
    """
    Connect to the LiveKit room and route incoming audio
    into the STT service.
    """

    livekit_url = os.getenv("LIVEKIT_URL")

    if not livekit_url:
        raise RuntimeError(
            "LIVEKIT_URL is missing from .env.local"
        )

    # Create the adapter once and reuse it for all audio chunks.
    stt_adapter = STTAdapter(
        endpoint=STT_ENDPOINT,
        input_sample_rate=LIVEKIT_SAMPLE_RATE,
        output_sample_rate=16_000,
        channels=AUDIO_CHANNELS,
    )

    room = rtc.Room()

    # Keep references to active STT processing tasks so they
    # can be cleaned up when the service stops.
    processing_tasks: set[asyncio.Task] = set()

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """
        Start consuming audio when a remote participant's
        audio track becomes available.
        """

        print(
            f"\nSubscribed to track "
            f"'{publication.name}' "
            f"from '{participant.identity}'"
        )

        task = asyncio.create_task(
            consume_audio_track(
                track,
                participant.identity,
                stt_adapter,
                processing_tasks,
            )
        )

        processing_tasks.add(task)

        task.add_done_callback(
            processing_tasks.discard
        )

    print("Connecting to LiveKit...")

    await room.connect(
        livekit_url,
        create_access_token(),
    )

    print("Connected")
    print(f"Room: {room.name}")
    print(
        f"Participant: "
        f"{room.local_participant.identity}"
    )
    print(
        f"STT endpoint: {STT_ENDPOINT}"
    )

    print(
        "\nWaiting for incoming microphone audio..."
    )

    try:
        # Keep the subscriber connected while audio is flowing.
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print(
            "\nStopping LiveKit audio subscriber..."
        )

    finally:
        # Cancel any remaining STT tasks.
        for task in processing_tasks:
            task.cancel()

        if processing_tasks:
            await asyncio.gather(
                *processing_tasks,
                return_exceptions=True,
            )

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