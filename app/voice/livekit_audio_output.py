"""
Agni AI - LiveKit Audio Output

Publishes synthesized speech audio into a LiveKit room.

This module is intentionally independent of any particular
TTS provider.

Flow:

    TTSProvider
        ↓
    TTSAudioChunk
        ↓
    PCM audio
        ↓
    LiveKit AudioSource
        ↓
    LocalAudioTrack
        ↓
    LiveKit Room
        ↓
    Caller / Remote Participant
"""

from __future__ import annotations

from livekit import rtc

from app.voice.tts_provider import TTSAudioChunk


class LiveKitAudioOutput:
    """
    Handles publishing synthesized speech audio through LiveKit.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
    ) -> None:
        """
        Initialize the audio output.

        The initial sample rate is kept configurable because
        different TTS providers may produce different audio formats.
        """

        self.sample_rate = sample_rate
        self.channels = channels

        # AudioSource receives PCM audio frames that will be
        # published through LiveKit.
        self.audio_source = rtc.AudioSource(
            sample_rate=self.sample_rate,
            num_channels=self.channels,
        )

        # The LocalAudioTrack represents the outgoing voice audio.
        self.audio_track = rtc.LocalAudioTrack.create_audio_track(
            "voice-output",
            self.audio_source,
        )

    async def publish(
        self,
        participant: rtc.LocalParticipant,
    ) -> rtc.LocalTrackPublication:
        """
        Publish the voice output track to the LiveKit room.
        """

        return await participant.publish_track(
            self.audio_track
        )

    async def send_chunk(
        self,
        audio_chunk: TTSAudioChunk,
    ) -> None:
        """
        Send one synthesized audio chunk into LiveKit.

        The provider's audio is converted into a LiveKit
        AudioFrame before being passed to the AudioSource.
        """

        if audio_chunk.channels != self.channels:
            raise ValueError(
                "TTS audio channel count does not match "
                "LiveKit audio output configuration"
            )

        if audio_chunk.encoding != "pcm_s16le":
            raise ValueError(
                "LiveKit audio output currently expects "
                "PCM 16-bit little-endian audio"
            )

        # LiveKit needs an AudioFrame describing the raw PCM data.
        frame = rtc.AudioFrame(
            data=audio_chunk.data,
            sample_rate=audio_chunk.sample_rate,
            num_channels=audio_chunk.channels,
            samples_per_channel=(
                len(audio_chunk.data)
                // 2
                // audio_chunk.channels
            ),
        )

        # Push the frame into the outgoing audio source.
        await self.audio_source.capture_frame(
            frame
        )

    async def close(self) -> None:
        """
        Release the audio source when the voice session ends.
        """

        await self.audio_source.aclose()