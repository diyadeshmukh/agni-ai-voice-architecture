"""
Agni AI - ElevenLabs TTS Provider

Concrete TTS implementation for the Agni AI TTSProvider interface.

Flow:

    Text
      ↓
    ElevenLabs
      ↓
    Streaming PCM16 audio
      ↓
    TTSAudioChunk
      ↓
    LiveKitAudioOutput
"""

from __future__ import annotations

import os
from typing import AsyncIterator

from dotenv import load_dotenv
from elevenlabs import AsyncElevenLabs

from app.voice.tts_provider import TTSProvider, TTSAudioChunk


load_dotenv(".env.local", override=True)


class ElevenLabsTTSProvider(TTSProvider):
    """
    ElevenLabs implementation of the Agni AI TTSProvider interface.

    The provider returns raw PCM16 mono audio at 16 kHz so the
    generated chunks can be passed directly into the existing
    LiveKitAudioOutput.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str | None = None,
        output_format: str | None = None,
    ) -> None:

        self.api_key = (
            api_key
            or os.getenv("ELEVENLABS_API_KEY")
        )

        if not self.api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is missing from .env.local"
            )

        self.voice_id = (
            voice_id
            or os.getenv("ELEVENLABS_VOICE_ID")
        )

        if not self.voice_id:
            raise RuntimeError(
                "ELEVENLABS_VOICE_ID is missing from .env.local"
            )

        self.model_id = (
            model_id
            or os.getenv(
                "ELEVENLABS_MODEL_ID",
                "eleven_flash_v2_5",
            )
        )

        self.output_format = (
            output_format
            or os.getenv(
                "ELEVENLABS_OUTPUT_FORMAT",
                "pcm_16000",
            )
        )

        self.client = AsyncElevenLabs(
            api_key=self.api_key
        )

    async def synthesize(
        self,
        text: str,
    ) -> AsyncIterator[TTSAudioChunk]:
        """
        Convert text into streaming PCM16 audio chunks.

        Args:
            text:
                Text to convert into speech.

        Yields:
            TTSAudioChunk objects containing raw PCM16
            mono 16 kHz audio.
        """

        text = text.strip()

        if not text:
            return

        audio_stream = (
            self.client.text_to_speech.stream(
                voice_id=self.voice_id,
                text=text,
                model_id=self.model_id,
                output_format=self.output_format,
            )
        )

        async for chunk in audio_stream:

            if not isinstance(chunk, bytes):
                continue

            if not chunk:
                continue

            yield TTSAudioChunk(
                data=chunk,
                sample_rate=16_000,
                channels=1,
                encoding="pcm_s16le",
            )