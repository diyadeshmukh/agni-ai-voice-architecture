"""
Agni AI - Text-to-Speech Provider Interface

Defines the common contract for text-to-speech providers.

The voice pipeline should depend on this interface rather than
depending directly on ElevenLabs or another specific TTS provider.

Architecture:

    LLM
      ↓
    TTSProvider
      ↓
    Synthesized audio chunks
      ↓
    LiveKit
      ↓
    Caller
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class TTSAudioChunk:
    """
    Represents a chunk of synthesized speech audio.

    Keeping the audio metadata with the bytes gives the rest of
    the application a consistent audio contract regardless of
    which TTS provider is used.
    """

    # Raw PCM audio bytes.
    data: bytes

    # Number of samples per second.
    sample_rate: int

    # Number of audio channels.
    channels: int = 1

    # Audio encoding.
    #
    # PCM 16-bit little-endian is represented as "pcm_s16le".
    encoding: str = "pcm_s16le"


class TTSProvider(ABC):
    """
    Common interface for all Agni AI TTS providers.

    Possible implementations:

        ElevenLabsTTSProvider
        OtherTTSProvider
        LocalTTSProvider

    The caller of this interface does not need to know which
    provider is being used.
    """

    @abstractmethod
    async def synthesize(
        self,
        text: str,
    ) -> AsyncIterator[TTSAudioChunk]:
        """
        Convert text into a stream of speech audio chunks.

        Args:
            text:
                Text that should be spoken.

        Yields:
            TTSAudioChunk objects containing synthesized audio.

        The method is asynchronous so providers that support
        streaming can produce audio incrementally.
        """

        raise NotImplementedError(
            "TTS providers must implement synthesize()"
        )