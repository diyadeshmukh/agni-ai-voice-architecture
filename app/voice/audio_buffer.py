"""
Agni AI - Audio Chunk Buffer

Collects realtime PCM audio frames from LiveKit and groups them
into fixed-duration chunks before sending them to the STT layer.

Flow:

    LiveKit AudioFrame
            ↓
       AudioChunkBuffer
            ↓
       2-second PCM chunk
            ↓
         STTAdapter
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AudioChunkBuffer:
    """
    Buffers PCM16 audio until the configured chunk duration is reached.

    The buffer works with raw PCM bytes and does not know anything
    about LiveKit, FastAPI, or Deepgram.
    """

    sample_rate: int = 48_000
    channels: int = 1
    chunk_duration_seconds: float = 2.0

    def __post_init__(self) -> None:
        # PCM16 uses 2 bytes per sample.
        self.bytes_per_sample = 2

        # Calculate the number of samples needed for one chunk.
        self.target_samples = int(
            self.sample_rate
            * self.chunk_duration_seconds
        )

        self._buffer = bytearray()

    @property
    def target_bytes(self) -> int:
        """Return the target size of one audio chunk in bytes."""

        return (
            self.target_samples
            * self.channels
            * self.bytes_per_sample
        )

    def add(self, pcm_bytes: bytes) -> bytes | None:
        """
        Add PCM audio to the buffer.

        Returns:
            A complete audio chunk when enough data has been
            collected; otherwise None.

        Extra data is retained for the next chunk.
        """

        if not pcm_bytes:
            return None

        self._buffer.extend(pcm_bytes)

        if len(self._buffer) < self.target_bytes:
            return None

        # Extract exactly one chunk.
        chunk = bytes(
            self._buffer[: self.target_bytes]
        )

        # Keep any remaining audio for the next chunk.
        del self._buffer[: self.target_bytes]

        return chunk

    def flush(self) -> bytes | None:
        """
        Return any remaining audio when the stream ends.

        This allows us to avoid silently losing the final partial chunk.
        """

        if not self._buffer:
            return None

        chunk = bytes(self._buffer)
        self._buffer.clear()

        return chunk

    def clear(self) -> None:
        """Discard any buffered audio."""

        self._buffer.clear()