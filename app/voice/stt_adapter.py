"""
Agni AI - Speech-to-Text Adapter

This module converts audio received from LiveKit into the format
expected by the existing STT backend.

LiveKit audio:
    - 48 kHz
    - mono
    - PCM 16-bit

STT backend:
    - WAV
    - 16 kHz
    - mono
    - PCM 16-bit

The adapter is intentionally independent of Deepgram.
The STT backend remains responsible for the actual STT provider.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import requests
from scipy.signal import resample_poly


class STTAdapter:
    """
    Converts PCM audio to the backend's WAV format and sends it
    to the existing STT endpoint.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000/audio",
        input_sample_rate: int = 48_000,
        output_sample_rate: int = 16_000,
        channels: int = 1,
    ) -> None:
        self.endpoint = endpoint
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.channels = channels

    def convert_to_wav(self, pcm_bytes: bytes) -> bytes:
        """
        Convert raw PCM16 audio from LiveKit into the WAV format
        required by the STT backend.

        Input:
            48 kHz, mono, PCM16

        Output:
            WAV containing 16 kHz, mono, PCM16
        """

        if not pcm_bytes:
            raise ValueError("Cannot convert empty audio")

        # Interpret the incoming bytes as signed 16-bit PCM samples.
        samples = np.frombuffer(
            pcm_bytes,
            dtype=np.int16,
        )

        if samples.size == 0:
            raise ValueError("No PCM samples received")

        # Resample the audio from 48 kHz to 16 kHz.
        resampled = resample_poly(
            samples,
            up=self.output_sample_rate,
            down=self.input_sample_rate,
        )

        # Make sure the final values fit inside int16.
        resampled_pcm = np.clip(
            resampled,
            -32768,
            32767,
        ).astype(np.int16)

        # Build the WAV file completely in memory.
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # PCM16 = 2 bytes/sample
            wav_file.setframerate(self.output_sample_rate)
            wav_file.writeframes(
                resampled_pcm.tobytes()
            )

        return wav_buffer.getvalue()

    def send_audio(
        self,
        pcm_bytes: bytes,
        filename: str,
    ) -> str:
        """
        Convert a PCM audio chunk to WAV and send it to
        the existing /audio endpoint.

        The backend expects:
            multipart/form-data
            field name = file

        Returns:
            Transcript returned by the backend.
        """

        wav_bytes = self.convert_to_wav(pcm_bytes)

        response = requests.post(
            self.endpoint,
            files={
                "file": (
                    filename,
                    wav_bytes,
                    "audio/wav",
                )
            },
            timeout=30,
        )

        response.raise_for_status()

        result = response.json()

        transcript = result.get("transcript")

        if transcript is None:
            raise RuntimeError(
                "STT backend response does not contain "
                "'transcript'"
            )

        return transcript