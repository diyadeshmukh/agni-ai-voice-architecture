"""
Agni AI - ElevenLabs TTS Provider

Concrete TTS implementation for the Agni AI TTSProvider interface.

Supports two modes:

1. Standard text -> streaming audio
2. Streaming LLM text -> ElevenLabs WebSocket -> streaming audio

Realtime optimized flow:

    OpenAI text chunks
          ↓
    ElevenLabs TTS WebSocket
          ↓
    Streaming PCM16 audio
          ↓
    TTSAudioChunk
          ↓
    LiveKitAudioOutput
"""

from __future__ import annotations

import asyncio
import base64
import json
import os

from typing import AsyncIterator

import websockets

from dotenv import load_dotenv
from elevenlabs import AsyncElevenLabs

from app.voice.tts_provider import (
    TTSProvider,
    TTSAudioChunk,
)


load_dotenv(
    ".env.local",
    override=True,
)


class ElevenLabsTTSProvider(TTSProvider):
    """
    ElevenLabs implementation of the Agni AI TTS provider.

    Standard synthesize():
        Complete text
            ->
        ElevenLabs HTTP streaming
            ->
        PCM16 audio

    synthesize_streaming_text():
        Streaming LLM text
            ->
        ElevenLabs bidirectional WebSocket
            ->
        PCM16 audio

    Output format:

        PCM16
        16 kHz
        mono
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
            or os.getenv(
                "ELEVENLABS_API_KEY"
            )
        )

        if not self.api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is missing "
                "from .env.local"
            )

        self.voice_id = (
            voice_id
            or os.getenv(
                "ELEVENLABS_VOICE_ID"
            )
        )

        if not self.voice_id:
            raise RuntimeError(
                "ELEVENLABS_VOICE_ID is missing "
                "from .env.local"
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

    # ------------------------------------------------------------------
    # Standard complete-text streaming
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
    ) -> AsyncIterator[TTSAudioChunk]:
        """
        Convert a complete text string into streaming PCM16 audio.

        This method is kept for existing standalone TTS tests
        and callers that already have the full response text.
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

            if not isinstance(
                chunk,
                bytes,
            ):
                continue

            if not chunk:
                continue

            yield TTSAudioChunk(
                data=chunk,
                sample_rate=16_000,
                channels=1,
                encoding="pcm_s16le",
            )

    # ------------------------------------------------------------------
    # Realtime streaming-text TTS
    # ------------------------------------------------------------------

    async def synthesize_streaming_text(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[TTSAudioChunk]:
        """
        Stream text into ElevenLabs while the LLM is still generating.

        This uses ElevenLabs' bidirectional TTS WebSocket.

        OpenAI:
            text chunk
            text chunk
            text chunk
                ↓
        ElevenLabs WebSocket
                ↓
        audio chunk
        audio chunk
        audio chunk

        The WebSocket connection and OpenAI generation can therefore
        overlap instead of waiting for the complete LLM response.
        """

        websocket_url = (
            "wss://api.elevenlabs.io/"
            f"v1/text-to-speech/"
            f"{self.voice_id}/stream-input"
            f"?model_id={self.model_id}"
            f"&output_format={self.output_format}"
        )

        async with websockets.connect(
            websocket_url
        ) as websocket:

            # ----------------------------------------------------------
            # Initialize ElevenLabs WebSocket
            # ----------------------------------------------------------

            await websocket.send(
                json.dumps(
                    {
                        "text": " ",
                        "xi_api_key": self.api_key,

                        # Lower first generation threshold than the
                        # default while keeping enough text context
                        # for reasonable speech quality.
                        "generation_config": {
                            "chunk_length_schedule": [
                                50,
                                120,
                                160,
                                290,
                            ]
                        },
                    }
                )
            )

            # ----------------------------------------------------------
            # Send streamed LLM text concurrently
            # ----------------------------------------------------------

            async def send_text() -> None:
                """
                Consume OpenAI text chunks and forward them to
                ElevenLabs.

                Small incomplete word fragments are temporarily
                buffered so ElevenLabs is not given artificially
                broken words.
                """

                pending_text = ""

                try:

                    async for text_chunk in text_stream:

                        if not text_chunk:
                            continue

                        pending_text += text_chunk

                        # Find the latest safe whitespace boundary.
                        boundary_positions = [
                            pending_text.rfind(" "),
                            pending_text.rfind("\n"),
                            pending_text.rfind("\t"),
                        ]

                        boundary = max(
                            boundary_positions
                        )

                        if boundary < 0:
                            continue

                        ready_text = (
                            pending_text[
                                : boundary + 1
                            ]
                        )

                        pending_text = (
                            pending_text[
                                boundary + 1 :
                            ]
                        )

                        if ready_text:

                            await websocket.send(
                                json.dumps(
                                    {
                                        "text": ready_text,
                                    }
                                )
                            )

                    # --------------------------------------------------
                    # Flush any final incomplete word / short response
                    # --------------------------------------------------

                    if pending_text:

                        await websocket.send(
                            json.dumps(
                                {
                                    "text": pending_text,
                                    "flush": True,
                                }
                            )
                        )

                finally:

                    # Empty text marks end of the input sequence
                    # and forces any remaining buffered audio.
                    try:

                        await websocket.send(
                            json.dumps(
                                {
                                    "text": "",
                                }
                            )
                        )

                    except Exception:
                        pass

            sender_task = asyncio.create_task(
                send_text()
            )

            try:

                # ------------------------------------------------------
                # Receive ElevenLabs audio while text is still arriving
                # ------------------------------------------------------

                while True:

                    raw_message = (
                        await websocket.recv()
                    )

                    message = json.loads(
                        raw_message
                    )

                    # ElevenLabs can return API errors inside
                    # the WebSocket message.
                    if message.get("error"):

                        raise RuntimeError(
                            "ElevenLabs WebSocket error: "
                            f"{message['error']}"
                        )

                    audio_base64 = (
                        message.get(
                            "audio"
                        )
                    )

                    if audio_base64:

                        audio_bytes = (
                            base64.b64decode(
                                audio_base64
                            )
                        )

                        if audio_bytes:

                            yield TTSAudioChunk(
                                data=audio_bytes,
                                sample_rate=16_000,
                                channels=1,
                                encoding="pcm_s16le",
                            )

                    # TTS WebSocket normally returns isFinal.
                    # Supporting both forms makes parsing robust.
                    is_final = bool(
                        message.get(
                            "isFinal",
                            False,
                        )
                        or message.get(
                            "is_final",
                            False,
                        )
                    )

                    if is_final:
                        break

            finally:

                # Make sure the text-producing task is finished.
                if not sender_task.done():

                    try:
                        await sender_task

                    except asyncio.CancelledError:
                        raise

                else:

                    sender_task.result()