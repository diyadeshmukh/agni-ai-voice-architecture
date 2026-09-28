"""
Agni AI - ElevenLabs TTS Provider

Concrete TTS implementation for the Agni AI TTSProvider interface.

Uses ElevenLabs Eleven v3 Conversational through the current
Text-to-Dialogue WebSocket API.

Supports:

1. Complete text -> streaming PCM16 audio
2. Streaming LLM text -> streaming PCM16 audio

Realtime optimized flow:

    OpenAI text chunks
          ↓
    ElevenLabs Text-to-Dialogue WebSocket
          ↓
    Eleven v3 Conversational
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

from websockets.exceptions import (
    ConnectionClosedOK,
)

from dotenv import load_dotenv

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

    Both complete-text and streaming-text synthesis use the
    ElevenLabs Text-to-Dialogue WebSocket with
    eleven_v3_conversational.

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
                "eleven_v3_conversational",
            )
        )

        # Text-to-Dialogue WebSocket is for
        # Eleven v3 models only.
        if not self.model_id.startswith(
            "eleven_v3"
        ):

            raise RuntimeError(
                "ElevenLabs Text-to-Dialogue "
                "WebSocket requires a v3 model. "
                "Use ELEVENLABS_MODEL_ID="
                "eleven_v3_conversational."
            )

        self.output_format = (
            output_format
            or os.getenv(
                "ELEVENLABS_OUTPUT_FORMAT",
                "pcm_16000",
            )
        )

        # Agni's current LiveKit output path expects
        # raw PCM16 audio at 16 kHz.
        if self.output_format != "pcm_16000":

            raise RuntimeError(
                "Agni AI currently expects "
                "ELEVENLABS_OUTPUT_FORMAT=pcm_16000 "
                "for the LiveKit audio pipeline."
            )

    # ------------------------------------------------------------------
    # Standard complete-text streaming
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
    ) -> AsyncIterator[TTSAudioChunk]:
        """
        Convert complete text into streaming PCM16 audio.

        Complete-text synthesis deliberately uses the same
        Eleven v3 Conversational Text-to-Dialogue path as
        realtime LLM streaming.

        This keeps one TTS architecture for:

            English
            Hindi
            Hinglish
            Marathi
        """

        text = text.strip()

        if not text:
            return

        async def text_stream(
        ) -> AsyncIterator[str]:

            yield text

        async for audio_chunk in (
            self.synthesize_streaming_text(
                text_stream()
            )
        ):

            yield audio_chunk

    # ------------------------------------------------------------------
    # Realtime streaming-text TTS
    # ------------------------------------------------------------------

    async def synthesize_streaming_text(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[TTSAudioChunk]:
        """
        Stream LLM text into ElevenLabs while OpenAI
        is still generating.

        Current ElevenLabs path:

            /v1/text-to-dialogue/stream-input

        Protocol:

            1. Register the configured voice.
            2. Send incremental text through "inputs".
            3. Receive base64 PCM audio concurrently.
            4. Send close_socket=True after text ends.
            5. Continue receiving until is_final=True.

        ElevenLabs performs its own contextual buffering.
        Agni only buffers incomplete word fragments locally
        so token boundaries are not sent as broken words.
        """

        websocket_url = (
            "wss://api.elevenlabs.io/"
            "v1/text-to-dialogue/stream-input"
            f"?model_id={self.model_id}"
            f"&output_format={self.output_format}"
        )

        async with websockets.connect(
            websocket_url
        ) as websocket:

            # ----------------------------------------------------------
            # Register voice / authenticate
            #
            # eleven_v3_conversational currently allows one
            # registered voice per WebSocket connection.
            # ----------------------------------------------------------

            await websocket.send(
                json.dumps(
                    {
                        "voices": [
                            self.voice_id
                        ],
                        "xi_api_key": self.api_key,
                    }
                )
            )

            # ----------------------------------------------------------
            # Send streamed LLM text concurrently
            # ----------------------------------------------------------

            async def send_text() -> None:
                """
                Consume OpenAI text chunks and send word-aligned
                incremental text to ElevenLabs.

                OpenAI can produce partial token/word fragments.
                We therefore retain only the incomplete trailing
                word locally.

                ElevenLabs itself decides when enough context
                exists to start generating audio.
                """

                pending_text = ""

                try:

                    async for text_chunk in text_stream:

                        if not text_chunk:
                            continue

                        pending_text += text_chunk

                        # ----------------------------------------------
                        # Find the latest safe whitespace boundary
                        # ----------------------------------------------

                        boundary_positions = [
                            pending_text.rfind(
                                " "
                            ),
                            pending_text.rfind(
                                "\n"
                            ),
                            pending_text.rfind(
                                "\t"
                            ),
                        ]

                        boundary = max(
                            boundary_positions
                        )

                        # No complete word yet.
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

                        if not ready_text:
                            continue

                        # ----------------------------------------------
                        # Send incremental dialogue input
                        # ----------------------------------------------

                        await websocket.send(
                            json.dumps(
                                {
                                    "inputs": [
                                        {
                                            "text": (
                                                ready_text
                                            ),
                                            "voice_id": (
                                                self.voice_id
                                            ),
                                            "new_turn": False,
                                        }
                                    ]
                                }
                            )
                        )

                    # --------------------------------------------------
                    # Send final incomplete word / short text
                    # --------------------------------------------------

                    if pending_text:

                        await websocket.send(
                            json.dumps(
                                {
                                    "inputs": [
                                        {
                                            "text": (
                                                pending_text
                                            ),
                                            "voice_id": (
                                                self.voice_id
                                            ),
                                            "new_turn": False,
                                        }
                                    ]
                                }
                            )
                        )

                finally:

                    # --------------------------------------------------
                    # Finish this dialogue session
                    #
                    # close_socket causes ElevenLabs to:
                    #
                    # - flush remaining buffered text
                    # - generate remaining audio
                    # - send is_final=True
                    # - close the WebSocket
                    # --------------------------------------------------

                    try:

                        await websocket.send(
                            json.dumps(
                                {
                                    "close_socket": True,
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
                # Receive audio while OpenAI may still be generating
                # ------------------------------------------------------

                while True:

                    try:

                        raw_message = (
                            await websocket.recv()
                        )

                    except ConnectionClosedOK:

                        break

                    message = json.loads(
                        raw_message
                    )

                    # --------------------------------------------------
                    # ElevenLabs API error
                    # --------------------------------------------------

                    if message.get(
                        "error"
                    ):

                        raise RuntimeError(
                            "ElevenLabs "
                            "Text-to-Dialogue "
                            "WebSocket error: "
                            f"{message}"
                        )

                    # --------------------------------------------------
                    # Audio chunk
                    # --------------------------------------------------

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

                    # --------------------------------------------------
                    # Final session frame
                    #
                    # Do not stop at is_final_audio_for_turn.
                    #
                    # We close only after close_socket has caused
                    # ElevenLabs to flush every remaining audio byte.
                    # --------------------------------------------------

                    if message.get(
                        "is_final",
                        False,
                    ):

                        break

            finally:

                # ------------------------------------------------------
                # Make sure the text sender finished cleanly
                # ------------------------------------------------------

                if not sender_task.done():

                    try:

                        await sender_task

                    except asyncio.CancelledError:

                        raise

                else:

                    sender_task.result()