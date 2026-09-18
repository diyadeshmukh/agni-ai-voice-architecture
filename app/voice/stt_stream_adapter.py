"""
Agni AI - Streaming STT Adapter

Connects the LiveKit audio pipeline to the team's streaming STT endpoint.

The adapter sends continuous raw PCM16 audio over WebSocket and receives
streaming transcription/VAD events from the STT service.

Expected audio format:
    Sample rate: 16 kHz
    Channels: 1 (mono)
    Encoding: linear16 / PCM16

Expected server endpoint:
    ws://127.0.0.1:8000/api/v1/stt/stream
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets


class STTStreamAdapter:
    """
    WebSocket client for the streaming STT service.

    Responsibilities:
        1. Open the STT WebSocket connection.
        2. Send continuous PCM16 audio chunks.
        3. Receive partial/final transcript and VAD events.
        4. Close the connection cleanly.
    """

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

        self.websocket: Any | None = None
        self.receive_task: asyncio.Task | None = None

        # Events received from the STT server are placed here so that
        # the LiveKit audio receiving loop is not blocked by STT responses.
        self.event_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def connect(self) -> None:
        """Open the WebSocket connection to the streaming STT service."""

        if self.websocket is not None:
            return

        self.websocket = await websockets.connect(self.endpoint)

        print(f"Connected to streaming STT: {self.endpoint}")

        # Start a background task to continuously receive transcripts/events.
        self.receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        """
        Continuously receive JSON events from the STT server.

        Examples:
            {
                "type": "partial",
                "transcript": "hello"
            }

            {
                "type": "final",
                "transcript": "hello world"
            }

            {
                "type": "speech_started"
            }

            {
                "type": "utterance_end"
            }
        """

        try:
            assert self.websocket is not None

            async for message in self.websocket:
                try:
                    payload = json.loads(message)

                    if isinstance(payload, dict):
                        await self.event_queue.put(payload)

                except json.JSONDecodeError:
                    print("Warning: received non-JSON message from STT service.")

        except asyncio.CancelledError:
            # Normal shutdown of the receiver task.
            raise

        except Exception as exc:
            print(f"STT receive error: {exc}")

            await self.event_queue.put(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )

    async def send_audio(self, audio_data: bytes) -> None:
        """
        Send one PCM16 audio chunk to the streaming STT service.

        Important:
            audio_data must already be:
                - 16 kHz
                - mono
                - PCM16 / linear16
        """

        if self.websocket is None:
            raise RuntimeError("STT streaming connection is not open.")

        if not audio_data:
            return

        await self.websocket.send(audio_data)

    async def receive_event(self) -> dict:
        """
        Wait for the next transcription/VAD event.

        The caller can use this from a separate async task while audio
        continues to be sent through send_audio().
        """

        return await self.event_queue.get()

    async def close(self) -> None:
        """Close the STT WebSocket and stop the receiver task."""

        if self.receive_task is not None:
            self.receive_task.cancel()

            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass

            self.receive_task = None

        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None

        print("Disconnected from streaming STT.")