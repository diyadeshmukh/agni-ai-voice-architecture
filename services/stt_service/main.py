"""
Agni AI - Speech-to-Text Service

This service exposes the STT APIs used by the Agni AI voice pipeline.

Endpoints:
    GET  /
    POST /audio
    WS   /api/v1/stt/stream

The /audio endpoint handles individual WAV uploads.

The /api/v1/stt/stream endpoint handles continuous raw PCM16
audio streaming for the LiveKit voice pipeline.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile

from deepgram import DeepgramClient

from app.api.v1.stt import router as streaming_stt_router


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

# Load local development secrets/configuration.
# This file must never be committed to Git.
load_dotenv(".env.local", override=True)


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agni AI STT Service",
    description=(
        "Speech-to-text service for the Agni AI voice pipeline"
    ),
)


# ---------------------------------------------------------------------------
# Streaming STT router
# ---------------------------------------------------------------------------

# Registers:
#
#     WS /api/v1/stt/stream
#
# This endpoint uses the streaming Deepgram service implemented in:
#
#     app/services/stt_service.py
#
# The existing /audio endpoint below remains unchanged.
app.include_router(
    streaming_stt_router,
    prefix="/api/v1",
)


# ---------------------------------------------------------------------------
# Deepgram client for the existing /audio endpoint
# ---------------------------------------------------------------------------

deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

if not deepgram_api_key:
    raise RuntimeError(
        "DEEPGRAM_API_KEY is missing from .env.local"
    )

deepgram = DeepgramClient(
    api_key=deepgram_api_key
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
async def home():
    """
    Simple service health endpoint.
    """

    return {
        "message": "Agni AI STT service is running"
    }


# ---------------------------------------------------------------------------
# Audio transcription endpoint
# ---------------------------------------------------------------------------

@app.post("/audio")
async def receive_audio(
    file: UploadFile = File(...),
):
    """
    Receive a WAV audio chunk and send it to Deepgram.

    This is the existing chunk-based transcription endpoint.

    Expected audio:
        WAV
        PCM 16-bit
        16 kHz
        Mono
    """

    # Read the uploaded WAV file into memory.
    audio_data = await file.read()

    print(
        f"Received audio: {file.filename}"
    )

    print(
        f"Audio size: {len(audio_data)} bytes"
    )

    # Send the audio bytes to Deepgram for transcription.
    response = deepgram.listen.v1.media.transcribe_file(
        request=audio_data,
        model="nova-3",
        smart_format=True,
    )

    # Extract the transcript from the first channel/alternative.
    transcript = (
        response
        .results
        .channels[0]
        .alternatives[0]
        .transcript
    )

    print(
        f"Transcript: {transcript}"
    )

    return {
        "message": "Audio processed successfully",
        "filename": file.filename,
        "transcript": transcript,
    }