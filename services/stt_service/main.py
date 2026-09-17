"""
Agni AI - Speech-to-Text Service

This service exposes the existing STT API used by the Agni AI
voice pipeline.

Endpoint:
    POST /audio

Request:
    multipart/form-data
    field name: file

Expected audio:
    WAV
    PCM 16-bit
    16 kHz
    Mono

Processing:
    WAV audio
        ↓
    FastAPI
        ↓
    Deepgram Nova-3
        ↓
    Transcript
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from deepgram import DeepgramClient


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
    description="Speech-to-text service for the Agni AI voice pipeline",
)


# ---------------------------------------------------------------------------
# Deepgram client
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
    file: UploadFile = File(...)
):
    """
    Receive a WAV audio chunk and send it to Deepgram.

    The LiveKit integration will eventually send 2-second
    WAV chunks to this endpoint.
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