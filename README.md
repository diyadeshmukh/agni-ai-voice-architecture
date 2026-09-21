# Agni AI — Voice Architecture & Real-Time Voice Pipeline

> A modular real-time voice pipeline for Agni AI using **LiveKit**, **Deepgram**, and **ElevenLabs**.

---

## Overview

Agni AI is being developed as a real-time voice AI system with a modular provider-based architecture.

This repository contains the current voice architecture, streaming Speech-to-Text (STT) implementation, Text-to-Speech (TTS) provider layer, and LiveKit proof-of-concept (POC) components.

The main goal is to keep the voice pipeline modular so that individual providers can be replaced or integrated independently.

### Current voice flow

```text
Caller / Microphone
        │
        ▼
     LiveKit
        │
        ▼
  Streaming STT
   (Deepgram)
        │
        ▼
    Transcript
        │
        ▼
      LLM
        │
        ▼
 AI Response Text
        │
        ▼
      TTS
  (ElevenLabs)
        │
        ▼
 Streaming PCM Audio
        │
        ▼
     LiveKit
        │
        ▼
Caller / Speaker
```

The current repository has working STT and TTS components. LLM and telephony integration are planned for the next stages.

---

# Current Status

| Component | Status | Description |
|---|---|---|
| LiveKit connection | ✅ Working | Room connection and audio transport |
| LiveKit audio publishing | ✅ Working | Publishes audio tracks to a room |
| LiveKit audio subscription | ✅ Working | Receives remote audio tracks |
| Microphone capture | ✅ Working | Local microphone testing |
| Streaming STT | ✅ Working | Deepgram real-time transcription |
| Interim transcripts | ✅ Working | Displays changing speech recognition results |
| Final transcripts | ✅ Working | Displays finalized recognition results |
| TTS provider interface | ✅ Working | Provider-independent TTS contract |
| ElevenLabs TTS | ✅ Working | Streaming text-to-speech |
| LiveKit TTS output | ✅ Working | Publishes generated speech into LiveKit |
| Local TTS playback | ✅ Working | TTS listener plays received audio |
| Custom TTS text | ✅ Working | Command-line text can be synthesized |
| LLM integration | 🔄 Planned | Connect AI response generation |
| STT → LLM → TTS | 🔄 Planned | Complete conversational pipeline |
| Telephony / Exotel | 🔄 Planned | Phone-call integration |
| Barge-in / interruption | 🔄 Planned | Interrupt active AI speech |
| Production orchestration | 🔄 Planned | Complete voice session runtime |

---

# Architecture

The repository follows a layered architecture.

```text
                    ┌───────────────────────┐
                    │        Caller         │
                    │   Phone / Microphone  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │        LiveKit        │
                    │    Voice Transport    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │         STT           │
                    │       Deepgram        │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │         LLM           │
                    │    AI Response Text   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │         TTS           │
                    │      ElevenLabs       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │        LiveKit        │
                    │     Audio Output      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │        Caller         │
                    └───────────────────────┘
```

---

# Streaming STT Architecture

The current STT path is:

```text
Microphone
    ↓
LiveKit
    ↓
AudioFrame
    ↓
Audio conversion / resampling
    ↓
16 kHz PCM16
    ↓
STTStreamAdapter
    ↓
Deepgram Streaming STT
    ↓
Interim / Final Transcript
```

The implementation is designed for continuous real-time audio instead of waiting for a complete recording.

### STT responsibilities

The STT layer handles:

- Streaming audio input
- Deepgram connection
- Audio transmission
- Interim results
- Final results
- Stream finalization
- STT event handling

---

# TTS Architecture

The current TTS path is:

```text
AI Response Text
       ↓
   TTSProvider
       ↓
ElevenLabsTTSProvider
       ↓
 Streaming PCM16
       ↓
   TTSAudioChunk
       ↓
LiveKitAudioOutput
       ↓
 LiveKit AudioSource
       ↓
 LocalAudioTrack
       ↓
    LiveKit
       ↓
 Remote Listener / Caller
```

The TTS layer is provider-independent.

This means the application can use an alternative TTS provider later without changing the rest of the voice pipeline.

---

# Provider Abstraction

The project defines a common TTS provider interface in:

```text
app/voice/tts_provider.py
```

The interface is based on:

```python
async def synthesize(
    text: str,
) -> AsyncIterator[TTSAudioChunk]:
    ...
```

The current implementation is:

```text
app/voice/elevenlabs_tts_provider.py
```

The architecture can later support:

```text
                 TTSProvider
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
      ElevenLabs  Provider B  Provider C
```

---

# Audio Contract

The TTS provider layer uses a standard audio representation:

```python
@dataclass
class TTSAudioChunk:
    data: bytes
    sample_rate: int
    channels: int = 1
    encoding: str = "pcm_s16le"
```

### Current audio format

| Property | Value |
|---|---|
| Encoding | PCM 16-bit little-endian |
| Sample rate | 16,000 Hz |
| Channels | Mono |
| TTS output format | `pcm_16000` |

Keeping the audio format explicit makes it easier to integrate TTS output with LiveKit and other downstream components.

---

# Project Structure

```text
agni-ai/
│
├── README.md
├── .env
├── .env.local
├── .gitignore
├── mic_test.py
│
├── app/
│   ├── __init__.py
│   ├── database.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── stt.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── stt_service.py
│   │
│   └── voice/
│       ├── __init__.py
│       ├── audio_buffer.py
│       ├── elevenlabs_tts_provider.py
│       ├── livekit_audio_output.py
│       ├── stt_adapter.py
│       ├── stt_stream_adapter.py
│       └── tts_provider.py
│
├── docs/
│   ├── Agni_AI_Voice_Architecture_Notes.docx
│   └── TTS_PROVIDER_INTERFACE.md
│
├── livekit_poc/
│   ├── __init__.py
│   ├── audio_publisher.py
│   ├── audio_subscriber.py
│   ├── connect_room.py
│   ├── tts_listener.py
│   └── tts_publisher.py
│
└── services/
    └── stt_service/
        ├── __init__.py
        └── main.py
```

---

# Main Components

## `app/voice/tts_provider.py`

Defines the common TTS provider interface.

It allows the main voice pipeline to use a provider through a common contract instead of directly depending on a vendor SDK.

---

## `app/voice/elevenlabs_tts_provider.py`

Concrete ElevenLabs implementation of the TTS provider interface.

Responsibilities include:

- ElevenLabs client initialization
- Authentication using environment variables
- Text-to-speech generation
- Streaming audio
- Conversion of provider output to `TTSAudioChunk`

Current configuration:

```text
Model:
eleven_flash_v2_5

Output:
pcm_16000

Channels:
1

Sample rate:
16,000 Hz
```

---

## `app/voice/livekit_audio_output.py`

Converts application-level TTS audio chunks into LiveKit audio frames.

Flow:

```text
TTSAudioChunk
      ↓
PCM16 AudioFrame
      ↓
LiveKit AudioSource
      ↓
LocalAudioTrack
      ↓
LiveKit Room
```

This keeps LiveKit-specific audio logic separate from the TTS provider.

---

## `app/services/stt_service.py`

Contains the current streaming Deepgram STT implementation.

Responsibilities include:

- Creating the Deepgram streaming connection
- Sending audio
- Receiving transcription events
- Processing interim results
- Processing final results
- Finalizing the stream

---

## `app/voice/stt_stream_adapter.py`

Provides an adapter around the streaming STT connection.

The adapter exposes a simpler application-level flow:

```text
Connect
   ↓
Send Audio
   ↓
Receive Events
   ↓
Close
```

---

## `app/api/v1/stt.py`

FastAPI WebSocket route used by the streaming STT service.

Current local endpoint:

```text
ws://127.0.0.1:8000/api/v1/stt/stream
```

---

## `services/stt_service/main.py`

FastAPI service entry point.

Current routes include:

```text
GET /
POST /audio
WebSocket /api/v1/stt/stream
```

The `/audio` route supports the existing file-based STT flow, while the WebSocket route is used for streaming STT.

---

# LiveKit POC

The `livekit_poc/` directory contains standalone programs used to validate pieces of the voice architecture.

These programs allow STT and TTS to be tested independently before integrating the complete AI voice agent.

---

# LiveKit + STT POC

The main LiveKit STT subscriber is:

```text
livekit_poc/audio_subscriber.py
```

It:

1. Connects to a LiveKit room.
2. Subscribes to incoming audio.
3. Receives LiveKit audio frames.
4. Converts/resamples the audio when required.
5. Sends audio to the streaming STT adapter.
6. Displays interim and final transcripts.

### Run

From the project root:

```powershell
conda activate agni-ai
python -m livekit_poc.audio_subscriber
```

---

# LiveKit + TTS POC

The TTS POC uses two programs:

```text
livekit_poc/tts_publisher.py
livekit_poc/tts_listener.py
```

The publisher generates speech and publishes the audio into LiveKit.

The listener subscribes to the generated track and plays the received audio through the local speaker.

### TTS flow

```text
Text
 ↓
ElevenLabs
 ↓
PCM16 Streaming Audio
 ↓
LiveKit AudioSource
 ↓
LiveKit AudioTrack
 ↓
TTS Listener
 ↓
Laptop Speaker
```

---

# Running the TTS Listener

Open a terminal from the project root and activate the environment:

```powershell
conda activate agni-ai
```

Run:

```powershell
python -m livekit_poc.tts_listener
```

Expected behavior:

```text
========================================================================
AGNI AI - TTS LISTENER
========================================================================

Connecting to LiveKit...
Connected to LiveKit.

Waiting for 'voice-output'...
Keep this terminal running while the TTS publisher is active.
```

The listener waits for the LiveKit audio track:

```text
voice-output
```

---

# Running the TTS Publisher

Open a second terminal.

Activate the environment:

```powershell
conda activate agni-ai
```

Run:

```powershell
python -m livekit_poc.tts_publisher
```

The publisher:

1. Loads the TTS provider.
2. Connects to LiveKit.
3. Publishes the `voice-output` track.
4. Sends the text to ElevenLabs.
5. Receives streaming PCM16 audio chunks.
6. Sends the chunks to LiveKit.
7. Waits for audio playout.
8. Disconnects.

---

# Custom TTS Text

The TTS publisher accepts custom text from the command line.

Example:

```powershell
python -m livekit_poc.tts_publisher "Hello, this is a custom Agni AI response being converted to speech and published through LiveKit."
```

This allows different response texts to be tested without modifying the source code.

---

# Development Environment

The current project is developed on Windows using Anaconda / Conda.

Recommended development environment:

| Component | Current setup |
|---|---|
| Operating system | Windows |
| Environment | Conda |
| Environment name | `agni-ai` |
| Python | 3.11.x |
| STT | Deepgram |
| TTS | ElevenLabs |
| Voice transport | LiveKit |
| API framework | FastAPI |

Activate the environment:

```powershell
conda activate agni-ai
```

Check Python:

```powershell
python --version
```

---

# Environment Variables

Local configuration is stored in:

```text
.env.local
```

Typical configuration:

```env
DEEPGRAM_API_KEY=<your-key>

ELEVENLABS_API_KEY=<your-key>
ELEVENLABS_VOICE_ID=<your-voice-id>
ELEVENLABS_MODEL_ID=eleven_flash_v2_5
ELEVENLABS_OUTPUT_FORMAT=pcm_16000

LIVEKIT_URL=<your-livekit-url>
LIVEKIT_API_KEY=<your-livekit-api-key>
LIVEKIT_API_SECRET=<your-livekit-api-secret>

LIVEKIT_ROOM_NAME=agni-ai-voice-poc

LIVEKIT_TTS_PUBLISHER_IDENTITY=agni-tts-publisher
LIVEKIT_TTS_LISTENER_IDENTITY=agni-tts-listener
```

> **Important:** Never put real API keys, API secrets, access tokens, or passwords in this README or in source code.

---

# Security

The following files may contain credentials:

```text
.env
.env.local
```

These files should remain excluded from version control.

Before committing changes, verify:

```powershell
git status
```

and make sure no credentials are being staged.

Never publish:

- API keys
- API secrets
- Access tokens
- Passwords
- Private credentials

---

# FastAPI STT Service

The FastAPI STT service can be started with:

```powershell
uvicorn services.stt_service.main:app --reload
```

Local server:

```text
http://127.0.0.1:8000
```

### Available routes

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service root |
| `POST` | `/audio` | File-based STT |
| `WebSocket` | `/api/v1/stt/stream` | Streaming STT |

Streaming WebSocket URL:

```text
ws://127.0.0.1:8000/api/v1/stt/stream
```

---

# Streaming Design

The project uses streaming instead of waiting for complete audio whenever possible.

## STT streaming

```text
Audio Frame 1 ─┐
Audio Frame 2 ─┤
Audio Frame 3 ─┤──→ Deepgram
Audio Frame 4 ─┤
Audio Frame 5 ─┘
```

## TTS streaming

```text
Text
 ↓
Audio Chunk 1 ─┐
Audio Chunk 2 ─┤
Audio Chunk 3 ─┤──→ LiveKit
Audio Chunk 4 ─┤
Audio Chunk 5 ─┘
```

Streaming makes it possible to begin processing or playing audio before the complete response has been generated.

---

# Interruption / Barge-In Design

The final voice agent is intended to support conversational interruption.

Target behavior:

```text
AI is speaking
      ↓
User starts speaking
      ↓
STT detects user speech
      ↓
Current TTS playback stops
      ↓
New transcript is processed
      ↓
LLM generates new response
      ↓
TTS generates new speech
      ↓
New response is played
```

The complete interruption and session orchestration logic is not yet finalized in the current repository.

---

# Target End-to-End Voice Flow

The final intended architecture is:

```text
User Speech
     │
     ▼
  LiveKit
     │
     ▼
Deepgram STT
     │
     ▼
 Transcript
     │
     ▼
    LLM
     │
     ▼
AI Response Text
     │
     ▼
TTSProvider
     │
     ▼
 ElevenLabs
     │
     ▼
Streaming PCM16
     │
     ▼
  LiveKit
     │
     ▼
   Caller
```

---

# Separation of Responsibilities

## STT

Responsible for:

- Receiving audio
- Streaming audio to the STT provider
- Speech recognition
- Interim results
- Final results

## LLM

Responsible for:

- Receiving transcript text
- Generating AI responses
- Streaming response text when supported

## TTS

Responsible for:

- Receiving AI-generated text
- Converting text to speech
- Streaming generated audio
- Providing a consistent audio contract

## LiveKit

Responsible for:

- Real-time voice transport
- Audio publishing
- Audio subscription
- Audio track management

## Telephony

Responsible for:

- Connecting external phone callers
- SIP / telephony transport
- Connecting phone calls to the voice pipeline

---

# Backend and Database Integration

The larger Agni AI system will also include backend and database components.

These may include:

- FastAPI application modules
- PostgreSQL
- SQLAlchemy
- Alembic
- Database models
- Authentication / application services
- Call and session management

These components are expected to be integrated with the voice layer as the team combines the individual modules.

When integrating them, dependency versions and existing voice components should be checked first to avoid breaking the working STT/TTS implementation.

---

# Documentation

Additional project documentation is available in the `docs/` directory.

Current documents:

- [`docs/Agni_AI_Voice_Architecture_Notes.docx`](docs/Agni_AI_Voice_Architecture_Notes.docx)
- [`docs/TTS_PROVIDER_INTERFACE.md`](docs/TTS_PROVIDER_INTERFACE.md)

---

# Development Principles

## Modularity

Each component should have a clear responsibility.

## Provider Independence

The application should depend on provider interfaces instead of vendor-specific implementations whenever practical.

## Streaming First

Audio and text should be processed incrementally whenever the provider supports streaming.

## Clear Audio Contracts

Audio format, sample rate, channel count, and encoding should be explicit between components.

## Independent Testing

Providers and transport components should be testable independently before full end-to-end integration.

## Integration Safety

Existing working components should not be replaced without checking compatibility, dependencies, and team ownership.

---

# Git Workflow

From the project root:

```powershell
cd C:\Users\91940\Projects\agni-ai
```

Check the repository:

```powershell
git status
```

Review changes:

```powershell
git diff
```

Stage changes:

```powershell
git add .
```

Commit:

```powershell
git commit -m "Add README documentation"
```

Push:

```powershell
git push origin main
```

Before committing, always verify that `.env`, `.env.local`, and other sensitive files are not being committed.

---

# Verified TTS POC Flow

The current TTS POC has successfully validated:

```text
Text
 ↓
ElevenLabs TTS Provider
 ↓
Streaming PCM16 Audio
 ↓
TTSAudioChunk
 ↓
LiveKitAudioOutput
 ↓
LiveKit AudioSource
 ↓
LiveKit LocalAudioTrack
 ↓
LiveKit Room
 ↓
TTS Listener
 ↓
Laptop Speaker
```

The publisher can also accept custom text from the command line.

---

# Verified STT POC Flow

The current STT POC has validated:

```text
Microphone
 ↓
LiveKit
 ↓
AudioFrame
 ↓
Audio Conversion / Resampling
 ↓
Deepgram Streaming STT
 ↓
Interim Transcript
 ↓
Final Transcript
```

---

# Next Integration Step

The next major integration is:

```text
STT
 ↓
LLM
 ↓
TTS
 ↓
LiveKit
```

The goal is to connect the currently working streaming STT and TTS components through the LLM.

The resulting flow will be:

```text
User Speech
     ↓
   LiveKit
     ↓
 Deepgram STT
     ↓
 Transcript
     ↓
     LLM
     ↓
AI Response
     ↓
 TTSProvider
     ↓
 ElevenLabs
     ↓
Streaming Audio
     ↓
   LiveKit
     ↓
    User
```

---

# Current Milestone

The repository currently provides a working foundation for the Agni AI real-time voice pipeline:

- ✅ LiveKit voice transport
- ✅ Streaming Deepgram STT
- ✅ TTS provider abstraction
- ✅ ElevenLabs streaming TTS
- ✅ LiveKit TTS audio output
- ✅ Local TTS listener/playback
- ✅ Standalone STT/TTS POCs

The next stage is to integrate the LLM and complete the conversational voice loop.

---

## Agni AI Voice Pipeline

```text
                AGNI AI

        ┌───────────────────┐
        │   User / Caller   │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │      LiveKit      │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   Deepgram STT    │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │        LLM        │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │    TTSProvider    │
        │     ElevenLabs    │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │      LiveKit      │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   User / Caller   │
        └───────────────────┘
```

---

**Agni AI — Voice Architecture & Provider Layer**
