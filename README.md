# Agni AI — Voice Architecture & Real-Time Voice Pipeline

> A modular real-time voice pipeline for Agni AI using **LiveKit**, **Deepgram**, **OpenAI**, **ElevenLabs**, and **FastAPI**.

---

## Overview

Agni AI is being developed as a real-time voice AI system with a modular provider-based architecture.

This repository contains the current voice architecture, streaming Speech-to-Text (STT) implementation, OpenAI LLM provider layer, Text-to-Speech (TTS) provider layer, and LiveKit proof-of-concept (POC) components.

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
Completed Utterance
        │
        ▼
   OpenAI LLM
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

> **Important:** Interim STT transcripts are displayed locally but are **not sent to OpenAI**.
> Only completed utterances are sent to the LLM.

---

## Current Status

| Component | Status | Description |
|---|---|---|
| LiveKit connection | ✅ Working | Room connection and audio transport |
| LiveKit audio publishing | ✅ Working | Publishes audio tracks to a room |
| LiveKit audio subscription | ✅ Working | Receives remote audio tracks |
| Microphone capture | ✅ Working | Local microphone testing |
| Streaming STT | ✅ Working | Deepgram real-time transcription |
| Interim transcripts | ✅ Working | Displayed locally only |
| Final transcripts | ✅ Working | Final speech recognition results |
| Utterance detection | ✅ Working | Uses final transcript + utterance end |
| LLM provider interface | ✅ Working | Provider-independent LLM contract |
| OpenAI LLM | ✅ Working | Generates concise AI responses |
| STT → LLM integration | ✅ Working | One LLM call per completed utterance |
| TTS provider interface | ✅ Working | Provider-independent TTS contract |
| ElevenLabs TTS | ✅ Working | Streaming text-to-speech |
| LiveKit TTS output | ✅ Working | Publishes generated speech into LiveKit |
| Local TTS playback | ✅ Working | TTS listener plays received audio |
| STT → LLM → TTS | ✅ Working | End-to-end local voice flow verified |
| Response latency logging | ✅ Working | Measures main voice response stages |
| LLM diagnostic POC | ✅ Working | Standalone OpenAI test |
| Echo / feedback suppression | ✅ POC | Mic audio suppressed while AI is active |
| Barge-in / interruption | 🚧 Pending | User interruption while AI speaks |
| Conversation history | 🚧 Pending | Current LLM request is stateless |
| Telephony / Exotel | 🚧 Planned | Phone-call integration |
| Production orchestration | 🚧 Planned | Session lifecycle and backend integration |

---

## Architecture

```text
                    ┌───────────────────────┐
                    │   User / Microphone   │
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
                    │     Deepgram STT      │
                    │    Streaming Speech   │
                    └───────────┬───────────┘
                                │
                        Completed Utterance
                                │
                                ▼
                    ┌───────────────────────┐
                    │      OpenAI LLM       │
                    │   AI Response Text    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     ElevenLabs TTS    │
                    │    Streaming Speech   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │        LiveKit        │
                    │     voice-output      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   User / TTS Listener │
                    └───────────────────────┘
```

---

## Completed-Utterance LLM Design

The current voice pipeline is designed to minimize unnecessary LLM requests.

### Interim transcript

```text
[PARTIAL] Hello, Agni...
        ↓
Displayed in terminal
        ↓
No OpenAI request
```

### Completed utterance

```text
[FINAL] Hello, Agni AI.
        ↓
[VAD] Utterance ended
        ↓
"Hello, Agni AI."
        ↓
One OpenAI request
```

This prevents:

- duplicate LLM calls
- unnecessary token usage
- responses to unfinished speech
- repeated AI replies from interim STT text

---

## Streaming STT Architecture

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
PCM16 @ 16 kHz mono
    ↓
STTStreamAdapter
    ↓
FastAPI WebSocket
    ↓
Deepgram Streaming STT
    ↓
Partial / Final / VAD events
```

### Current STT endpoint

```text
ws://127.0.0.1:8000/api/v1/stt/stream
```

### STT events

The streaming service can emit:

```text
partial
final
speech_started
utterance_end
silence
incomplete_speech
error
```

---

## LLM Architecture

The LLM layer uses a provider interface so the voice pipeline is not tightly coupled to one implementation.

### LLM provider interface

File:

```text
app/voice/llm_provider.py
```

Core contract:

```python
async def stream_response(
    self,
    user_text: str,
) -> AsyncIterator[str]:
    ...
```

The interface also provides:

```python
async def generate_response(
    self,
    user_text: str,
) -> str:
    ...
```

### OpenAI provider

File:

```text
app/voice/openai_llm_provider.py
```

Responsibilities:

- load OpenAI configuration
- initialize the asynchronous OpenAI client
- accept completed user utterances
- stream response text
- keep voice responses concise
- limit output tokens

Typical configuration:

```env
OPENAI_MODEL=gpt-5.6-luna
OPENAI_MAX_OUTPUT_TOKENS=120
```

---

## TTS Architecture

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
  voice-output
       ↓
    LiveKit
```

### Current audio format

| Property | Value |
|---|---|
| Encoding | PCM 16-bit little-endian |
| Sample rate | 16,000 Hz |
| Channels | Mono |
| ElevenLabs output | `pcm_16000` |
| TTS model | `eleven_flash_v2_5` |

---

## Provider Abstraction

The voice architecture keeps provider-specific logic separate from the main pipeline.

```text
                Voice Pipeline
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
        STT          LLM        TTS
          │          │          │
          ▼          ▼          ▼
      Deepgram    OpenAI    ElevenLabs
```

Current provider files:

```text
app/voice/llm_provider.py
app/voice/openai_llm_provider.py
app/voice/tts_provider.py
app/voice/elevenlabs_tts_provider.py
```

---

## Project Structure

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
│       ├── llm_provider.py
│       ├── openai_llm_provider.py
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
│   ├── llm_poc.py
│   ├── tts_listener.py
│   └── tts_publisher.py
│
└── services/
    └── stt_service/
        ├── __init__.py
        └── main.py
```

> `latency_log.csv` is generated during STT testing and is ignored by Git.

---

## Main Components

### `livekit_poc/audio_subscriber.py`

This is the current integrated voice pipeline.

It:

1. Connects to LiveKit.
2. Waits for the microphone track.
3. Connects to the streaming STT service.
4. Sends microphone audio to Deepgram.
5. Receives STT events.
6. Collects completed utterances.
7. Sends only completed utterances to OpenAI.
8. Sends the LLM response to ElevenLabs.
9. Publishes generated AI speech to LiveKit.
10. Measures response latency.
11. Suppresses microphone audio while the AI is active.

---

### `livekit_poc/audio_publisher.py`

Publishes local microphone audio into the LiveKit room.

---

### `livekit_poc/tts_listener.py`

Subscribes to:

```text
voice-output
```

and plays received AI audio through the local speaker/output device.

---

### `livekit_poc/tts_publisher.py`

Standalone ElevenLabs → LiveKit TTS test.

Example:

```powershell
python -m livekit_poc.tts_publisher "Hello from Agni AI."
```

---

### `livekit_poc/llm_poc.py`

Standalone OpenAI diagnostic.

```powershell
python -m livekit_poc.llm_poc "Reply only with: Agni AI ready."
```

> **Note:** This command makes a real OpenAI API request and consumes API usage.

---

## Development Environment

Current development setup:

| Component | Current Setup |
|---|---|
| Operating system | Windows |
| Environment | Conda |
| Environment name | `agni-ai` |
| Python | 3.11.x |
| STT | Deepgram |
| LLM | OpenAI |
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

## Environment Variables

Local provider configuration is stored in:

```text
.env.local
```

Example:

```env
DEEPGRAM_API_KEY=<your-key>

OPENAI_API_KEY=<your-key>
OPENAI_MODEL=gpt-5.6-luna
OPENAI_MAX_OUTPUT_TOKENS=120

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

> **Important:** Never place real API keys, API secrets, tokens, or passwords in the README.

---

## Running the Integrated Voice POC

Use four terminals.

### Terminal 1 — STT Service

Run:

```powershell
uvicorn services.stt_service.main:app
```

> For voice-pipeline testing, avoid `--reload` because file changes can restart the FastAPI process and interrupt an active WebSocket connection.

---

### Terminal 2 — TTS Listener

Run:

```powershell
python -m livekit_poc.tts_listener
```

Expected:

```text
AGNI AI - TTS LISTENER

Connecting to LiveKit...
Connected to LiveKit.

Waiting for 'voice-output'...
```

---

### Terminal 3 — Integrated Voice Pipeline

Run:

```powershell
python -m livekit_poc.audio_subscriber
```

Expected:

```text
Waiting for microphone track...
Microphone track ready.
Connecting to streaming STT...
Streaming STT connected.

Voice pipeline ready.
Only completed utterances are sent to OpenAI.
```

---

### Terminal 4 — Microphone Publisher

Run:

```powershell
python -m livekit_poc.audio_publisher
```

Expected:

```text
Microphone audio published
Track: microphone

Speak into the microphone...
```

---

## Verified End-to-End Flow

The complete local flow has been successfully verified:

```text
Microphone
    ↓
LiveKit
    ↓
Deepgram STT
    ↓
Final Transcript
    ↓
Utterance End
    ↓
OpenAI LLM
    ↓
AI Response Text
    ↓
ElevenLabs TTS
    ↓
PCM16 Audio
    ↓
LiveKit voice-output
    ↓
TTS Listener
    ↓
Speaker
```

Example verified interaction:

```text
User:
Hello, Agni AI?

Agni AI:
Hello! You’ve reached Agni AI. How can I help you today?
```

---

## Response Latency

The integrated runtime reports:

```text
Final transcript -> utterance end
Pipeline queue delay
LLM first text chunk
LLM complete response
ElevenLabs first audio chunk
Utterance end -> first AI audio
TTS generation/streaming
Utterance end -> playback complete
```

One local POC run measured:

| Stage | Observed Time |
|---|---:|
| Final transcript → utterance end | 1.015 s |
| Pipeline queue delay | 0.001 s |
| LLM first text chunk | 2.402 s |
| LLM complete response | 2.683 s |
| ElevenLabs first audio chunk | 0.556 s |
| Utterance end → first AI audio | 3.241 s |
| TTS generation / streaming | 2.993 s |
| Utterance end → playback complete | 6.657 s |

> These values are from one local POC run and should not be treated as production benchmarks.

---

## Standalone TTS Test

The TTS path can be tested separately:

```text
Text
 ↓
ElevenLabs
 ↓
PCM16
 ↓
LiveKitAudioOutput
 ↓
LiveKit
 ↓
TTS Listener
 ↓
Speaker
```

Example:

```powershell
python -m livekit_poc.tts_publisher "Hello! You've reached Agni AI. How can I help you today?"
```

Standalone playback has been verified as clear.

---

## Echo / Feedback Handling

The integrated POC currently uses a simple suppression mechanism:

```text
AI generating / speaking
        ↓
Microphone frames received
        ↓
Replace frames with silence
        ↓
Continue Deepgram connection
```

This prevents the local speaker output from being immediately treated as new user speech.

> This is a POC mechanism, not full production acoustic echo cancellation.

---

## Known POC Limitations

### Barge-In

True interruption is not implemented yet.

Current behavior:

```text
AI speaking
    ↓
Microphone suppressed
    ↓
User cannot interrupt AI
```

Future behavior should support:

```text
AI speaking
    ↓
User starts speaking
    ↓
Detect interruption
    ↓
Cancel current playback
    ↓
Transcribe user
    ↓
Generate new response
```

### Conversation History

The current OpenAI request contains only the current completed utterance.

No conversation history is sent yet.

### Occasional Integrated Playback Repetition

Occasional word repetition/stuttering was heard during an integrated response.

The same sentence played clearly through the standalone:

```text
ElevenLabs → LiveKit → TTS Listener
```

path.

Therefore, the standalone TTS implementation is verified, while the integrated repetition issue remains for later diagnosis.

### Telephony

The current POC uses:

- local microphone publishing
- local LiveKit room
- local TTS listener

Exotel / SIP integration is still pending.

---

## Security

The following files may contain credentials:

```text
.env
.env.local
```

Generated runtime files such as:

```text
latency_log.csv
```

must also remain untracked.

Current `.gitignore` should include:

```gitignore
.env
.env.local
.venv/
__pycache__/
*.pyc
*.wav
audio_chunks/
latency_log.csv
```

Before committing:

```powershell
git status
git diff --cached --check
```

Never commit:

- API keys
- API secrets
- passwords
- access tokens
- private credentials

---

## Development Principles

### Modularity

Keep STT, LLM, TTS, LiveKit transport, telephony, and backend logic separate.

### Provider Independence

Use provider interfaces instead of tightly coupling application logic to vendor SDKs.

### Completed-Utterance LLM Calls

Never call the LLM from interim STT output.

### Streaming First

Use streaming where it improves real-time behavior without causing duplicate requests.

### Clear Audio Contracts

Keep encoding, sample rate, channel count, and frame format explicit.

### Independent Testing

Test STT, LLM, and TTS separately before debugging the complete pipeline.

### API Usage Awareness

Avoid unnecessary provider tests when local syntax checks or static verification are sufficient.

---

## Backend and Database Integration

The larger Agni AI system is expected to include:

- FastAPI application modules
- PostgreSQL
- SQLAlchemy
- Alembic
- database models
- authentication / application services
- call and session management

These modules should be integrated without replacing already working voice components unless compatibility has been verified.

---

## Next Steps

The next development phase is no longer basic LLM integration.

Current next steps are:

1. Diagnose occasional integrated playback repetition when required.
2. Add true barge-in / interruption handling.
3. Add conversation and session context.
4. Integrate backend and database modules.
5. Integrate Exotel / SIP telephony.
6. Add production error handling and retries.
7. Add call/session tracing and observability.
8. Optimize latency after correctness is stable.

---

## Current Milestone

The repository now contains a verified local end-to-end Agni AI voice POC:

```text
LiveKit
   ↓
Deepgram STT
   ↓
OpenAI LLM
   ↓
ElevenLabs TTS
   ↓
LiveKit
```

The next phase focuses on **robustness**, **barge-in**, **backend integration**, **telephony**, and **production orchestration**.

---

**Agni AI — Voice Architecture & Provider Layer**
