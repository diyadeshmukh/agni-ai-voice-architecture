# Agni AI — Real-Time Voice Agent

> A modular real-time voice AI pipeline built with **LiveKit**, **Deepgram**, **OpenAI**, **ElevenLabs**, and **FastAPI**.

---

## Overview

Agni AI is a real-time voice-agent system designed to replace a human voice agent in a call flow.

The current repository contains:

- streaming Speech-to-Text (STT)
- OpenAI LLM integration
- configurable per-session LLM system prompt
- streaming ElevenLabs Text-to-Speech (TTS)
- LiveKit audio transport
- hard barge-in / interruption support
- multilingual STT routing
- frontend-facing configuration and session APIs
- local development and diagnostic POCs

The production telephony layer is planned around Exotel. Exotel integration is not implemented yet.

### Current local voice flow

```text
Laptop Microphone
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
 Streaming Text
        │
        ▼
 ElevenLabs TTS
        │
        ▼
Streaming PCM Audio
        │
        ▼
     LiveKit
   voice-output
        │
        ▼
 Laptop Speaker
```

Interim STT transcripts are displayed for diagnostics but are **not sent to OpenAI**. Only completed utterances are sent to the LLM.

---

## Current Status

| Component | Status | Description |
|---|---|---|
| LiveKit connection | ✅ Working | Room connection and audio transport |
| LiveKit microphone publishing | ✅ Working | Publishes local microphone audio |
| LiveKit audio subscription | ✅ Working | Receives remote audio tracks |
| Streaming STT | ✅ Working | Deepgram realtime transcription |
| STT language routing | ✅ Working | English, Hindi, Hinglish, Marathi routing |
| Interim transcripts | ✅ Working | Diagnostic only; never sent to OpenAI |
| Completed utterance dispatch | ✅ Working | `speech_final` fast path with `utterance_end` fallback |
| OpenAI LLM | ✅ Working | Streaming response generation |
| System prompt configuration | ✅ Working | Optional per-session LLM instructions |
| ElevenLabs TTS | ✅ Working | Streaming Text-to-Dialogue WebSocket |
| OpenAI → ElevenLabs overlap | ✅ Working | TTS can begin before LLM completion |
| LiveKit AI audio output | ✅ Working | Publishes `voice-output` track |
| Local speaker playback | ✅ Working | `audio_publisher.py` plays `voice-output` |
| Hard barge-in | ✅ Core implemented | Cancels active LLM/TTS and clears queued AI audio |
| Local first-attempt barge-in | ⚠️ POC limitation | Detection can be inconsistent with laptop speaker/microphone acoustics |
| Agent API | ✅ Working | Language catalog, health, and session lifecycle APIs |
| Conversation history | 🚧 Pending | Current LLM request is stateless |
| Exotel telephony | 🚧 Planned | Production phone-call transport not implemented yet |
| Production orchestration | 🚧 Planned | Persistent sessions/workers/storage can be added later |

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
                    │  Flux / Nova routing  │
                    └───────────┬───────────┘
                                │
                        Completed Utterance
                                │
                                ▼
                    ┌───────────────────────┐
                    │      OpenAI LLM       │
                    │   Streaming Response  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     ElevenLabs TTS    │
                    │    Streaming PCM16    │
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
                    │    User / Speaker     │
                    └───────────────────────┘
```

---

## STT Language Routing

The user-facing language modes are:

| Mode | Deepgram backend | Model / configuration |
|---|---|---|
| `english` | Flux | `flux-general-en` |
| `hindi` | Flux | `flux-general-multi` + `language_hint=["hi"]` |
| `hinglish` | Flux | `flux-general-multi` + `language_hint=["en", "hi"]` |
| `marathi` | Nova-3 | `nova-3` + `language="mr"` |
| `multi` | Nova-3 | Legacy/debug multilingual mode |

Aliases such as `en`, `en-IN`, `hi`, and `mr` are normalized by the STT API.

### Streaming STT endpoint

```text
ws://127.0.0.1:8000/api/v1/stt/stream?language=english
```

Supported normalized STT events include:

```text
partial
final
speech_started
utterance_end
silence
incomplete_speech
error
```

For Flux, `StartOfTurn` is normalized to `speech_started`. `StartOfTurn` may contain an empty transcript, so Agni emits `speech_started` immediately instead of waiting for transcript text.

---

## Completed-Utterance LLM Design

The voice pipeline minimizes unnecessary LLM requests.

```text
[PARTIAL] Explain AI...
        ↓
Displayed for diagnostics
        ↓
No OpenAI request

[FINAL] Explain AI in detail. [SPEECH FINAL]
        ↓
Completed utterance dispatched immediately
        ↓
One OpenAI request
```

`speech_final=True` is the fast path. `utterance_end` remains as a fallback so the same user turn is not sent twice.

This prevents duplicate LLM calls, unnecessary token usage, responses to unfinished speech, and repeated AI replies from interim transcript text.

---

## Barge-In / Interruption

The microphone remains connected to STT while Agni is speaking.

For Flux languages, semantic `StartOfTurn` is used as the primary hard-interrupt signal:

```text
Agni speaking
    ↓
User begins speaking
    ↓
Flux StartOfTurn
    ↓
speech_started
    ↓
interrupt_event set
    ↓
Current LLM/TTS response cancelled
    ↓
LiveKit queued AI audio cleared
    ↓
User's completed utterance is processed
    ↓
Agni starts the new response
```

For Nova-3 / Marathi, raw VAD `speech_started` does not immediately interrupt. A non-empty partial or final transcript confirms the interruption.

`LiveKitAudioOutput` rejects remaining chunks from an interrupted response and clears queued audio before the next response begins.

### Local POC limitation

The core interruption path is implemented and runtime-tested. With a laptop speaker and laptop microphone, the **first** interruption attempt can still be missed because of the local acoustic/audio-processing path. Repeated speech can trigger the interruption correctly.

This local issue should not be treated as proof that the production telephony path will behave the same way. The production Exotel media path will be a separate audio transport and must be tested independently once integrated.

---

## LLM Architecture

The LLM layer uses a provider interface so the main pipeline is not tightly coupled to one implementation.

Provider interface:

```text
app/voice/llm_provider.py
```

OpenAI implementation:

```text
app/voice/openai_llm_provider.py
```

Core streaming contract:

```python
async def stream_response(
    self,
    user_text: str,
) -> AsyncIterator[str]:
    ...
```

Typical configuration:

```env
OPENAI_MODEL=gpt-5.6-luna
OPENAI_MAX_OUTPUT_TOKENS=120
```

OpenAI text is streamed toward ElevenLabs rather than waiting for the entire LLM response to finish first.

### Per-session system prompt

The Agent API supports an optional `system_prompt` when creating a voice session.

The value is passed into the dedicated session subprocess through:

```text
AGNI_SESSION_SYSTEM_PROMPT
```

The integrated voice pipeline then initializes the OpenAI provider with that prompt as its instructions.

If no session-specific prompt is provided, the existing default LLM behavior is used.

---

## TTS Architecture

Current TTS path:

```text
OpenAI text stream
       ↓
   TTSProvider
       ↓
ElevenLabsTTSProvider
       ↓
Text-to-Dialogue WebSocket
       ↓
 Streaming PCM16
       ↓
   TTSAudioChunk
       ↓
LiveKitAudioOutput
       ↓
 LiveKit AudioSource
       ↓
  voice-output
       ↓
    LiveKit
```

### Current audio contract

| Property | Value |
|---|---|
| Encoding | PCM 16-bit little-endian |
| Sample rate | 16,000 Hz |
| Channels | Mono |
| ElevenLabs output | `pcm_16000` |
| TTS model | `eleven_v3_conversational` |

The current provider uses the ElevenLabs WebSocket directly through the `websockets` package.

---

## Frontend Agent API

The frontend-facing Agent API exposes configuration discovery, session lifecycle, and service health endpoints.

Run it with:

```powershell
uvicorn services.agent_service.main:app --port 8001
```

Swagger/OpenAPI is available at:

```text
http://127.0.0.1:8001/docs
```

### Configuration catalog

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/catalog/languages` | List supported agent languages |

Current supported languages:

```text
english
hindi
hinglish
marathi
```

Example response:

```json
{
  "languages": [
    {
      "id": "english",
      "label": "English"
    },
    {
      "id": "hindi",
      "label": "Hindi"
    },
    {
      "id": "hinglish",
      "label": "Hinglish"
    },
    {
      "id": "marathi",
      "label": "Marathi"
    }
  ]
}
```

### Session runtime

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/sessions` | Start an Agni voice session |
| `GET` | `/api/v1/sessions/{session_id}` | Read session status |
| `DELETE` | `/api/v1/sessions/{session_id}` | Stop the session |

Example create request:

```json
{
  "language": "english",
  "system_prompt": "You are a helpful customer support assistant."
}
```

`system_prompt` is optional.

The selected language is validated by the Agent API before a session is created.

If an unsupported language is supplied, request validation fails before the voice-agent subprocess is started.

When a session is created, the Agent API:

1. generates a unique session ID
2. generates a unique LiveKit room
3. creates unique frontend and agent participant identities
4. creates a short-lived frontend LiveKit token
5. starts a dedicated `livekit_poc.audio_subscriber` subprocess
6. passes per-session configuration to that subprocess

Per-session configuration currently includes:

```text
AGNI_SESSION_ROOM_NAME
AGNI_SESSION_PARTICIPANT_IDENTITY
AGNI_SESSION_LANGUAGE
AGNI_SESSION_SYSTEM_PROMPT
```

A successful create-session response includes:

- `session_id`
- current session `status`
- selected `language`
- LiveKit URL
- generated room name
- short-lived frontend participant token
- expected microphone track name: `microphone`
- expected agent audio track name: `voice-output`
- session creation time

Example response shape:

```json
{
  "session_id": "876ff8bdca15475b836288049ab9a848",
  "status": "active",
  "language": "english",
  "livekit": {
    "url": "wss://...",
    "room_name": "agni-session-876ff8bdca15",
    "token": "..."
  },
  "tracks": {
    "microphone": "microphone",
    "agent_audio": "voice-output"
  },
  "created_at": "..."
}
```

### Health

```text
GET /api/v1/health
```

Example response:

```json
{
  "status": "ok",
  "service": "Agni Agent API",
  "livekit_configured": true
}
```

`AgentSessionManager` currently keeps active session state in memory and launches one `audio_subscriber` subprocess per voice session.

Persistent or distributed production session orchestration is not implemented yet.

---

## Project Structure

```text
agni-ai/
│
├── README.md
├── .gitignore
├── mic_test.py
├── requirements.txt
│
├── app/
│   ├── __init__.py
│   ├── database.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── catalog.py
│   │       ├── health.py
│   │       ├── sessions.py
│   │       └── stt.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent_session_manager.py
│   │   └── stt_service.py
│   │
│   └── voice/
│       ├── __init__.py
│       ├── elevenlabs_tts_provider.py
│       ├── livekit_audio_output.py
│       ├── llm_provider.py
│       ├── openai_llm_provider.py
│       ├── stt_stream_adapter.py
│       └── tts_provider.py
│
├── docs/
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
    ├── agent_service/
    │   ├── __init__.py
    │   └── main.py
    │
    └── stt_service/
        ├── __init__.py
        └── main.py
```

`audio_buffer.py` and the old HTTP `stt_adapter.py` have been removed because the active voice pipeline uses continuous streaming through `stt_stream_adapter.py`.

The old combined `app/api/v1/agent.py` API module has also been removed. Frontend-facing API responsibilities are now separated into:

```text
catalog.py
health.py
sessions.py
```

`latency_log.csv` is generated during STT testing and should remain ignored by Git.

---

## Main Components

### `livekit_poc/audio_subscriber.py`

This is the current integrated Agni voice-agent pipeline.

It connects to LiveKit, receives microphone audio, streams it to STT, processes completed utterances, streams OpenAI output into ElevenLabs, publishes `voice-output`, and handles hard interruption.

For frontend-created sessions, it receives session-specific configuration through `AGNI_SESSION_*` environment variables.

### `livekit_poc/audio_publisher.py`

This is a **local development harness**, not the planned production telephony client.

It currently uses the laptop microphone for input and laptop speakers for output.

The local full-duplex setup enables AEC and keeps noise suppression and automatic gain control disabled.

The publisher also subscribes to `voice-output`, so a separate `tts_listener` is not required for the integrated local POC.

When `AGNI_SESSION_ROOM_NAME` is available, the publisher can use the session-specific room name. Otherwise it falls back to the normal local `LIVEKIT_ROOM_NAME` configuration.

### `livekit_poc/tts_listener.py`

Standalone diagnostic listener for `voice-output`.

Keep it for isolated TTS / LiveKit testing; it is not part of the normal integrated three-terminal local run.

### `livekit_poc/tts_publisher.py`

Standalone ElevenLabs → LiveKit TTS diagnostic.

```powershell
python -m livekit_poc.tts_publisher "Hello from Agni AI."
```

### `livekit_poc/llm_poc.py`

Standalone OpenAI diagnostic.

```powershell
python -m livekit_poc.llm_poc "Reply only with: Agni AI ready."
```

This command makes a real OpenAI API request and consumes provider usage.

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

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Environment Variables

Local credentials and provider configuration are stored in `.env.local`.

Example names only — never put real credentials in this README:

```env
DEEPGRAM_API_KEY=<your-key>

OPENAI_API_KEY=<your-key>
OPENAI_MODEL=gpt-5.6-luna
OPENAI_MAX_OUTPUT_TOKENS=120

ELEVENLABS_API_KEY=<your-key>
ELEVENLABS_VOICE_ID=<your-voice-id>
ELEVENLABS_MODEL_ID=eleven_v3_conversational
ELEVENLABS_OUTPUT_FORMAT=pcm_16000

LIVEKIT_URL=<your-livekit-url>
LIVEKIT_API_KEY=<your-livekit-api-key>
LIVEKIT_API_SECRET=<your-livekit-api-secret>
LIVEKIT_ROOM_NAME=agni-ai-voice-poc

STT_STREAM_ENDPOINT=ws://127.0.0.1:8000/api/v1/stt/stream
AGNI_STT_LANGUAGE=english

AGNI_FRONTEND_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173
```

The Agent API passes per-session room, participant, language, and system-prompt configuration to the spawned agent subprocess using `AGNI_SESSION_*` environment variables.

Current internal session variables include:

```text
AGNI_SESSION_ROOM_NAME
AGNI_SESSION_PARTICIPANT_IDENTITY
AGNI_SESSION_LANGUAGE
AGNI_SESSION_SYSTEM_PROMPT
```

These are internal session values and normally do not need to be set manually.

---

## Running the Integrated Local Voice POC

The normal local integrated test uses **three terminals**.

### Terminal 1 — STT service

```powershell
uvicorn services.stt_service.main:app --port 8000
```

For local voice testing, avoid `--reload` because a reload can interrupt the active streaming WebSocket.

### Terminal 2 — Integrated voice pipeline

Choose a language, for example English:

```powershell
$env:AGNI_STT_LANGUAGE="english"

python -m livekit_poc.audio_subscriber
```

Other supported values:

```text
hindi
hinglish
marathi
```

### Terminal 3 — Local microphone + speaker client

```powershell
python -m livekit_poc.audio_publisher
```

Expected startup includes the selected laptop microphone and speaker, LiveKit connection, microphone publishing, and local `voice-output` playback.

---

## Running the Frontend Agent API

The STT service must be available on port `8000` when a real voice session is created.

### Terminal 1 — STT service

```powershell
uvicorn services.stt_service.main:app --port 8000
```

### Terminal 2 — Agent API

```powershell
uvicorn services.agent_service.main:app --port 8001
```

Swagger/OpenAPI is available at:

```text
http://127.0.0.1:8001/docs
```

Current frontend-facing endpoints:

```text
GET    /api/v1/catalog/languages

POST   /api/v1/sessions
GET    /api/v1/sessions/{session_id}
DELETE /api/v1/sessions/{session_id}

GET    /api/v1/health
```

When the frontend calls `POST /api/v1/sessions`, the Agent API creates a unique LiveKit room/token and starts a dedicated Agni `audio_subscriber` subprocess for that session.

The frontend is responsible for:

```text
connecting to the returned LiveKit room
        ↓
publishing microphone track "microphone"
        ↓
receiving/subscribing to "voice-output"
        ↓
playing Agni's generated audio
```

The Agent API never exposes backend provider secrets such as:

```text
OPENAI_API_KEY
DEEPGRAM_API_KEY
ELEVENLABS_API_KEY
LIVEKIT_API_SECRET
```

Only the temporary frontend LiveKit participant token is returned for room access.

---

## Standalone Diagnostics

### STT from microphone

```powershell
python -m app.services.stt_service --source mic --language english
```

### STT from WAV file

```powershell
python -m app.services.stt_service --source file --path <file.wav> --language english
```

### LLM

```powershell
python -m livekit_poc.llm_poc "Reply only with: Agni AI ready."
```

### Standalone TTS through LiveKit

Run the listener:

```powershell
python -m livekit_poc.tts_listener
```

Then run the publisher:

```powershell
python -m livekit_poc.tts_publisher "Hello from Agni AI."
```

---

## Response Latency Design

Two important latency optimizations are already part of the pipeline:

```text
Deepgram completed speech
        ↓
Immediate completed-turn dispatch
        ↓
OpenAI starts streaming
        ↓
First useful text forwarded to ElevenLabs
        ↓
TTS can start before the LLM has finished its full response
```

The runtime logs important stages including STT latency, LLM first text, LLM completion, ElevenLabs first audio, and playback completion.

`latency_log.csv` records STT response timing for diagnostic use. It is not a production benchmark.

---

## Local Echo / Feedback Handling

The local development publisher uses LiveKit `MediaDevices` for both input and output so the audio-processing path can use speaker playback as the AEC reference.

Current local capture configuration:

```text
AEC = enabled
Noise suppression = disabled
High-pass filter = enabled
Automatic gain control = disabled
```

The microphone is **not replaced with silence while Agni speaks**.

Real microphone audio continues flowing to STT so user interruption can be detected.

This is a local development setup. Production Exotel audio handling will be implemented and validated separately.

---

## Planned Exotel Telephony Architecture

The target production call flow is approximately:

```text
Customer phone
      ↓
Exotel number
      ↓
Exotel media / voicebot integration
      ↓
Agni audio bridge
      ↓
STT → LLM → TTS
      ↓
Audio returned to Exotel
      ↓
Customer hears Agni
```

The caller's phone handles its own earpiece, speaker, wired headset, Bluetooth headset, or car audio.

Agni receives the telephony audio stream rather than selecting the caller's physical audio device.

Exotel integration is still pending and should be treated as a separate transport layer from the current local LiveKit development harness.

---

## Known Limitations

### Local first-attempt barge-in

Hard interruption is implemented, but first-attempt speech detection is not perfectly reliable in the laptop speaker/microphone POC.

This remains a local full-duplex test limitation and should be re-evaluated on the production telephony media path.

### Conversation history

The current OpenAI request is stateless.

Conversation/session history is not yet included in LLM context.

### In-memory Agent API sessions

`AgentSessionManager` stores active sessions in memory and launches subprocesses directly.

This is sufficient for the current implementation but is not final production orchestration.

### Telephony

Exotel / PSTN integration is not implemented yet.

### Voice configuration

Selectable male/female voices and accent configuration are not implemented yet.

These should only be exposed through the API after the underlying TTS voice-selection functionality is implemented and tested.

---

## Security

Credentials belong in local environment files such as:

```text
.env
.env.local
```

Never commit API keys, API secrets, passwords, tokens, or private credentials.

Generated/local files such as `latency_log.csv`, Python caches, local audio files, and virtual environments should remain ignored by Git.

Before committing:

```powershell
git status
git diff --check
git diff --cached --check
```

---

## Development Principles

### Modularity

Keep STT, LLM, TTS, LiveKit transport, telephony, and backend orchestration separate.

### Provider independence

Use provider interfaces instead of tightly coupling the pipeline to provider-specific logic.

### Completed-utterance LLM calls

Never call the LLM from interim STT output.

### Streaming first

Use streaming when it reduces latency without creating duplicate responses.

### Explicit audio contracts

Keep encoding, sample rate, channel count, and frame format explicit.

### Independent testing

Test STT, LLM, TTS, LiveKit transport, and telephony paths independently before debugging the complete stack.

### API usage awareness

Prefer syntax checks, static verification, and targeted local tests before paid provider calls.

### API scope

Expose APIs for features that are actually implemented.

Do not add speculative endpoints for future product features until the underlying functionality exists.

---

## Next Steps

1. Finalize the frontend Agent API structure.
2. Add configurable voice selection and accent support.
3. Expose implemented voice/accent options through the configuration catalog.
4. Integrate Exotel telephony/media transport.
5. Validate barge-in on the real phone-call media path.
6. Add conversation/session context.
7. Replace in-memory orchestration when production scaling requires it.
8. Add automated tests for API validation, STT routing, turn dispatch, interruption, and session lifecycle.
