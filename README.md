# Jeeva AI — Real-Time Voice Agent

> A modular real-time voice AI pipeline built with **LiveKit**, **Deepgram**, **OpenAI**, **ElevenLabs**, and **FastAPI**.

---

## Overview

Jeeva AI is a real-time voice-agent system for conversational voice interactions.

The current repository includes:

- streaming Speech-to-Text (STT)
- OpenAI LLM integration
- configurable per-session system prompt
- configurable welcome message support
- streaming ElevenLabs Text-to-Speech (TTS)
- LiveKit microphone/audio transport
- hard barge-in / interruption support
- multilingual STT routing
- Agent session lifecycle APIs
- post-call structured data extraction
- local development and diagnostic POCs

The current development voice flow uses LiveKit. Production PSTN/telephony transport such as Exotel is a separate integration layer and is not yet part of the production voice runtime in this repository.

---

## Current Local Voice Flow

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

Interim STT transcripts may be shown for diagnostics, but they are **not sent to OpenAI**. Only completed user utterances are dispatched to the LLM.

---

## Current Status

| Component | Status | Description |
|---|---|---|
| LiveKit connection | ✅ Working | Room connection and audio transport |
| LiveKit microphone publishing | ✅ Working | Publishes local microphone audio |
| LiveKit audio subscription | ✅ Working | Receives remote audio tracks |
| Streaming STT | ✅ Working | Deepgram realtime transcription |
| STT language routing | ✅ Working | English, Hindi, Hinglish, Marathi |
| English end-to-end demo | ✅ Tested | STT → LLM → TTS → LiveKit |
| Hindi end-to-end demo | ✅ Tested | STT → LLM → TTS → LiveKit |
| Hinglish end-to-end demo | ✅ Tested | STT → LLM → TTS → LiveKit |
| Marathi end-to-end demo | ✅ Tested | STT → LLM → TTS → LiveKit |
| Interim transcripts | ✅ Working | Diagnostic only; not sent to OpenAI |
| Completed utterance dispatch | ✅ Working | Completed speech is sent once to the LLM |
| OpenAI LLM | ✅ Working | Streaming response generation |
| Per-session system prompt | ✅ Working | Optional runtime LLM instructions |
| Welcome message | ✅ Working | Session-specific welcome message support |
| ElevenLabs TTS | ✅ Working | Streaming speech generation |
| OpenAI → ElevenLabs overlap | ✅ Working | TTS can begin before the LLM fully finishes |
| LiveKit AI audio output | ✅ Working | Publishes `voice-output` |
| Local speaker playback | ✅ Working | `audio_publisher.py` plays agent audio |
| Hard barge-in | ✅ Implemented | Cancels active response and clears queued AI audio |
| Latest demo barge-in validation | ⚠️ Not re-tested | Core exists, but it was not re-demonstrated in the latest multilingual execution pass |
| Agent Session API | ✅ Working | Create, inspect, and stop runtime sessions |
| Clean local session shutdown | ✅ Tested | Ctrl+C disconnects LiveKit and ends backend session |
| Post-call data extraction | ✅ Working | Structured extraction runs when a session ends |
| Conversation history in voice LLM | 🚧 Pending | Current voice LLM request flow is not yet a persistent conversational memory system |
| Exotel telephony | 🚧 Planned | Production phone-call media transport still needs integration |
| Production orchestration | 🚧 Planned | Current session orchestration is suitable for local/demo execution |

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
                    │      Deepgram STT     │
                    │  Flux / Nova routing  │
                    └───────────┬───────────┘
                                │
                      Completed Utterance
                                │
                                ▼
                    ┌───────────────────────┐
                    │      OpenAI LLM       │
                    │  Streaming Response   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    ElevenLabs TTS     │
                    │   Streaming PCM16     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │        LiveKit        │
                    │      voice-output     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    User / Speaker     │
                    └───────────────────────┘
```

---

## STT Language Routing

The current user-facing demo modes are:

| Mode | Deepgram backend | Model / configuration |
|---|---|---|
| `english` | Flux | `flux-general-en` |
| `hindi` | Flux | multilingual Flux with Hindi routing |
| `hinglish` | Flux | multilingual Flux with English/Hindi routing |
| `marathi` | Nova-3 | `nova-3` with Marathi language routing |

The STT API also normalizes supported aliases where configured.

### Streaming STT endpoint

```text
ws://127.0.0.1:8000/api/v1/stt/stream?language=english
```

Normalized STT events include:

```text
partial
final
speech_started
utterance_end
silence
incomplete_speech
error
```

For Flux-based modes, semantic turn-start events can be normalized to `speech_started`.

---

## Completed-Utterance LLM Design

The pipeline avoids sending interim speech to the LLM.

```text
[PARTIAL] Explain AI...
        ↓
Diagnostic output only
        ↓
No OpenAI request

[FINAL] Explain AI in detail.
        ↓
Completed utterance dispatched
        ↓
One OpenAI request
```

The completed-utterance path prevents:

- duplicate LLM calls
- responses to unfinished speech
- unnecessary token usage
- repeated AI replies caused by interim transcript updates

---

## Barge-In / Interruption

The microphone continues to flow to STT while Jeeva is speaking.

The intended interruption flow is:

```text
Jeeva speaking
    ↓
User begins speaking
    ↓
speech_started / confirmed user speech
    ↓
interrupt event set
    ↓
Current LLM/TTS response cancelled
    ↓
Queued AI audio cleared
    ↓
User's completed utterance is processed
    ↓
Jeeva begins the new response
```

`LiveKitAudioOutput` is designed to reject remaining chunks from an interrupted response and clear queued audio before the next response begins.

### Latest demo note

The barge-in implementation remains present in the codebase, but **barge-in was not re-tested during the latest multilingual execution-demo pass**.

Do not treat the latest language demo results as a fresh validation of interruption behavior.

---

## LLM Architecture

The LLM layer uses a provider abstraction so the main pipeline is not tightly coupled to one implementation.

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
OPENAI_API_KEY=<your-key>
OPENAI_MODEL=<your-model>
OPENAI_MAX_OUTPUT_TOKENS=120
```

OpenAI text is streamed toward ElevenLabs rather than waiting for the entire LLM response to complete.

### Per-session system prompt

The Agent Session API can pass an optional system prompt into the runtime session.

Jeeva is the product-facing name, but existing internal environment-variable names remain under the **`AGNI_*` namespace for compatibility**.

Examples of internal session variables include:

```text
AGNI_SESSION_ROOM_NAME
AGNI_SESSION_PARTICIPANT_IDENTITY
AGNI_SESSION_LANGUAGE
AGNI_SESSION_SYSTEM_PROMPT
```

These are internal runtime values and normally do not need to be set manually by a frontend client.

---

## TTS Architecture

```text
OpenAI text stream
       ↓
   TTSProvider
       ↓
ElevenLabsTTSProvider
       ↓
Streaming audio
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

Provider-specific model and voice IDs are configured through environment variables.

---

## Agent Session API

Run the Agent API with:

```powershell
uvicorn services.agent_service.main:app --port 8001
```

Swagger/OpenAPI:

```text
http://127.0.0.1:8001/docs
```

> **Windows note:** do not use `--reload` for the Agent API during voice-session testing. Reloading can break or duplicate subprocess/session behavior.

### Session endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/sessions` | Start a Jeeva voice session |
| `GET` | `/api/v1/sessions/{session_id}` | Read session status |
| `DELETE` | `/api/v1/sessions/{session_id}` | Stop the session and return post-call extraction output |

A session create request can include runtime configuration such as language, system prompt, welcome message, and post-call extraction settings when used by the corresponding frontend/backend integration.

A successful session creation starts a dedicated voice-agent subprocess and returns the session/LiveKit information needed by the local client or frontend.

---

## Post-Call Data Extraction

Jeeva supports structured extraction from the persisted runtime transcript when a session ends.

Supported extraction field types:

| Type | Purpose |
|---|---|
| `text` | Free-text value |
| `selector` | One value from configured choices |
| `yes_no` | Boolean value |
| `number` | Numeric value |

Design rules:

- extraction runs after the voice session is stopped
- the post-call model is independent from the conversational model
- missing information is returned as `null`
- the extractor must not invent information that is absent from the transcript
- selector values are constrained to configured choices

The session end response can include:

```json
{
  "status": "ended",
  "post_call_result": {},
  "post_call_error": null
}
```

The post-call runtime is implemented in:

```text
app/services/post_call_data_extractor.py
app/services/openai_post_call_data_extractor.py
```

The Agent-side post-call configuration contract is defined separately from runtime extraction so it can be integrated with the main Agent CRUD/DB layer.

---

## Local Demo Execution

The current execution-demo flow uses **three terminals**.

### Terminal 1 — STT service

```powershell
conda activate agni-ai
uvicorn services.stt_service.main:app --port 8000
```

### Terminal 2 — Agent API

```powershell
conda activate agni-ai
uvicorn services.agent_service.main:app --port 8001
```

Do **not** add `--reload` on Windows for this service.

### Terminal 3 — Local microphone + speaker client

English:

```powershell
python -m livekit_poc.audio_publisher --language english
```

Hindi:

```powershell
python -m livekit_poc.audio_publisher --language hindi
```

Hinglish:

```powershell
python -m livekit_poc.audio_publisher --language hinglish
```

Marathi:

```powershell
python -m livekit_poc.audio_publisher --language marathi
```

The latest execution-demo pass successfully completed end-to-end turns in all four language modes.

### What the publisher does

The local publisher:

1. selects the requested demo language
2. creates an Agent API session
3. receives the generated LiveKit session information
4. connects the laptop microphone to the LiveKit room
5. receives the Jeeva `voice-output` track
6. plays generated audio through the local speaker
7. disconnects cleanly on Ctrl+C
8. calls the Agent API to end the backend session

Expected shutdown output includes:

```text
Disconnected from LiveKit
Backend session ended: ended
Application stopped by user.
```

This clean shutdown prevents demo sessions from being left active after the local client exits.

### Optional Agent API URL

The local publisher can use the configured Agent API URL.

Example:

```env
AGENT_API_URL=http://127.0.0.1:8001
```

---

## Development Environment

Current local development setup:

| Component | Current Setup |
|---|---|
| Operating system | Windows |
| Environment | Conda |
| Environment name | `agni-ai` |
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

Local credentials and provider configuration belong in local environment files such as `.env` or `.env.local`.

Never put real credentials in this README.

Example names only:

```env
DEEPGRAM_API_KEY=<your-key>

OPENAI_API_KEY=<your-key>
OPENAI_MODEL=<your-model>
OPENAI_MAX_OUTPUT_TOKENS=120

ELEVENLABS_API_KEY=<your-key>
ELEVENLABS_VOICE_ID=<your-voice-id>
ELEVENLABS_MODEL_ID=<your-model-id>
ELEVENLABS_OUTPUT_FORMAT=pcm_16000

LIVEKIT_URL=<your-livekit-url>
LIVEKIT_API_KEY=<your-livekit-api-key>
LIVEKIT_API_SECRET=<your-livekit-api-secret>
LIVEKIT_ROOM_NAME=<your-room-name>

STT_STREAM_ENDPOINT=ws://127.0.0.1:8000/api/v1/stt/stream
AGENT_API_URL=http://127.0.0.1:8001
```

### Branding and internal configuration names

Use **Jeeva / Jeeva AI** for product-facing text.

Existing internal technical environment variables remain **`AGNI_*`** for compatibility. Do not rename internal configuration keys to `JEEVA_*` unless the corresponding code migration is deliberately performed across the whole runtime.

---

## Project Structure

Key runtime files include:

```text
agni-ai/
│
├── README.md
├── requirements.txt
│
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── agents.py
│   │       ├── catalog.py
│   │       ├── health.py
│   │       ├── sessions.py
│   │       └── stt.py
│   │
│   ├── services/
│   │   ├── agent_session_manager.py
│   │   ├── openai_post_call_data_extractor.py
│   │   ├── post_call_data_extractor.py
│   │   └── stt_service.py
│   │
│   └── voice/
│       ├── elevenlabs_tts_provider.py
│       ├── livekit_audio_output.py
│       ├── llm_provider.py
│       ├── openai_llm_provider.py
│       ├── stt_stream_adapter.py
│       └── tts_provider.py
│
├── livekit_poc/
│   ├── audio_publisher.py
│   ├── audio_subscriber.py
│   ├── connect_room.py
│   ├── llm_poc.py
│   ├── tts_listener.py
│   └── tts_publisher.py
│
└── services/
    ├── agent_service/
    │   └── main.py
    └── stt_service/
        └── main.py
```

`latency_log.csv`, Python caches, local audio files, virtual environments, and credential files should remain ignored by Git.

---

## Main Components

### `livekit_poc/audio_subscriber.py`

This is the integrated Jeeva voice-agent runtime.

It:

- connects to LiveKit
- receives microphone audio
- streams audio to STT
- processes completed utterances
- streams OpenAI output
- sends text into ElevenLabs TTS
- publishes `voice-output`
- handles interruption/barge-in behavior

For API-created sessions, runtime configuration is supplied by the Agent Session Manager through internal environment variables.

### `livekit_poc/audio_publisher.py`

This is the local execution/demo client.

It now supports:

```text
--language english
--language hindi
--language hinglish
--language marathi
```

It uses the laptop microphone for input and the local speaker for Jeeva audio output.

The publisher also performs clean backend-session shutdown when the local process is stopped.

### `livekit_poc/tts_listener.py`

Standalone diagnostic listener for `voice-output`.

Use it for isolated TTS/LiveKit troubleshooting; it is not required in the normal Agent API demo flow.

### `livekit_poc/tts_publisher.py`

Standalone TTS diagnostic:

```powershell
python -m livekit_poc.tts_publisher "Hello from Jeeva AI."
```

### `livekit_poc/llm_poc.py`

Standalone OpenAI diagnostic:

```powershell
python -m livekit_poc.llm_poc "Reply only with: Jeeva AI ready."
```

This command makes a real OpenAI API request and may consume provider usage.

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
python -m livekit_poc.llm_poc "Reply only with: Jeeva AI ready."
```

### Standalone TTS through LiveKit

Run the listener:

```powershell
python -m livekit_poc.tts_listener
```

Then run:

```powershell
python -m livekit_poc.tts_publisher "Hello from Jeeva AI."
```

---

## Response Latency Design

The pipeline uses streaming to reduce perceived response latency:

```text
Deepgram completed speech
        ↓
Immediate completed-turn dispatch
        ↓
OpenAI starts streaming
        ↓
First useful text forwarded to ElevenLabs
        ↓
TTS can start before the LLM finishes
```

Runtime diagnostics may log stages such as:

- STT completion
- LLM first text
- LLM completion
- ElevenLabs first audio
- playback completion

`latency_log.csv` is a local diagnostic artifact, not a production benchmark.

---

## Local Echo / Feedback Handling

The local laptop demo is a full-duplex microphone/speaker environment.

The microphone must remain available to STT while Jeeva is speaking so interruption can be detected.

Local speaker/microphone acoustics can behave differently from a production telephony media stream. Interruption behavior therefore needs separate validation when the final telephony transport is integrated.

---

## Planned Telephony Architecture

A future production phone-call flow is expected to look approximately like:

```text
Customer phone
      ↓
Telephony provider
      ↓
Media / voicebot integration
      ↓
Jeeva audio bridge
      ↓
STT → LLM → TTS
      ↓
Audio returned to telephony provider
      ↓
Customer hears Jeeva
```

The telephony layer is separate from the current local LiveKit execution harness.

---

## Known Limitations

### Latest barge-in validation

Hard interruption exists in the runtime, but it was **not re-tested during the latest multilingual demo execution pass**.

### Conversation history

The current voice LLM flow does not yet provide a full persistent multi-turn memory system.

### Session orchestration

The current Agent Session Manager keeps runtime state locally and launches subprocesses for active sessions.

This is suitable for current development/demo work but is not final distributed production orchestration.

### Telephony

Production Exotel/PSTN media integration is not yet part of the current voice pipeline.

### Agent CRUD integration

Post-call extraction runtime support exists, while the final mapping of Agent configuration into the main Agent CRUD/DB layer must remain aligned with the backend that owns Agent persistence.

---

## Security

Credentials belong in local environment files such as:

```text
.env
.env.local
```

Never commit:

- API keys
- API secrets
- database passwords
- access tokens
- refresh tokens
- LiveKit secrets
- provider credentials

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

Use provider interfaces instead of tightly coupling the pipeline to one provider.

### Completed-utterance LLM calls

Do not call the LLM from interim STT output.

### Streaming first

Use streaming when it reduces latency without creating duplicate responses.

### Explicit audio contracts

Keep encoding, sample rate, channel count, and frame format explicit.

### Independent testing

Test STT, LLM, TTS, LiveKit transport, and telephony paths independently before debugging the complete stack.

### API usage awareness

Prefer syntax checks, static verification, and targeted local tests before paid provider calls.

### API scope

Expose APIs for functionality that actually exists. Avoid speculative endpoints for features that are not implemented.

---

## Latest Demo Verification

The current execution-demo branch has been validated with:

```text
English   ✅
Hindi     ✅
Hinglish  ✅
Marathi   ✅
```

The tested startup pattern is:

```text
Terminal 1 → STT service on port 8000
Terminal 2 → Agent API on port 8001
Terminal 3 → audio_publisher --language <mode>
```

The clean shutdown path has also been verified:

```text
Ctrl+C
  ↓
LiveKit client disconnects
  ↓
DELETE /api/v1/sessions/{session_id}
  ↓
Backend session ends
```

Barge-in should not be marked as re-tested by this latest execution pass.

---

## Next Steps

1. Keep the execution-demo README and commands aligned with the tested branch.
2. Finalize integration between Agent persistence and runtime session configuration.
3. Keep post-call extraction configuration aligned with the Agent CRUD/DB owner.
4. Integrate production telephony/media transport.
5. Re-test barge-in on the final production audio path.
6. Add persistent conversational context where required.
7. Replace local/in-memory session orchestration when production scaling requires it.
8. Add automated coverage for language routing, session lifecycle, interruption, and post-call extraction.
