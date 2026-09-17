# Agni AI - TTS Provider Interface

## Purpose

The Agni AI voice system uses a provider-independent Text-to-Speech (TTS)
interface.

The application should depend on `TTSProvider` rather than directly
depending on a specific TTS vendor such as ElevenLabs.

This allows the TTS provider to be replaced without changing the
voice-session, LLM, or LiveKit integration code.

---

## Voice Pipeline

The outbound voice flow is:

```text
LLM
  ↓
TTSProvider
  ↓
TTSAudioChunk
  ↓
LiveKitAudioOutput
  ↓
LiveKit Audio Track
  ↓
Caller