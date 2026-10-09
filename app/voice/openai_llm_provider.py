"""
Jeeva AI - OpenAI LLM Provider

Concrete OpenAI implementation of the Jeeva AI LLMProvider interface.

Flow:

    User Transcript
         ↓
    OpenAI Responses API
         ↓
    Streaming Text
         ↓
    LLMProvider
         ↓
    TTS Provider
         ↓
    LiveKit
"""

from __future__ import annotations

import os
from typing import AsyncIterator

from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.voice.llm_provider import LLMProvider


load_dotenv(".env.local", override=True)


DEFAULT_SYSTEM_INSTRUCTIONS = """
You are Jeeva AI, a helpful real-time voice assistant.

Respond naturally and conversationally.

Always reply in the same language style as the user's latest message.

Language behavior:
- English input -> reply in English.
- Hindi input -> reply naturally in Hindi using Devanagari.
- Marathi input -> reply naturally in Marathi using Devanagari.
- Hinglish input -> reply naturally in Hinglish, primarily using Latin script.
- If the user mixes languages, mirror that mix naturally.
- Do not translate the user's message unless they ask for translation.
- Keep common technical terms and proper names in English when that sounds natural.

Keep responses concise because they will be converted to speech
and played to a caller in real time.

Prefer short, clear spoken responses.

Avoid unnecessary markdown, headings, bullet points, or special
formatting unless the user explicitly asks for them.
""".strip()


# ---------------------------------------------------------------------------
# Session language routing
# ---------------------------------------------------------------------------

RESPONSE_LANGUAGE_ALIASES = {
    # English
    "en": "english",
    "en-in": "english",
    "english": "english",

    # Hindi
    "hi": "hindi",
    "hindi": "hindi",

    # Hinglish
    "hinglish": "hinglish",

    # Marathi
    "mr": "marathi",
    "marathi": "marathi",

    # Optional automatic/inference mode
    "multi": "multi",
}


SESSION_LANGUAGE_INSTRUCTIONS = {
    "english": (
        "The user explicitly selected English for this voice session. "
        "Always reply in English, even when the latest transcript is "
        "short, ambiguous, contains an Indian place or name, or does "
        "not clearly indicate a language. This session selection "
        "overrides language inference from the transcript."
    ),

    "hindi": (
        "The user explicitly selected Hindi for this voice session. "
        "Always reply naturally in Hindi using Devanagari. "
        "Keep common technical terms and proper names in English "
        "when that sounds natural. This session selection overrides "
        "language inference from the transcript."
    ),

    "hinglish": (
        "The user explicitly selected Hinglish for this voice session. "
        "Always reply naturally in Hinglish, primarily using Latin "
        "script. Mix Hindi and English naturally as people do in "
        "conversation. This session selection overrides language "
        "inference from the transcript."
    ),

    "marathi": (
        "The user explicitly selected Marathi for this voice session. "
        "Always reply naturally in Marathi using Devanagari. "
        "Keep common technical terms and proper names in English "
        "when that sounds natural. This session selection overrides "
        "language inference from the transcript."
    ),
}


class OpenAILLMProvider(LLMProvider):
    """
    OpenAI implementation of the Jeeva AI LLM provider interface.

    response_language is optional.

    If supplied by the integrated voice pipeline, it locks the
    response style to the user's selected session language.

    If omitted, language continues to be inferred from user_text.
    This preserves standalone LLM POC behavior.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        response_language: str | None = None,
    ) -> None:

        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
        )

        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing from .env.local"
            )

        self.model = (
            model
            or os.getenv(
                "OPENAI_MODEL",
                "gpt-5.6-luna",
            )
        )

        # ------------------------------------------------------------------
        # Resolve optional session response language
        # ------------------------------------------------------------------

        self.response_language: str | None = None

        if response_language is not None:

            requested_language = (
                response_language.strip().lower()
            )

            if requested_language:

                resolved_language = (
                    RESPONSE_LANGUAGE_ALIASES.get(
                        requested_language
                    )
                )

                if resolved_language is None:
                    raise RuntimeError(
                        "Unsupported LLM response language: "
                        f"{response_language}. "
                        "Use English, Hindi, Hinglish, Marathi, "
                        "or multi."
                    )

                # "multi" deliberately keeps the original
                # transcript-based language inference behavior.
                if resolved_language != "multi":
                    self.response_language = (
                        resolved_language
                    )

        # ------------------------------------------------------------------
        # Build instructions
        # ------------------------------------------------------------------

        base_instructions = (
            instructions
            or DEFAULT_SYSTEM_INSTRUCTIONS
        )

        if self.response_language is not None:

            session_instruction = (
                SESSION_LANGUAGE_INSTRUCTIONS[
                    self.response_language
                ]
            )

            self.instructions = (
                f"{base_instructions}\n\n"
                "Session language rule:\n"
                f"{session_instruction}"
            )

        else:

            self.instructions = (
                base_instructions
            )

        # ------------------------------------------------------------------
        # Output limit
        # ------------------------------------------------------------------

        if max_output_tokens is not None:

            self.max_output_tokens = (
                max_output_tokens
            )

        else:

            self.max_output_tokens = int(
                os.getenv(
                    "OPENAI_MAX_OUTPUT_TOKENS",
                    "300",
                )
            )

        # ------------------------------------------------------------------
        # OpenAI client
        # ------------------------------------------------------------------

        self.client = AsyncOpenAI(
            api_key=self.api_key,
        )

    async def stream_response(
        self,
        user_text: str,
    ) -> AsyncIterator[str]:
        """
        Stream AI-generated text using the OpenAI Responses API.
        """

        user_text = user_text.strip()

        if not user_text:
            return

        stream = await self.client.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=user_text,
            max_output_tokens=self.max_output_tokens,
            stream=True,
        )

        async for event in stream:

            if (
                event.type
                != "response.output_text.delta"
            ):
                continue

            delta = event.delta

            if not delta:
                continue

            yield delta