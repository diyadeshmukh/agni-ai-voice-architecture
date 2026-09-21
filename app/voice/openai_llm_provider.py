"""
Agni AI - OpenAI LLM Provider

Concrete OpenAI implementation of the Agni AI LLMProvider interface.

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
You are Agni AI, a helpful real-time voice assistant.

Respond naturally and conversationally.

Keep responses concise because they will be converted to speech
and played to a caller in real time.

Avoid unnecessary markdown, bullet points, headings, or special
formatting unless the user explicitly asks for them.

Prefer short, clear spoken responses.
""".strip()


class OpenAILLMProvider(LLMProvider):
    """
    OpenAI implementation of the Agni AI LLM provider interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
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

        self.instructions = (
            instructions
            or DEFAULT_SYSTEM_INSTRUCTIONS
        )

        if max_output_tokens is not None:
            self.max_output_tokens = max_output_tokens
        else:
            self.max_output_tokens = int(
                os.getenv(
                    "OPENAI_MAX_OUTPUT_TOKENS",
                    "300",
                )
            )

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

            if event.type != "response.output_text.delta":
                continue

            delta = event.delta

            if not delta:
                continue

            yield delta