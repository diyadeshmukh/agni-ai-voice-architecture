"""
Agni AI - LLM Provider Interface

Defines the common contract for Large Language Model providers.

The voice pipeline should depend on this interface rather than
depending directly on OpenAI or another specific LLM provider.

Architecture:

    STT Final Transcript
            ↓
       LLMProvider
            ↓
     AI Response Text
            ↓
       TTSProvider
            ↓
         LiveKit
            ↓
         Caller
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMProvider(ABC):
    """
    Common interface for all Agni AI LLM providers.

    Possible implementations:

        OpenAILLMProvider
        OtherLLMProvider
        LocalLLMProvider
    """

    @abstractmethod
    async def stream_response(
        self,
        user_text: str,
    ) -> AsyncIterator[str]:
        """
        Generate an AI response as a stream of text chunks.

        Streaming is preferred for the voice pipeline because
        generated text can later be forwarded to TTS without
        waiting for the complete response.
        """

        raise NotImplementedError(
            "LLM providers must implement stream_response()"
        )

    async def generate_response(
        self,
        user_text: str,
    ) -> str:
        """
        Generate and return the complete AI response.

        This helper collects all streamed text chunks into
        one final response.
        """

        chunks: list[str] = []

        async for chunk in self.stream_response(user_text):
            chunks.append(chunk)

        return "".join(chunks).strip()