"""
Agni AI - OpenAI LLM POC

Standalone proof-of-concept for testing the LLM provider.

This POC verifies:

    User Text
        ↓
    OpenAILLMProvider
        ↓
    OpenAI Responses API
        ↓
    Streaming AI Response Text

This file does not yet connect STT or TTS.

After this POC works, the next integration will be:

    STT
     ↓
    LLM
     ↓
    TTS
     ↓
    LiveKit
"""

from __future__ import annotations

import asyncio
import sys
import time

from app.voice.openai_llm_provider import (
    OpenAILLMProvider,
)


DEFAULT_PROMPT = (
    "Introduce yourself as Agni AI in two short "
    "sentences suitable for a voice conversation."
)


def get_prompt() -> str:
    """
    Read a prompt from command-line arguments.

    If no command-line prompt is supplied,
    use the default test prompt.
    """

    if len(sys.argv) > 1:

        prompt = " ".join(
            sys.argv[1:]
        ).strip()

        if prompt:
            return prompt

    return DEFAULT_PROMPT


async def main() -> None:

    prompt = get_prompt()

    print()
    print("=" * 72)
    print("AGNI AI - OPENAI LLM POC")
    print("=" * 72)
    print()

    print("Initializing OpenAI LLM provider...")

    llm_provider = OpenAILLMProvider()

    print("OpenAI LLM provider initialized.")
    print()

    print(f"Model: {llm_provider.model}")
    print()

    print("User:")
    print(prompt)
    print()

    print("Agni AI:")

    start_time = time.perf_counter()

    first_chunk_time: float | None = None
    response_chunks: list[str] = []

    async for chunk in llm_provider.stream_response(
        prompt
    ):

        if first_chunk_time is None:
            first_chunk_time = (
                time.perf_counter()
                - start_time
            )

        response_chunks.append(chunk)

        print(
            chunk,
            end="",
            flush=True,
        )

    total_time = (
        time.perf_counter()
        - start_time
    )

    response_text = "".join(
        response_chunks
    ).strip()

    print()
    print()

    print("-" * 72)

    if first_chunk_time is not None:
        print(
            "Time to first text chunk: "
            f"{first_chunk_time:.3f} seconds"
        )

    print(
        "Total response time: "
        f"{total_time:.3f} seconds"
    )

    print(
        "Response characters: "
        f"{len(response_text)}"
    )

    print("-" * 72)
    print()

    if not response_text:
        raise RuntimeError(
            "The LLM returned an empty response."
        )

    print(
        "OpenAI LLM POC completed successfully."
    )


if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print()
        print("LLM POC stopped by user.")

    except Exception as exc:

        print()
        print(f"LLM POC error: {exc}")