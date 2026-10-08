"""
Agni AI - Text Chat Service

Purpose
-------
This file handles the floating text chatbot shown on the frontend.

This chatbot is completely separate from the realtime voice-agent system.

Text chatbot flow:

    User types a message
            ↓
        ChatService
            ↓
          OpenAI
            ↓
      Text response
            ↓
    Frontend chat bubble

This file DOES NOT use:

- LiveKit
- Deepgram STT
- ElevenLabs TTS
- microphone audio
- AgentSessionManager

Voice-agent logic should remain separate.
"""

from __future__ import annotations

import asyncio
import os
import uuid

from datetime import datetime, timezone

from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Default chatbot instructions
# ---------------------------------------------------------------------------
#
# These instructions tell OpenAI how the floating Agni chatbot should behave.
#
# Later, if required, this can be moved into configuration or database settings.
#
# We intentionally keep the chatbot concise because it is displayed inside
# a small floating chat widget on the frontend.
# ---------------------------------------------------------------------------

DEFAULT_CHAT_INSTRUCTIONS = """
You are Agni, the text chatbot for the Agni AI platform.

Agni AI is a voice AI platform for building and running
AI voice agents.

Your job in this chat widget is to:
- answer general questions about Agni AI
- help users understand the platform
- help users who want to book a demo
- help users describe an issue
- answer normal general questions when appropriate

Rules:
- Be friendly, clear, and concise.
- This is a text chatbot, so return text only.
- Reply in the same language style as the user when practical.
- Do not pretend that you completed a booking, ticket, payment,
  or other external action unless the system actually performed it.
- Do not invent pricing, policies, integrations, or unsupported
  product capabilities.
- If you do not know an Agni-specific fact, say so clearly.
""".strip()


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
#
# Raised when frontend sends a conversation_id that does not exist.
#
# Example:
#
# conversation_id = "chat_invalid123"
#
# If that conversation is not present in memory,
# the API can return a clean 404 response.
# ---------------------------------------------------------------------------

class ChatConversationNotFound(Exception):
    """Raised when a requested chat conversation does not exist."""


# ---------------------------------------------------------------------------
# Chat Service
# ---------------------------------------------------------------------------

class ChatService:
    """
    Handles text-only chatbot conversations.

    Current V1 behavior:
    --------------------
    - Creates a conversation ID.
    - Stores conversation messages in memory.
    - Sends conversation history to OpenAI.
    - Stores the assistant response.
    - Returns the complete conversation to the API.

    Important:
    ----------
    Conversation history is currently stored in Python memory.

    This means:
        backend restart
            ↓
        stored chatbot conversations are cleared

    Later, Pooja's database layer can persist these conversations.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:

        # ------------------------------------------------------------------
        # OpenAI API key
        # ------------------------------------------------------------------
        #
        # Normally comes from:
        #
        # OPENAI_API_KEY
        #
        # We also allow passing it directly for testing if needed.
        # ------------------------------------------------------------------

        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
        )

        # ------------------------------------------------------------------
        # Chat model
        # ------------------------------------------------------------------
        #
        # Priority:
        #
        # 1. model passed directly
        # 2. AGNI_CHAT_MODEL
        # 3. OPENAI_MODEL
        # 4. fallback model
        #
        # Keeping AGNI_CHAT_MODEL separate allows us to use a different
        # model for the text chatbot later without changing the voice agent.
        # ------------------------------------------------------------------

        self.model = (
            model
            or os.getenv(
                "AGNI_CHAT_MODEL",
            )
            or os.getenv(
                "OPENAI_MODEL",
                "gpt-5.6-luna",
            )
        )

        # ------------------------------------------------------------------
        # Chatbot system instructions
        # ------------------------------------------------------------------
        #
        # Can be overridden through an environment variable later.
        # ------------------------------------------------------------------

        self.instructions = os.getenv(
            "AGNI_CHAT_SYSTEM_PROMPT",
            DEFAULT_CHAT_INSTRUCTIONS,
        )

        # ------------------------------------------------------------------
        # Maximum size of assistant response
        # ------------------------------------------------------------------
        #
        # Floating chatbot answers should normally be short.
        # ------------------------------------------------------------------

        self.max_output_tokens = int(
            os.getenv(
                "AGNI_CHAT_MAX_OUTPUT_TOKENS",
                "500",
            )
        )

        # OpenAI client is created only when it is actually needed.
        self._client: AsyncOpenAI | None = None

        # ------------------------------------------------------------------
        # In-memory conversation storage
        # ------------------------------------------------------------------
        #
        # Structure:
        #
        # {
        #     "chat_123": [
        #         {
        #             "role": "user",
        #             "content": "Hello",
        #             "created_at": "..."
        #         },
        #         {
        #             "role": "assistant",
        #             "content": "Hi!",
        #             "created_at": "..."
        #         }
        #     ]
        # }
        #
        # Later this can be replaced by database storage.
        # ------------------------------------------------------------------

        self._conversations: dict[
            str,
            list[dict[str, str]],
        ] = {}

        # Lock protects conversation data if multiple requests
        # arrive at nearly the same time.
        self._lock = asyncio.Lock()

    # ----------------------------------------------------------------------
    # Timestamp helper
    # ----------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        """
        Return the current UTC timestamp.

        Example:

        2026-10-08T09:30:00+00:00
        """

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ----------------------------------------------------------------------
    # OpenAI client helper
    # ----------------------------------------------------------------------

    def _get_client(self) -> AsyncOpenAI:
        """
        Create the OpenAI client only when required.

        This prevents OpenAI initialization from happening immediately
        when the FastAPI application starts.
        """

        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing from .env.local"
            )

        # Reuse the same OpenAI client after it has been created.
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key
            )

        return self._client

    # ----------------------------------------------------------------------
    # Send chatbot message
    # ----------------------------------------------------------------------

    async def send_message(
        self,
        *,
        conversation_id: str | None,
        message: str,
    ) -> tuple[
        str,
        list[dict[str, str]],
    ]:
        """
        Send one user message to the Agni text chatbot.

        Parameters
        ----------
        conversation_id:
            Existing conversation ID.

            If None, a new conversation is created.

        message:
            Text typed by the user.

        Returns
        -------
        tuple:
            (
                conversation_id,
                complete conversation messages
            )

        Example first request:

            conversation_id = None
            message = "Tell me about Agni"

        Example next request:

            conversation_id = "chat_abc123"
            message = "What can it do?"
        """

        # Remove unwanted spaces around the message.
        message = message.strip()

        if not message:
            raise ValueError(
                "message must not be empty"
            )

        # ------------------------------------------------------------------
        # Step 1: Create or load conversation
        # ------------------------------------------------------------------

        async with self._lock:

            # If frontend does not provide conversation_id,
            # this is the first message of a new conversation.
            if conversation_id is None:

                conversation_id = (
                    "chat_"
                    + uuid.uuid4().hex
                )

                self._conversations[
                    conversation_id
                ] = []

            # If frontend provides a conversation_id that does not exist,
            # return a controlled error instead of silently creating one.
            elif (
                conversation_id
                not in self._conversations
            ):
                raise ChatConversationNotFound(
                    conversation_id
                )

            # Get the existing conversation messages.
            conversation = self._conversations[
                conversation_id
            ]

            # Save the new user message.
            conversation.append(
                {
                    "role": "user",
                    "content": message,
                    "created_at": self._now(),
                }
            )

            # --------------------------------------------------------------
            # Prepare conversation for OpenAI
            # --------------------------------------------------------------
            #
            # OpenAI only needs:
            #
            # role
            # content
            #
            # created_at is our own frontend/backend metadata,
            # so we do not send it to the model.
            # --------------------------------------------------------------

            model_input = [
                {
                    "role": item["role"],
                    "content": item["content"],
                }
                for item in conversation
            ]

        # ------------------------------------------------------------------
        # Step 2: Call OpenAI
        # ------------------------------------------------------------------

        client = self._get_client()

        response = await client.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=model_input,
            max_output_tokens=self.max_output_tokens,
        )

        # Responses API provides combined text through output_text.
        reply = (
            response.output_text
            or ""
        ).strip()

        # Safety fallback in case OpenAI returns no usable text.
        if not reply:
            reply = (
                "Sorry, I couldn't generate a response. "
                "Please try again."
            )

        # ------------------------------------------------------------------
        # Step 3: Build assistant message
        # ------------------------------------------------------------------

        assistant_message = {
            "role": "assistant",
            "content": reply,
            "created_at": self._now(),
        }

        # ------------------------------------------------------------------
        # Step 4: Save assistant response
        # ------------------------------------------------------------------

        async with self._lock:

            self._conversations[
                conversation_id
            ].append(
                assistant_message
            )

            # Return a copy rather than exposing our internal list directly.
            messages = [
                dict(item)
                for item in self._conversations[
                    conversation_id
                ]
            ]

        # ------------------------------------------------------------------
        # Final result used by the API layer
        # ------------------------------------------------------------------

        return conversation_id, messages