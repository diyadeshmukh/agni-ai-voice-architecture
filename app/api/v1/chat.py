"""
Agni AI - Text Chat API

Purpose
-------
This file exposes the API used by the floating Agni text chatbot.

Frontend flow:

    User types message
            ↓
    POST /api/v1/chat
            ↓
       ChatService
            ↓
          OpenAI
            ↓
      Text response
            ↓
    Frontend chat bubble

Important:
This API is completely separate from the realtime voice-agent session API.

It does NOT use:
- LiveKit
- Deepgram
- ElevenLabs
- microphone/audio
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from app.services.chat_service import (
    ChatConversationNotFound,
    ChatService,
)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
#
# This is what frontend sends to:
#
# POST /api/v1/chat
#
# First message:
#
# {
#     "conversation_id": null,
#     "message": "Tell me about Agni"
# }
#
# Next message:
#
# {
#     "conversation_id": "chat_abc123",
#     "message": "What can it do?"
# }
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):

    # None means this is a brand-new conversation.
    conversation_id: str | None = None

    # Text entered by the user.
    message: str

    # ----------------------------------------------------------------------
    # conversation_id validation
    # ----------------------------------------------------------------------

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(
        cls,
        value: str | None,
    ) -> str | None:

        # New conversation.
        if value is None:
            return None

        # Remove accidental spaces.
        value = value.strip()

        # Treat empty string like no conversation ID.
        if not value:
            return None

        return value

    # ----------------------------------------------------------------------
    # message validation
    # ----------------------------------------------------------------------

    @field_validator("message")
    @classmethod
    def validate_message(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        # Do not allow empty chatbot messages.
        if not value:
            raise ValueError(
                "message must not be empty"
            )

        # Basic protection against extremely large input.
        if len(value) > 4000:
            raise ValueError(
                "message must not exceed 4000 characters"
            )

        return value


# ---------------------------------------------------------------------------
# Individual chat message returned to frontend
# ---------------------------------------------------------------------------
#
# Example:
#
# {
#     "role": "assistant",
#     "content": "Agni is a voice AI platform...",
#     "created_at": "2026-10-08T10:00:00+00:00"
# }
#
# Frontend can use role to decide which side the message appears on:
#
# user      → right side
# assistant → left side
# ---------------------------------------------------------------------------

class ChatMessageResponse(BaseModel):

    role: str

    content: str

    created_at: datetime


# ---------------------------------------------------------------------------
# Complete API response
# ---------------------------------------------------------------------------
#
# We return:
#
# - conversation_id
# - complete message history
#
# This allows frontend to directly render the floating chatbot.
# ---------------------------------------------------------------------------

class ChatResponse(BaseModel):

    conversation_id: str

    messages: list[ChatMessageResponse]


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------
#
# ChatService is passed into this function from main.py.
#
# This keeps:
#
# API responsibility     → chat.py
# chatbot/OpenAI logic   → chat_service.py
#
# separated cleanly.
# ---------------------------------------------------------------------------

def create_chat_router(
    chat_service: ChatService,
) -> APIRouter:

    router = APIRouter(
        prefix="/chat",
        tags=["chat"],
    )

    # ----------------------------------------------------------------------
    # POST /api/v1/chat
    # ----------------------------------------------------------------------
    #
    # This is the ONLY main API required for the floating chatbot right now.
    #
    # It handles both:
    #
    # - starting a conversation
    # - continuing an existing conversation
    #
    # We therefore do not need separate:
    #
    # /chat/start
    # /chat/message
    # /chat/history
    #
    # at this stage.
    # ----------------------------------------------------------------------

    @router.post(
        "",
        response_model=ChatResponse,
    )
    async def chat(
        request: ChatRequest,
    ) -> ChatResponse:

        try:

            # Send user message to the ChatService.
            #
            # ChatService handles:
            #
            # - conversation creation
            # - conversation history
            # - OpenAI call
            # - assistant response
            (
                conversation_id,
                messages,
            ) = await chat_service.send_message(
                conversation_id=request.conversation_id,
                message=request.message,
            )

        # ------------------------------------------------------------------
        # Unknown conversation
        # ------------------------------------------------------------------
        #
        # Example:
        #
        # frontend sends "chat_wrong_id"
        #
        # but backend does not have that conversation.
        # ------------------------------------------------------------------

        except ChatConversationNotFound as exc:

            raise HTTPException(
                status_code=404,
                detail="Chat conversation not found",
            ) from exc

        # ------------------------------------------------------------------
        # Invalid input
        # ------------------------------------------------------------------

        except ValueError as exc:

            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        # ------------------------------------------------------------------
        # Configuration/runtime error
        # ------------------------------------------------------------------
        #
        # Example:
        #
        # OPENAI_API_KEY is missing.
        # ------------------------------------------------------------------

        except RuntimeError as exc:

            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc

        # ------------------------------------------------------------------
        # Convert service data into our API response model
        # ------------------------------------------------------------------

        return ChatResponse(
            conversation_id=conversation_id,
            messages=[
                ChatMessageResponse(
                    role=item["role"],
                    content=item["content"],
                    created_at=item["created_at"],
                )
                for item in messages
            ],
        )

    return router