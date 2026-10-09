"""
Jeeva AI - Text Chat API

Purpose
-------
This file exposes the API used by the floating Jeeva text chatbot.

The same endpoint supports:

    POST /api/v1/chat

Features:
- normal text messages
- emoji messages
- continuing an existing conversation
- starting a new conversation
- attachment-only messages
- text + attachment messages

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
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)
from pydantic import BaseModel, Field

# request.form() returns Starlette UploadFile objects,
# so we use the Starlette class for reliable type checking.
from starlette.datastructures import UploadFile

from app.services.chat_service import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS_PER_MESSAGE,
    ChatAttachmentError,
    ChatAttachmentInput,
    ChatConversationNotFound,
    ChatService,
)


# ---------------------------------------------------------------------------
# Attachment response model
# ---------------------------------------------------------------------------

class ChatAttachmentResponse(BaseModel):
    """
    File information returned to the frontend.

    We return only safe metadata.

    The actual Base64 file content used internally
    is NOT returned to frontend.
    """

    name: str

    content_type: str

    size_bytes: int

    # "image" or "document"
    kind: str


# ---------------------------------------------------------------------------
# Individual chat message returned to frontend
# ---------------------------------------------------------------------------

class ChatMessageResponse(BaseModel):
    """
    One message displayed inside the Jeeva chatbot.
    """

    role: str

    content: str

    created_at: datetime

    # Empty list for messages that do not contain files.
    attachments: list[
        ChatAttachmentResponse
    ] = Field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Complete API response
# ---------------------------------------------------------------------------

class ChatResponse(BaseModel):
    """
    Complete conversation response returned to frontend.
    """

    conversation_id: str

    messages: list[
        ChatMessageResponse
    ]


# ---------------------------------------------------------------------------
# Helper: convert service result into API response
# ---------------------------------------------------------------------------

def _build_chat_response(
    conversation_id: str,
    messages: list[dict[str, Any]],
) -> ChatResponse:
    """
    Convert ChatService message dictionaries into
    validated FastAPI/Pydantic response objects.
    """

    return ChatResponse(
        conversation_id=conversation_id,
        messages=[
            ChatMessageResponse(
                role=item["role"],
                content=item["content"],
                created_at=item["created_at"],
                attachments=[
                    ChatAttachmentResponse(
                        **attachment
                    )
                    for attachment
                    in item.get(
                        "attachments",
                        [],
                    )
                ],
            )
            for item in messages
        ],
    )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def create_chat_router(
    chat_service: ChatService,
) -> APIRouter:
    """
    Create the Jeeva chatbot router.

    ChatService is injected from main.py so:

        chat.py
            handles HTTP/API concerns

        chat_service.py
            handles conversation/OpenAI logic
    """

    router = APIRouter(
        prefix="/chat",
        tags=["chat"],
    )

    # ----------------------------------------------------------------------
    # POST /api/v1/chat
    # ----------------------------------------------------------------------

    @router.post(
        "",
        response_model=ChatResponse,

        # Because this endpoint supports BOTH JSON and multipart requests,
        # we document both request formats manually for Swagger.
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {

                    # ------------------------------------------------------
                    # Normal text / emoji / new conversation
                    # ------------------------------------------------------

                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "conversation_id": {
                                    "type": [
                                        "string",
                                        "null",
                                    ],
                                    "example": "chat_abc123",
                                },
                                "message": {
                                    "type": "string",
                                    "example": "Hello Jeeva 👋",
                                },
                                "action": {
                                    "type": "string",
                                    "enum": [
                                        "message",
                                        "new_conversation",
                                    ],
                                    "default": "message",
                                },
                            },
                        },
                    },

                    # ------------------------------------------------------
                    # Attachment request
                    # ------------------------------------------------------

                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "conversation_id": {
                                    "type": "string",
                                },
                                "message": {
                                    "type": "string",
                                },
                                "action": {
                                    "type": "string",
                                    "default": "message",
                                },
                                "attachment": {
                                    "type": "string",
                                    "format": "binary",
                                },
                            },
                        },
                    },
                },
            },
        },
    )
    async def chat(
        request: Request,
    ) -> ChatResponse:
        """
        Handle all Jeeva chatbot operations through one API.

        JSON:
            normal text
            emoji text
            start new conversation

        multipart/form-data:
            attachment only
            text + attachment
        """

        # ---------------------------------------------------------------
        # Default values
        # ---------------------------------------------------------------

        conversation_id: str | None = None

        message = ""

        action = "message"

        attachments: list[
            ChatAttachmentInput
        ] = []

        # Read Content-Type so we know whether frontend
        # sent JSON or multipart/form-data.
        content_type = (
            request.headers.get(
                "content-type",
                "",
            )
            .lower()
        )

        # ===============================================================
        # JSON REQUEST
        # ===============================================================

        if "application/json" in content_type:

            try:

                payload = await request.json()

            except Exception as exc:

                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON request",
                ) from exc

            if not isinstance(
                payload,
                dict,
            ):

                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Request body must be "
                        "a JSON object"
                    ),
                )

            # -----------------------------------------------------------
            # conversation_id
            # -----------------------------------------------------------

            raw_conversation_id = (
                payload.get(
                    "conversation_id"
                )
            )

            if raw_conversation_id is not None:

                conversation_id = str(
                    raw_conversation_id
                ).strip()

                if not conversation_id:
                    conversation_id = None

            # -----------------------------------------------------------
            # message
            # -----------------------------------------------------------

            message = str(
                payload.get(
                    "message",
                    "",
                )
                or ""
            )

            # -----------------------------------------------------------
            # action
            # -----------------------------------------------------------

            action = str(
                payload.get(
                    "action",
                    "message",
                )
                or "message"
            ).strip().lower()

        # ===============================================================
        # MULTIPART REQUEST
        # ===============================================================

        elif (
            "multipart/form-data"
            in content_type
        ):

            try:

                form = await request.form()

            except Exception as exc:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unable to read "
                        "multipart form data"
                    ),
                ) from exc

            # -----------------------------------------------------------
            # conversation_id
            # -----------------------------------------------------------

            raw_conversation_id = (
                form.get(
                    "conversation_id"
                )
            )

            if raw_conversation_id is not None:

                value = str(
                    raw_conversation_id
                ).strip()

                # Browsers/forms sometimes send "null"
                # as literal text.
                if value.lower() not in {
                    "",
                    "null",
                    "none",
                }:
                    conversation_id = value

            # -----------------------------------------------------------
            # message
            # -----------------------------------------------------------

            message = str(
                form.get(
                    "message",
                    "",
                )
                or ""
            )

            # -----------------------------------------------------------
            # action
            # -----------------------------------------------------------

            action = str(
                form.get(
                    "action",
                    "message",
                )
                or "message"
            ).strip().lower()

            # -----------------------------------------------------------
            # Attachments
            # -----------------------------------------------------------
            #
            # Our official frontend field can be:
            #
            # attachment
            #
            # But we also accept common names so Pooja/frontend
            # integration does not require another backend change.
            # -----------------------------------------------------------

            accepted_file_fields = (
                "attachment",
                "attachments",
                "attachments[]",
                "file",
                "files",
                "files[]",
            )

            uploads: list[
                UploadFile
            ] = []

            # Prevent the same UploadFile object from
            # accidentally being added twice.
            seen_uploads: set[int] = set()

            for field_name in accepted_file_fields:

                for item in form.getlist(
                    field_name
                ):

                    if (
                        isinstance(
                            item,
                            UploadFile,
                        )
                        and id(item)
                        not in seen_uploads
                    ):

                        seen_uploads.add(
                            id(item)
                        )

                        uploads.append(
                            item
                        )

            # -----------------------------------------------------------
            # Maximum number of attachments
            # -----------------------------------------------------------

            if (
                len(uploads)
                > MAX_ATTACHMENTS_PER_MESSAGE
            ):

                raise HTTPException(
                    status_code=422,
                    detail=(
                        "A maximum of "
                        f"{MAX_ATTACHMENTS_PER_MESSAGE} "
                        "attachments is allowed "
                        "per message"
                    ),
                )

            # -----------------------------------------------------------
            # Read uploaded files
            # -----------------------------------------------------------

            for upload in uploads:

                # Read only up to:
                #
                # max size + 1 byte
                #
                # If that extra byte exists,
                # we know the file is too large.
                file_data = await upload.read(
                    MAX_ATTACHMENT_BYTES + 1
                )

                filename = (
                    upload.filename
                    or "attachment"
                )

                content_type_value = (
                    upload.content_type
                )

                await upload.close()

                # ------------------------------------------------------
                # File too large
                # ------------------------------------------------------

                if (
                    len(file_data)
                    > MAX_ATTACHMENT_BYTES
                ):

                    max_mb = (
                        MAX_ATTACHMENT_BYTES
                        // (1024 * 1024)
                    )

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"{filename} exceeds "
                            f"the {max_mb} MB "
                            "attachment limit"
                        ),
                    )

                # Convert FastAPI/Starlette file into the
                # service-layer attachment object.
                attachments.append(
                    ChatAttachmentInput(
                        filename=filename,
                        content_type=(
                            content_type_value
                        ),
                        data=file_data,
                    )
                )

        # ===============================================================
        # Unsupported Content-Type
        # ===============================================================

        else:

            raise HTTPException(
                status_code=415,
                detail=(
                    "Use application/json for "
                    "normal chat or "
                    "multipart/form-data "
                    "for attachments"
                ),
            )

        # ===============================================================
        # START NEW CONVERSATION
        # ===============================================================

        if action == "new_conversation":

            (
                new_conversation_id,
                messages,
            ) = await (
                chat_service
                .start_conversation()
            )

            return _build_chat_response(
                new_conversation_id,
                messages,
            )

        # Only two actions are currently valid.
        if action != "message":

            raise HTTPException(
                status_code=422,
                detail=(
                    "action must be either "
                    "'message' or "
                    "'new_conversation'"
                ),
            )

        # ===============================================================
        # MESSAGE VALIDATION
        # ===============================================================

        message = message.strip()

        # Empty text is okay ONLY when a file exists.
        if (
            not message
            and not attachments
        ):

            raise HTTPException(
                status_code=422,
                detail=(
                    "message or attachment "
                    "is required"
                ),
            )

        if len(message) > 4000:

            raise HTTPException(
                status_code=422,
                detail=(
                    "message must not exceed "
                    "4000 characters"
                ),
            )

        # ===============================================================
        # SEND MESSAGE TO SERVICE
        # ===============================================================

        try:

            (
                conversation_id,
                messages,
            ) = await chat_service.send_message(
                conversation_id=conversation_id,
                message=message,
                attachments=attachments,
            )

        # ---------------------------------------------------------------
        # Unknown conversation
        # ---------------------------------------------------------------

        except ChatConversationNotFound as exc:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Chat conversation "
                    "not found"
                ),
            ) from exc

        # ---------------------------------------------------------------
        # Invalid attachment
        # ---------------------------------------------------------------

        except ChatAttachmentError as exc:

            raise HTTPException(
                status_code=415,
                detail=str(exc),
            ) from exc

        # ---------------------------------------------------------------
        # Other invalid input
        # ---------------------------------------------------------------

        except ValueError as exc:

            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        # ---------------------------------------------------------------
        # Configuration/runtime problem
        # ---------------------------------------------------------------

        except RuntimeError as exc:

            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc

        # ===============================================================
        # RESPONSE
        # ===============================================================

        return _build_chat_response(
            conversation_id,
            messages,
        )

    return router