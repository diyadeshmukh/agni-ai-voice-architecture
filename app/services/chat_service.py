"""
Jeeva AI - Text Chat Service

Purpose
-------
This file handles the floating text chatbot shown on the frontend.

This chatbot is completely separate from the realtime voice-agent system.

Supported chat features:
- normal text messages
- emoji characters
- conversation history
- start new conversation
- document attachments
- image attachments
- text + attachment messages

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
import base64
import mimetypes
import os
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Default chatbot instructions
# ---------------------------------------------------------------------------

DEFAULT_CHAT_INSTRUCTIONS = """
You are Jeeva, the text chatbot for the Jeeva AI platform.

Jeeva AI is a voice AI platform for building and running
AI voice agents.

Your job in this chat widget is to:
- answer general questions about Jeeva AI
- help users understand the Jeeva AI platform
- help users who want to book a demo
- help users describe an issue
- answer normal general questions when appropriate

Rules:
- Be friendly, clear, and concise.
- This is a text chatbot, so return text only.
- Reply in the same language style as the user when practical.
- When the current user message includes an attachment, use the attachment
  contents when answering.
- If attachment content is available to you, do not say that no attachment
  was provided or that you cannot see the attachment.
- If an attachment genuinely cannot be read, say clearly that the file
  could not be read and do not invent its contents.
- Do not pretend that you completed a booking, ticket, payment,
  or other external action unless the system actually performed it.
- Do not invent pricing, policies, integrations, or unsupported
  product capabilities.
- If you do not know a Jeeva-specific fact, say so clearly.
""".strip()


# ---------------------------------------------------------------------------
# Attachment settings
# ---------------------------------------------------------------------------

# Maximum attachment size allowed by our backend.
#
# Default:
# 10 MB
#
# This can later be changed through:
#
# AGNI_CHAT_MAX_ATTACHMENT_BYTES
#
MAX_ATTACHMENT_BYTES = int(
    os.getenv(
        "AGNI_CHAT_MAX_ATTACHMENT_BYTES",
        str(10 * 1024 * 1024),
    )
)


# Maximum number of attachments allowed with one chat message.
MAX_ATTACHMENTS_PER_MESSAGE = int(
    os.getenv(
        "AGNI_CHAT_MAX_ATTACHMENTS",
        "3",
    )
)


# Common document formats accepted by the Responses API.
DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".json",
    ".html",
    ".xml",
    ".doc",
    ".docx",
    ".rtf",
    ".odt",
    ".ppt",
    ".pptx",
    ".csv",
    ".xls",
    ".xlsx",
}


# Image formats that we will allow in the Jeeva chatbot.
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ChatConversationNotFound(Exception):
    """
    Raised when frontend sends a conversation_id
    that does not exist.
    """


class ChatAttachmentError(ValueError):
    """
    Raised when an uploaded chatbot attachment
    is invalid or unsupported.
    """


# ---------------------------------------------------------------------------
# Attachment input object
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ChatAttachmentInput:
    """
    Internal representation of an uploaded file.

    The API layer will read FastAPI UploadFile objects
    and convert them into this simple structure.

    Keeping this object here means ChatService does not
    need to depend directly on FastAPI.
    """

    filename: str
    content_type: str | None
    data: bytes


# ---------------------------------------------------------------------------
# Chat Service
# ---------------------------------------------------------------------------

class ChatService:
    """
    Handles Jeeva chatbot conversations.

    Current behavior:
    -----------------
    - Creates conversation IDs.
    - Stores conversation messages in memory.
    - Supports normal text.
    - Supports emoji characters.
    - Supports images/documents.
    - Sends conversation history to OpenAI.
    - Stores assistant responses.
    - Returns complete conversation history.

    Important:
    ----------
    Conversation history is currently stored in Python memory.

    This means:

        backend restart
            ->
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

        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
        )

        # ------------------------------------------------------------------
        # Chat model
        # ------------------------------------------------------------------

        self.model = (
            model
            or os.getenv("AGNI_CHAT_MODEL")
            or os.getenv(
                "OPENAI_MODEL",
                "gpt-5.6-luna",
            )
        )

        # ------------------------------------------------------------------
        # Chatbot instructions
        # ------------------------------------------------------------------

        self.instructions = os.getenv(
            "AGNI_CHAT_SYSTEM_PROMPT",
            DEFAULT_CHAT_INSTRUCTIONS,
        )

        # ------------------------------------------------------------------
        # Maximum assistant response size
        # ------------------------------------------------------------------

        self.max_output_tokens = int(
            os.getenv(
                "AGNI_CHAT_MAX_OUTPUT_TOKENS",
                "500",
            )
        )

        # OpenAI client is created only when required.
        self._client: AsyncOpenAI | None = None

        # ------------------------------------------------------------------
        # In-memory conversation storage
        # ------------------------------------------------------------------
        #
        # Messages may now contain attachment metadata,
        # so the value type is Any rather than only str.
        #
        # Example:
        #
        # {
        #     "chat_123": [
        #         {
        #             "role": "user",
        #             "content": "Check this PDF",
        #             "created_at": "...",
        #             "attachments": [...]
        #         }
        #     ]
        # }
        # ------------------------------------------------------------------

        self._conversations: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        # Protect shared in-memory conversation data.
        self._lock = asyncio.Lock()

    # ----------------------------------------------------------------------
    # Timestamp helper
    # ----------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        """Return the current UTC timestamp."""

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ----------------------------------------------------------------------
    # OpenAI client helper
    # ----------------------------------------------------------------------

    def _get_client(self) -> AsyncOpenAI:
        """
        Create the OpenAI client only when required.
        """

        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing from .env.local"
            )

        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key
            )

        return self._client

    # ----------------------------------------------------------------------
    # Start new conversation
    # ----------------------------------------------------------------------

    async def start_conversation(
        self,
    ) -> tuple[
        str,
        list[dict[str, Any]],
    ]:
        """
        Explicitly create a new empty chatbot conversation.

        This method will be used by the frontend button:

            Start new conversation

        We are NOT creating a separate API.

        The existing POST /api/v1/chat endpoint will later call
        this method when action="new_conversation".
        """

        conversation_id = (
            "chat_"
            + uuid.uuid4().hex
        )

        async with self._lock:

            self._conversations[
                conversation_id
            ] = []

        return conversation_id, []

    # ----------------------------------------------------------------------
    # Attachment preparation
    # ----------------------------------------------------------------------

    def _prepare_attachment(
        self,
        attachment: ChatAttachmentInput,
    ) -> dict[str, Any]:
        """
        Validate and prepare one attachment.

        Documents become:
            input_file

        Images become:
            input_image

        We also create frontend-safe metadata so the response
        can show which file was attached without returning the
        actual Base64 file data.
        """

        # Remove any path information and keep only filename.
        filename = Path(
            attachment.filename
        ).name

        if not filename:
            raise ChatAttachmentError(
                "Attachment must have a filename."
            )

        if not attachment.data:
            raise ChatAttachmentError(
                f"{filename} is empty."
            )

        size_bytes = len(
            attachment.data
        )

        if size_bytes > MAX_ATTACHMENT_BYTES:

            max_mb = (
                MAX_ATTACHMENT_BYTES
                // (1024 * 1024)
            )

            raise ChatAttachmentError(
                f"{filename} is too large. "
                f"Maximum attachment size is {max_mb} MB."
            )

        extension = Path(
            filename
        ).suffix.lower()

        allowed_extensions = (
            DOCUMENT_EXTENSIONS
            | IMAGE_EXTENSIONS
        )

        if extension not in allowed_extensions:

            raise ChatAttachmentError(
                "Unsupported attachment type: "
                f"{extension or 'unknown'}."
            )

        # Use browser MIME type when available.
        # Otherwise infer it from the filename.
        content_type = (
            attachment.content_type
            or mimetypes.guess_type(
                filename
            )[0]
            or "application/octet-stream"
        )

        # Convert file bytes to Base64 because OpenAI Responses
        # can receive file/image data directly in the request.
        encoded_data = base64.b64encode(
            attachment.data
        ).decode("ascii")

        data_url = (
            f"data:{content_type};base64,"
            f"{encoded_data}"
        )

        # ------------------------------------------------------------------
        # Image
        # ------------------------------------------------------------------

        if extension in IMAGE_EXTENSIONS:

            kind = "image"

            model_attachment = {
                "type": "input_image",
                "image_url": data_url,
                "detail": "auto",
            }

        # ------------------------------------------------------------------
        # Document
        # ------------------------------------------------------------------

        else:

            kind = "document"

            model_attachment = {
                "type": "input_file",
                "filename": filename,
                "file_data": data_url,
            }

        # This metadata is safe to return to frontend.
        public_attachment = {
            "name": filename,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "kind": kind,
        }

        return {
            "public": public_attachment,

            # Internal model representation.
            # This will NOT be returned to frontend.
            "model": model_attachment,
        }

    # ----------------------------------------------------------------------
    # Build OpenAI conversation input
    # ----------------------------------------------------------------------

    def _build_model_input(
        self,
        conversation: list[
            dict[str, Any]
        ],
    ) -> list[dict[str, Any]]:
        """
        Convert stored conversation history into the format
        required by OpenAI Responses API.
        """

        model_input: list[
            dict[str, Any]
        ] = []

        for item in conversation:

            role = item["role"]

            # Assistant messages contain text only.
            if role == "assistant":

                model_input.append(
                    {
                        "role": "assistant",
                        "content": item["content"],
                    }
                )

                continue

            # User messages can contain text + attachments.
            content: list[
                dict[str, Any]
            ] = []

            # Add model-ready attachments.
            for attachment in item.get(
                "_model_attachments",
                [],
            ):

                content.append(
                    attachment
                )

            text = (
                item.get(
                    "content",
                    "",
                )
                or ""
            ).strip()

            # Add normal user text.
            if text:

                content.append(
                    {
                        "type": "input_text",
                        "text": text,
                    }
                )

            # If user sends only a file with no text,
            # give the model a simple instruction.
            elif item.get(
                "_model_attachments"
            ):

                content.append(
                    {
                        "type": "input_text",
                        "text": (
                            "Please review the attached file "
                            "and respond helpfully."
                        ),
                    }
                )

            model_input.append(
                {
                    "role": "user",
                    "content": content,
                }
            )

        return model_input

    # ----------------------------------------------------------------------
    # Public conversation copy
    # ----------------------------------------------------------------------

    @staticmethod
    def _public_messages(
        conversation: list[
            dict[str, Any]
        ],
    ) -> list[dict[str, Any]]:
        """
        Return messages that are safe for the frontend.

        The private _model_attachments field contains Base64 data,
        so it must never be returned to the frontend.
        """

        public_messages: list[
            dict[str, Any]
        ] = []

        for item in conversation:

            public_messages.append(
                {
                    "role": item["role"],
                    "content": item["content"],
                    "created_at": item[
                        "created_at"
                    ],
                    "attachments": item.get(
                        "attachments",
                        [],
                    ),
                }
            )

        return public_messages

    # ----------------------------------------------------------------------
    # Send chatbot message
    # ----------------------------------------------------------------------

    async def send_message(
        self,
        *,
        conversation_id: str | None,
        message: str,
        attachments: list[
            ChatAttachmentInput
        ] | None = None,
    ) -> tuple[
        str,
        list[dict[str, Any]],
    ]:
        """
        Send one message to the Jeeva chatbot.

        Supported combinations:
        -----------------------
        1. Text only

        2. Emoji text
           Example:
               Hello 👋😊

        3. Attachment only

        4. Text + attachment

        If conversation_id is None,
        a new conversation is automatically created.
        """

        message = (
            message
            or ""
        ).strip()

        attachments = (
            attachments
            or []
        )

        # A completely empty request is invalid.
        if (
            not message
            and not attachments
        ):

            raise ValueError(
                "message or attachment is required"
            )

        if len(message) > 4000:

            raise ValueError(
                "message must contain at most "
                "4000 characters"
            )

        if (
            len(attachments)
            > MAX_ATTACHMENTS_PER_MESSAGE
        ):

            raise ChatAttachmentError(
                "A maximum of "
                f"{MAX_ATTACHMENTS_PER_MESSAGE} "
                "attachments is allowed per message."
            )

        # Prepare and validate attachments before
        # calling OpenAI.
        prepared_attachments = [
            self._prepare_attachment(
                attachment
            )
            for attachment
            in attachments
        ]

        # Build the new user message.
        user_message: dict[str, Any] = {
            "role": "user",
            "content": message,
            "created_at": self._now(),

            # Frontend-visible attachment metadata.
            "attachments": [
                item["public"]
                for item
                in prepared_attachments
            ],

            # Internal OpenAI attachment data.
            "_model_attachments": [
                item["model"]
                for item
                in prepared_attachments
            ],
        }

        # ------------------------------------------------------------------
        # Step 1: Create/load conversation
        # ------------------------------------------------------------------

        async with self._lock:

            if conversation_id is None:

                conversation_id = (
                    "chat_"
                    + uuid.uuid4().hex
                )

                self._conversations[
                    conversation_id
                ] = []

            elif (
                conversation_id
                not in self._conversations
            ):

                raise ChatConversationNotFound(
                    conversation_id
                )

            # Make a temporary copy.
            #
            # We do NOT permanently save the user message yet.
            # If OpenAI fails, we do not want a half-finished message
            # stored in the conversation.
            conversation_for_model = [
                dict(item)
                for item
                in self._conversations[
                    conversation_id
                ]
            ]

            conversation_for_model.append(
                user_message
            )

        # ------------------------------------------------------------------
        # Step 2: Build model input
        # ------------------------------------------------------------------

        model_input = (
            self._build_model_input(
                conversation_for_model
            )
        )

        # ------------------------------------------------------------------
        # Step 3: Call OpenAI
        # ------------------------------------------------------------------

        client = self._get_client()

        response = await client.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=model_input,
            max_output_tokens=self.max_output_tokens,
        )

        reply = (
            response.output_text
            or ""
        ).strip()

        if not reply:

            reply = (
                "Sorry, I couldn't generate a response. "
                "Please try again."
            )

        # ------------------------------------------------------------------
        # Step 4: Build assistant response
        # ------------------------------------------------------------------

        assistant_message: dict[
            str,
            Any,
        ] = {
            "role": "assistant",
            "content": reply,
            "created_at": self._now(),
            "attachments": [],
        }

        # ------------------------------------------------------------------
        # Step 5: Save successful conversation turn
        # ------------------------------------------------------------------

        async with self._lock:

            self._conversations[
                conversation_id
            ].append(
                user_message
            )

            self._conversations[
                conversation_id
            ].append(
                assistant_message
            )

            messages = (
                self._public_messages(
                    self._conversations[
                        conversation_id
                    ]
                )
            )

        # ------------------------------------------------------------------
        # Final result used by the API layer
        # ------------------------------------------------------------------

        return (
            conversation_id,
            messages,
        )