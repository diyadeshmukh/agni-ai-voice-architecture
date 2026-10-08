"""
Agni AI - Voice Session API

Frontend-facing endpoints for creating, inspecting,
and ending Agni AI realtime voice sessions.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from pydantic import (
    BaseModel,
    field_validator,
)

from app.services.agent_session_manager import (
    AgentSession,
    AgentSessionManager,
    SUPPORTED_AGENT_LANGUAGES,
)


class CreateSessionRequest(BaseModel):
    language: str = "english"
    voice_id: str | None = None
    system_prompt: str | None = None
    welcome_message: str | None = None

    @field_validator("language")
    @classmethod
    def validate_language(
        cls,
        value: str,
    ) -> str:

        normalized = (
            value
            .strip()
            .lower()
        )

        if (
            normalized
            not in SUPPORTED_AGENT_LANGUAGES
        ):
            raise ValueError(
                "language must be one of: "
                "english, hindi, "
                "hinglish, marathi"
            )

        return normalized

    @field_validator("voice_id")
    @classmethod
    def validate_voice_id(
        cls,
        value: str | None,
    ) -> str | None:

        if value is None:
            return None

        normalized = (
            value
            .strip()
            .lower()
        )

        if not normalized:
            raise ValueError(
                "voice_id must not be empty"
            )

        return normalized


class LiveKitConnectionResponse(BaseModel):
    url: str
    room_name: str
    token: str


class TrackContractResponse(BaseModel):
    microphone: str
    agent_audio: str


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str
    language: str
    voice_id: str | None

    livekit: LiveKitConnectionResponse
    tracks: TrackContractResponse

    created_at: datetime

class TranscriptMessageResponse(
    BaseModel
):
    role: str
    content: str
    created_at: datetime
    interrupted: bool = False


class RealtimeStateResponse(
    BaseModel
):
    state: str

    last_event: str | None = None

    updated_at: datetime | None = None

class SessionResponse(BaseModel):
    session_id: str
    status: str
    language: str
    voice_id: str | None

    room_name: str

    frontend_identity: str
    agent_identity: str

    transcript: list[
        TranscriptMessageResponse
    ]

    realtime: RealtimeStateResponse

    created_at: datetime
    ended_at: datetime | None


class EndSessionResponse(BaseModel):
    session_id: str
    status: str


def _session_response(
    session: AgentSession,
) -> SessionResponse:

    session.refresh_status()

    runtime = (
        session.runtime_snapshot()
    )

    return SessionResponse(
        session_id=session.session_id,

        status=session.status,

        language=session.language,

        voice_id=session.voice_id,

        room_name=session.room_name,

        frontend_identity=(
            session.frontend_identity
        ),

        agent_identity=(
            session.agent_identity
        ),

        transcript=[
            TranscriptMessageResponse(
                role=message["role"],
                content=message["content"],
                created_at=(
                    message["created_at"]
                ),
                interrupted=message.get(
                    "interrupted",
                    False,
                ),
            )
            for message
            in runtime["transcript"]
            if (
                isinstance(
                    message,
                    dict,
                )
                and message.get(
                    "role"
                )
                and message.get(
                    "content"
                )
                and message.get(
                    "created_at"
                )
            )
        ],

        realtime=RealtimeStateResponse(
            state=(
                runtime[
                    "realtime"
                ]["state"]
            ),

            last_event=(
                runtime[
                    "realtime"
                ]["last_event"]
            ),

            updated_at=(
                runtime[
                    "realtime"
                ]["updated_at"]
            ),
        ),

        created_at=session.created_at,

        ended_at=session.ended_at,
    )


def create_sessions_router(
    session_manager: AgentSessionManager,
) -> APIRouter:

    router = APIRouter(
        prefix="/sessions",
        tags=["sessions"],
    )

    @router.post(
        "",
        response_model=CreateSessionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_session(
        request: CreateSessionRequest,
    ):

        try:
            (
                session,
                frontend_token,
            ) = await session_manager.create_session(
                language=request.language,
                system_prompt=request.system_prompt,
                voice_id=request.voice_id,
                welcome_message=request.welcome_message,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc

        return CreateSessionResponse(
            session_id=session.session_id,
            status=session.status,
            language=session.language,
            voice_id=session.voice_id,
            livekit=LiveKitConnectionResponse(
                url=session_manager.livekit_url,
                room_name=session.room_name,
                token=frontend_token,
            ),
            tracks=TrackContractResponse(
                microphone="microphone",
                agent_audio="voice-output",
            ),
            created_at=session.created_at,
        )

    @router.get(
        "/{session_id}",
        response_model=SessionResponse,
    )
    async def get_session(
        session_id: str,
    ):

        session = session_manager.get_session(
            session_id
        )

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Agent session not found",
            )

        return _session_response(
            session
        )

    @router.delete(
        "/{session_id}",
        response_model=EndSessionResponse,
    )
    async def end_session(
        session_id: str,
    ):

        session = await session_manager.end_session(
            session_id
        )

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Agent session not found",
            )

        return EndSessionResponse(
            session_id=session.session_id,
            status=session.status,
        )

    return router
