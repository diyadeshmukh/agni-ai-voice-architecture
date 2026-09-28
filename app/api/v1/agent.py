"""
Agni AI - Frontend Agent API

Frontend-facing endpoints used to create and manage
Agni AI voice sessions.
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


router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


session_manager = (
    AgentSessionManager()
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CreateAgentSessionRequest(
    BaseModel
):
    language: str = "english"

    @field_validator(
        "language"
    )
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


class LiveKitConnectionResponse(
    BaseModel
):
    url: str
    room_name: str
    token: str


class TrackContractResponse(
    BaseModel
):
    microphone: str
    agent_audio: str


class CreateAgentSessionResponse(
    BaseModel
):
    session_id: str
    status: str
    language: str

    livekit: (
        LiveKitConnectionResponse
    )

    tracks: (
        TrackContractResponse
    )

    created_at: datetime


class AgentSessionResponse(
    BaseModel
):
    session_id: str
    status: str
    language: str

    room_name: str

    frontend_identity: str
    agent_identity: str

    created_at: datetime
    ended_at: datetime | None


class EndAgentSessionResponse(
    BaseModel
):
    session_id: str
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def session_response(
    session: AgentSession,
) -> AgentSessionResponse:

    session.refresh_status()

    return AgentSessionResponse(
        session_id=(
            session.session_id
        ),
        status=session.status,
        language=session.language,
        room_name=session.room_name,
        frontend_identity=(
            session.frontend_identity
        ),
        agent_identity=(
            session.agent_identity
        ),
        created_at=session.created_at,
        ended_at=session.ended_at,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get(
    "/health"
)
async def agent_health():

    try:

        livekit_url = (
            session_manager.livekit_url
        )

        livekit_configured = bool(
            livekit_url
        )

    except RuntimeError:

        livekit_configured = False

    return {
        "status": (
            "ok"
            if livekit_configured
            else "configuration_error"
        ),
        "service": "Agni Agent API",
        "languages": sorted(
            SUPPORTED_AGENT_LANGUAGES
        ),
        "livekit_configured": (
            livekit_configured
        ),
    }


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

@router.post(
    "/sessions",
    response_model=(
        CreateAgentSessionResponse
    ),
    status_code=(
        status.HTTP_201_CREATED
    ),
)
async def create_agent_session(
    request: CreateAgentSessionRequest,
):

    try:

        (
            session,
            frontend_token,
        ) = (
            await session_manager
            .create_session(
                request.language
            )
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return (
        CreateAgentSessionResponse(
            session_id=(
                session.session_id
            ),
            status=session.status,
            language=session.language,
            livekit=(
                LiveKitConnectionResponse(
                    url=(
                        session_manager
                        .livekit_url
                    ),
                    room_name=(
                        session.room_name
                    ),
                    token=frontend_token,
                )
            ),
            tracks=(
                TrackContractResponse(
                    microphone=(
                        "microphone"
                    ),
                    agent_audio=(
                        "voice-output"
                    ),
                )
            ),
            created_at=(
                session.created_at
            ),
        )
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get(
    "/sessions/{session_id}",
    response_model=(
        AgentSessionResponse
    ),
)
async def get_agent_session(
    session_id: str,
):

    session = (
        session_manager
        .get_session(
            session_id
        )
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Agent session not found"
            ),
        )

    return session_response(
        session
    )


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

@router.delete(
    "/sessions/{session_id}",
    response_model=(
        EndAgentSessionResponse
    ),
)
async def end_agent_session(
    session_id: str,
):

    session = (
        await session_manager
        .end_session(
            session_id
        )
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Agent session not found"
            ),
        )

    return (
        EndAgentSessionResponse(
            session_id=(
                session.session_id
            ),
            status=session.status,
        )
    )