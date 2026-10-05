"""
Agni AI - Health API

Service health and configuration status.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.services.agent_session_manager import (
    AgentSessionManager,
)


def create_health_router(
    session_manager: AgentSessionManager,
) -> APIRouter:

    router = APIRouter(
        tags=["system"],
    )

    @router.get("/health")
    async def health():

        try:
            livekit_configured = bool(
                session_manager.livekit_url
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
            "livekit_configured": (
                livekit_configured
            ),
        }

    return router
