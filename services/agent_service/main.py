"""
Agni AI - Agent API Service

Frontend-facing API for starting and managing
Agni AI voice sessions.

Run:

    uvicorn services.agent_service.main:app --port 8001
"""

from __future__ import annotations

import os

from contextlib import (
    asynccontextmanager,
)

from dotenv import load_dotenv
from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware,
)


load_dotenv(
    ".env.local",
    override=True,
)


from app.api.v1.catalog import (  # noqa: E402
    router as catalog_router,
)

from app.api.v1.health import (  # noqa: E402
    create_health_router,
)

from app.api.v1.sessions import (  # noqa: E402
    create_sessions_router,
)

# Text chatbot API used by the floating Agni chat widget.
# This is separate from the realtime LiveKit voice-session API.
from app.api.v1.chat import (  # noqa: E402
    create_chat_router,
)

from app.services.agent_session_manager import (  # noqa: E402
    AgentSessionManager,
)

# Handles text-only chatbot conversations and OpenAI responses.
from app.services.chat_service import (  # noqa: E402
    ChatService,
)


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------
#
# Voice session manager:
# Handles the realtime LiveKit voice-agent sessions.
#
# Chat service:
# Handles the separate floating text chatbot.
# ---------------------------------------------------------------------------

session_manager = AgentSessionManager()

chat_service = ChatService()


# ---------------------------------------------------------------------------
# API Routers
# ---------------------------------------------------------------------------

# Realtime voice-agent sessions.
sessions_router = create_sessions_router(
    session_manager
)

# Floating text chatbot.
chat_router = create_chat_router(
    chat_service
)

# Existing service health endpoint.
health_router = create_health_router(
    session_manager
)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    yield

    await session_manager.shutdown()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agni AI Agent API",
    description=(
        "Frontend API for Agni AI realtime voice sessions "
        "and the floating text chatbot"
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

default_frontend_origins = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)

configured_origins = os.getenv(
    "AGNI_FRONTEND_ORIGINS",
    default_frontend_origins,
)

frontend_origins = [
    origin.strip()
    for origin in (
        configured_origins.split(",")
    )
    if origin.strip()
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(
    catalog_router,
    prefix="/api/v1",
)

# Realtime voice-agent sessions.
app.include_router(
    sessions_router,
    prefix="/api/v1",
)

# Floating Agni text chatbot.
#
# Final endpoint:
# POST /api/v1/chat
app.include_router(
    chat_router,
    prefix="/api/v1",
)

app.include_router(
    health_router,
    prefix="/api/v1",
)


@app.get("/")
async def home():

    return {
        "message": (
            "Agni AI Agent API is running"
        )
    }