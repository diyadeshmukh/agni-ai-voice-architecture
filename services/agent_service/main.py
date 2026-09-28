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


from app.api.v1.agent import (  # noqa: E402
    router as agent_router,
    session_manager,
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
        "Frontend API for creating and managing "
        "Agni AI realtime voice sessions"
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
    agent_router,
    prefix="/api/v1",
)


@app.get("/")
async def home():

    return {
        "message": (
            "Agni AI Agent API is running"
        )
    }