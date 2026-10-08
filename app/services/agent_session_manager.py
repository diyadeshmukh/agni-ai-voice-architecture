"""
Agni AI - Agent Session Manager

Creates and manages frontend voice-agent sessions.

Phase 1 architecture:

    Frontend
        ↓
    Agent API
        ↓
    AgentSessionManager
        ↓
    dedicated audio_subscriber subprocess
        ↓
    LiveKit / STT / OpenAI / ElevenLabs

Each session receives its own:

- session ID
- LiveKit room
- frontend participant identity
- Agni participant identity
- language configuration
- agent subprocess

This keeps the already-tested voice pipeline isolated while
allowing multiple frontend sessions to run independently.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
import tempfile

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from livekit import api

from app.voice.voice_registry import (
    get_voice_profile,
)


load_dotenv(
    ".env.local",
    override=True,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


SUPPORTED_AGENT_LANGUAGES = {
    "english",
    "hindi",
    "hinglish",
    "marathi",
}


@dataclass
class AgentSession:
    """
    Internal representation of one running Agni voice session.
    """

    session_id: str
    language: str
    voice_id: str | None
    ready_file: Path
    runtime_file: Path

    room_name: str

    frontend_identity: str
    agent_identity: str

    process: asyncio.subprocess.Process

    created_at: datetime

    status: str = "starting"
    ended_at: datetime | None = None

    def refresh_status(self) -> str:
        """
        Update the session status from the subprocess
        and voice-pipeline readiness marker.
        """

        if self.status == "ended":
            return self.status

        return_code = self.process.returncode

        if return_code is not None:
            self.status = "failed"

            if self.ended_at is None:
                self.ended_at = datetime.now(
                    timezone.utc
                )

            return self.status

        if self.ready_file.exists():
            self.status = "ready"
        else:
            self.status = "starting"

        return self.status
    
    def runtime_snapshot(self) -> dict:
        """
        Read transcript and realtime state
        written by the realtime voice worker.
        """

        fallback_state = (
            "ended"
            if self.status in {
                "ended",
                "failed",
            }
            else self.status
        )

        fallback = {
            "transcript": [],
            "realtime": {
                "state": fallback_state,
                "last_event": None,
                "updated_at": None,
            },
        }

        if not self.runtime_file.exists():
            return fallback

        try:

            with self.runtime_file.open(
                "r",
                encoding="utf-8",
            ) as handle:

                data = json.load(
                    handle
                )

        except (
            OSError,
            json.JSONDecodeError,
        ):
            return fallback

        if not isinstance(
            data,
            dict,
        ):
            return fallback

        transcript = data.get(
            "transcript",
            [],
        )

        realtime = data.get(
            "realtime",
            {},
        )

        if not isinstance(
            transcript,
            list,
        ):
            transcript = []

        if not isinstance(
            realtime,
            dict,
        ):
            realtime = {}

        return {
            "transcript": transcript,
            "realtime": {
                "state": realtime.get(
                    "state",
                    fallback_state,
                ),
                "last_event": realtime.get(
                    "last_event"
                ),
                "updated_at": realtime.get(
                    "updated_at"
                ),
            },
        }


class AgentSessionManager:
    """
    In-memory session manager used by the Agent API.

    This is intentionally simple for the current project/demo.

    A production system could later replace the in-memory state
    with Redis/database-backed session management and workers.
    """

    def __init__(self) -> None:

        self.sessions: dict[
            str,
            AgentSession,
        ] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def livekit_url(self) -> str:

        value = os.getenv(
            "LIVEKIT_URL"
        )

        if not value:
            raise RuntimeError(
                "LIVEKIT_URL is missing from .env.local"
            )

        return value

    def _livekit_credentials(
        self,
    ) -> tuple[str, str]:

        api_key = os.getenv(
            "LIVEKIT_API_KEY"
        )

        api_secret = os.getenv(
            "LIVEKIT_API_SECRET"
        )

        if not api_key or not api_secret:

            raise RuntimeError(
                "LIVEKIT_API_KEY or LIVEKIT_API_SECRET "
                "is missing from .env.local"
            )

        return (
            api_key,
            api_secret,
        )

    # ------------------------------------------------------------------
    # Frontend LiveKit token
    # ------------------------------------------------------------------

    def create_frontend_token(
        self,
        *,
        room_name: str,
        participant_identity: str,
    ) -> str:
        """
        Create a short-lived participant token for the frontend.

        The frontend receives this token.

        LIVEKIT_API_SECRET is never sent to the frontend.
        """

        (
            api_key,
            api_secret,
        ) = self._livekit_credentials()

        return (
            api.AccessToken(
                api_key,
                api_secret,
            )
            .with_identity(
                participant_identity
            )
            .with_name(
                "Agni AI Web User"
            )
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .to_jwt()
        )

    # ------------------------------------------------------------------
    # Create session
    # ------------------------------------------------------------------

    async def create_session(
        self,
        language: str,
        system_prompt: str | None = None,
        voice_id: str | None = None,
        welcome_message: str | None = None,
    ) -> tuple[
        AgentSession,
        str,
    ]:
        """
        Start one isolated Agni agent process.

        Returns:

            AgentSession
            frontend LiveKit token
        """

        language = (
            language
            .strip()
            .lower()
        )

        if (
            language
            not in SUPPORTED_AGENT_LANGUAGES
        ):

            raise ValueError(
                "Unsupported language. "
                "Use english, hindi, "
                "hinglish, or marathi."
            )

        selected_voice_id: str | None = None
        provider_voice_id: str | None = None

        if voice_id is not None:

            voice = get_voice_profile(
                voice_id
            )

            if voice is None:
                raise ValueError(
                    "Unknown voice_id."
                )

            if language not in voice.languages:
                raise ValueError(
                    "Selected voice does not support "
                    f"language '{language}'."
                )

            if voice.provider != "elevenlabs":
                raise ValueError(
                    "Selected voice provider is not "
                    "supported by the current TTS runtime."
                )

            provider_voice_id = (
                voice.provider_voice_id
            )

            if provider_voice_id is None:
                raise ValueError(
                    "Selected voice is not configured."
                )

            selected_voice_id = voice.id

        # Validate LiveKit configuration before spawning anything.
        _ = self.livekit_url

        session_id = (
            uuid.uuid4().hex
        )

        short_id = (
            session_id[:12]
        )

        ready_file = (
            Path(
                tempfile.gettempdir()
            )
            / "agni-ai"
            / "session-readiness"
            / f"{session_id}.ready"
        )
        
        runtime_file = (
            Path(
                tempfile.gettempdir()
            )
            / "agni-ai"
            / "session-runtime"
            / f"{session_id}.json"
        )

        room_name = (
            f"agni-session-{short_id}"
        )

        frontend_identity = (
            f"agni-web-{short_id}"
        )

        agent_identity = (
            f"agni-agent-{short_id}"
        )

        frontend_token = (
            self.create_frontend_token(
                room_name=room_name,
                participant_identity=(
                    frontend_identity
                ),
            )
        )

        # --------------------------------------------------------------
        # Dedicated process environment
        #
        # These AGNI_SESSION_* variables intentionally have different
        # names from the existing CLI variables so .env.local cannot
        # accidentally override the per-session configuration.
        # --------------------------------------------------------------

        process_env = os.environ.copy()

        process_env[
            "AGNI_SESSION_ROOM_NAME"
        ] = room_name

        process_env[
            "AGNI_SESSION_PARTICIPANT_IDENTITY"
        ] = agent_identity

        process_env[
            "AGNI_SESSION_LANGUAGE"
        ] = language

        process_env[
            "AGNI_SESSION_SYSTEM_PROMPT"
        ] = system_prompt or ""

        process_env[
            "AGNI_SESSION_WELCOME_MESSAGE"
        ] = welcome_message or ""

        process_env[
            "AGNI_SESSION_VOICE_ID"
        ] = provider_voice_id or ""

        process_env[
            "AGNI_SESSION_READY_FILE"
        ] = str(
            ready_file
        )
        
        process_env[
            "AGNI_SESSION_RUNTIME_FILE"
        ] = str(
            runtime_file
        )

        process_env[
            "PYTHONUNBUFFERED"
        ] = "1"

        # Ensure a stale readiness marker cannot make
        # a new session appear ready immediately.
        ready_file.unlink(
            missing_ok=True
        )
        
        runtime_file.unlink(
            missing_ok=True
        )

        process = (
            await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "livekit_poc.audio_subscriber",
                cwd=str(
                    PROJECT_ROOT
                ),
                env=process_env,
            )
        )

        session = AgentSession(
            session_id=session_id,
            language=language,
            voice_id=selected_voice_id,
            ready_file=ready_file,
            runtime_file=runtime_file,
            room_name=room_name,
            
            frontend_identity=(
                frontend_identity
            ),
            agent_identity=(
                agent_identity
            ),
            process=process,
            created_at=datetime.now(
                timezone.utc
            ),
        )

        self.sessions[
            session_id
        ] = session

        return (
            session,
            frontend_token,
        )

    # ------------------------------------------------------------------
    # Get session
    # ------------------------------------------------------------------

    def get_session(
        self,
        session_id: str,
    ) -> AgentSession | None:

        session = self.sessions.get(
            session_id
        )

        if session is None:
            return None

        session.refresh_status()

        return session

    # ------------------------------------------------------------------
    # End session
    # ------------------------------------------------------------------

    async def end_session(
        self,
        session_id: str,
    ) -> AgentSession | None:
        """
        Stop the agent subprocess belonging to this session.
        """

        session = self.sessions.get(
            session_id
        )

        if session is None:
            return None

        session.refresh_status()

        if (
            session.process.returncode
            is None
        ):

            session.process.terminate()

            try:

                await asyncio.wait_for(
                    session.process.wait(),
                    timeout=5.0,
                )

            except asyncio.TimeoutError:

                session.process.kill()

                await session.process.wait()

        session.ready_file.unlink(
            missing_ok=True
        )

        session.status = "ended"

        session.ended_at = (
            datetime.now(
                timezone.utc
            )
        )

        return session

    # ------------------------------------------------------------------
    # Service shutdown
    # ------------------------------------------------------------------

    async def shutdown(
        self,
    ) -> None:
        """
        Stop all agent sessions when the Agent API shuts down.
        """

        active_session_ids = [
            session_id
            for (
                session_id,
                session,
            )
            in self.sessions.items()
            if (
                session.process.returncode
                is None
            )
        ]

        for session_id in (
            active_session_ids
        ):

            await self.end_session(
                session_id
            )
