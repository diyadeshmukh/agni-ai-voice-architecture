"""
Jeeva AI - Agent Session Manager

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
- Jeeva participant identity
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

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from livekit import api

from app.voice.voice_registry import (
    get_voice_profile,
)

from app.services.post_call_data_extractor import (
    PostCallExtractionResult,
    PostCallField,
    build_post_call_fields,
    build_post_call_transcript,
)

from app.services.openai_post_call_data_extractor import (
    OpenAIPostCallDataExtractor,
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
    Internal representation of one running Jeeva voice session.
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

    # --------------------------------------------------------------
    # Post-call extraction configuration
    #
    # These values come from the Agent's Post-Call Data Extraction
    # section. The session manager keeps them with the running call
    # so extraction can happen after the call finishes.
    # --------------------------------------------------------------

    post_call_model: str | None = None

    post_call_fields: list[PostCallField] = field(
        default_factory=list
    )

    # Filled only after post-call extraction has completed.
    # Pooja's Call persistence layer can later store this result.
    post_call_result: PostCallExtractionResult | None = None

    # If post-call processing fails, the call itself should still
    # finish normally. This stores the processing error separately.
    post_call_error: str | None = None

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
                "Jeeva AI Web User"
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
        post_call_data_extraction: dict[str, object] | None = None,
    ) -> tuple[
        AgentSession,
        str,
    ]:
        """
        Start one isolated Jeeva agent process.

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

        # --------------------------------------------------------------
        # Post-Call Data Extraction configuration
        # --------------------------------------------------------------
        #
        # This configuration comes from:
        #
        # Agent.post_call_data_extraction
        #
        # Frontend / Agent shape:
        #
        # {
        #     "model": "gpt-4o-mini",
        #     "fields": [...]
        # }
        #
        # Convert the stored Agent JSON into the runtime objects used by
        # the post-call extractor.
        # --------------------------------------------------------------

        post_call_model: str | None = None

        post_call_fields: list[PostCallField] = []

        if post_call_data_extraction:

            # ----------------------------------------------------------
            # Extraction model
            # ----------------------------------------------------------

            raw_model = (
                post_call_data_extraction.get(
                    "model"
                )
            )

            if raw_model is not None:

                if not isinstance(
                    raw_model,
                    str,
                ):
                    raise ValueError(
                        "Post-call extraction model "
                        "must be a string."
                    )

                post_call_model = (
                    raw_model.strip()
                    or None
                )

            # ----------------------------------------------------------
            # Extraction fields
            # ----------------------------------------------------------

            raw_fields = (
                post_call_data_extraction.get(
                    "fields",
                    [],
                )
            )

            if raw_fields is None:
                raw_fields = []

            if not isinstance(
                raw_fields,
                list,
            ):
                raise ValueError(
                    "Post-call extraction fields "
                    "must be a list."
                )

            field_configs: list[
                dict[str, object]
            ] = []

            for field_config in raw_fields:

                if not isinstance(
                    field_config,
                    dict,
                ):
                    raise ValueError(
                        "Each post-call extraction "
                        "field must be an object."
                    )

                field_configs.append(
                    field_config
                )

            post_call_fields = (
                build_post_call_fields(
                    field_configs
                )
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
            / "jeeva-ai"
            / "session-readiness"
            / f"{session_id}.ready"
        )

        runtime_file = (
            Path(
                tempfile.gettempdir()
            )
            / "jeeva-ai"
            / "session-runtime"
            / f"{session_id}.json"
        )

        room_name = (
            f"jeeva-session-{short_id}"
        )

        frontend_identity = (
            f"jeeva-web-{short_id}"
        )

        agent_identity = (
            f"jeeva-agent-{short_id}"
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
        # Technical environment-variable names remain AGNI_* for
        # backward compatibility with the existing runtime/integration.
        #
        # Product-facing branding is Jeeva AI, but changing these
        # internal variable names is not required for the rebrand.
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
            "AGNI_SESSION_ID"
        ] = session_id

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
            post_call_model=post_call_model,

            post_call_fields=post_call_fields,
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
    # Post-call extraction
    # ------------------------------------------------------------------

    async def _run_post_call_extraction(
        self,
        session: AgentSession,
    ) -> None:
        """
        Run the Agent's configured Post-Call Data Extraction
        after the realtime voice process has stopped.

        The transcript already lives in SessionRuntimeState,
        so there is no need to duplicate transcript collection
        inside the voice pipeline.
        """

        # No configured extraction fields means there is
        # nothing to process for this Agent.
        if not session.post_call_fields:
            session.post_call_result = {}
            session.post_call_error = None
            return

        runtime = session.runtime_snapshot()

        transcript = build_post_call_transcript(
            runtime["transcript"]
        )

        try:
            extractor = OpenAIPostCallDataExtractor()

            session.post_call_result = (
                await extractor.extract(
                    transcript=transcript,
                    fields=session.post_call_fields,
                    model=session.post_call_model,
                )
            )

            session.post_call_error = None

            print()
            print(
                "[POST-CALL] Extraction completed: "
                f"{session.post_call_result}"
            )

        except Exception as exc:
            # Post-call processing must never prevent the call
            # itself from ending successfully.
            session.post_call_result = None
            session.post_call_error = str(exc)

            print()
            print(
                "[POST-CALL ERROR] "
                f"{exc}"
            )

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

        # ----------------------------------------------------------
        # Post-call processing
        # ----------------------------------------------------------
        #
        # The voice subprocess has stopped, so the persisted
        # transcript is now treated as the completed call transcript.
        #
        # Extraction runs in the parent API process because this
        # process owns the Agent/session configuration and can later
        # hand the result to Call History persistence.
        # ----------------------------------------------------------

        await self._run_post_call_extraction(
            session
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
