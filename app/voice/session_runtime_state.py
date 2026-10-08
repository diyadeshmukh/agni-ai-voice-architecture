"""
Agni AI - Per-session runtime state bridge.

The realtime voice worker runs in a separate subprocess from FastAPI.
This helper writes a small JSON snapshot that the API process can read
without changing the existing LiveKit/STT/LLM/TTS flow.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SessionRuntimeState:
    """Maintain transcript and lightweight realtime UI state for one session."""

    def __init__(self, file_path: str | None) -> None:
        self.path = Path(file_path) if file_path else None

        self._lock = asyncio.Lock()

        self._data: dict[str, Any] = {
            "transcript": [],
            "realtime": {
                "state": "starting",
                "last_event": "session_starting",
                "updated_at": self._now(),
            },
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    async def initialize(self) -> None:
        async with self._lock:
            self._persist_locked()

    async def add_message(
        self,
        role: str,
        content: str,
        interrupted: bool = False,
    ) -> None:

        content = content.strip()

        if not content:
            return

        async with self._lock:

            self._data[
                "transcript"
            ].append(
                {
                    "role": role,
                    "content": content,
                    "created_at": self._now(),
                    "interrupted": interrupted,
                }
            )

            self._persist_locked()

    async def set_realtime(
        self,
        state: str,
        last_event: str,
    ) -> None:

        async with self._lock:

            self._data["realtime"] = {
                "state": state,
                "last_event": last_event,
                "updated_at": self._now(),
            }

            self._persist_locked()

    def _persist_locked(self) -> None:

        if self.path is None:
            return

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )

        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                self._data,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        os.replace(
            temp_path,
            self.path,
        )