"""
Agni AI - Voice Registry

Central source of truth for frontend-selectable voices.

Frontend-facing voice IDs stay stable and provider-neutral.
Actual ElevenLabs voice IDs are loaded from environment variables
and are never exposed through the API.
"""

from __future__ import annotations

import os

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    name: str
    gender: str
    accent: str
    accent_label: str
    languages: frozenset[str]
    provider: str
    provider_voice_env: str

    @property
    def provider_voice_id(self) -> str | None:
        value = os.getenv(
            self.provider_voice_env
        )

        if not value:
            return None

        value = value.strip()

        return value or None

    @property
    def configured(self) -> bool:
        return self.provider_voice_id is not None


# ---------------------------------------------------------------------------
# Supported metadata
# ---------------------------------------------------------------------------

SUPPORTED_VOICE_GENDERS = {
    "male",
    "female",
}


# ---------------------------------------------------------------------------
# Voice profiles
#
# Do not add arbitrary provider voice IDs here.
#
# Each production voice should have:
#
# - stable Agni public ID
# - display name
# - gender
# - accent
# - supported languages
# - environment variable containing the provider voice ID
#
# Profiles will be added once the actual ElevenLabs voices
# for the Agni account have been selected.
# ---------------------------------------------------------------------------

VOICE_PROFILES: dict[str, VoiceProfile] = {}


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def get_voice_profile(
    voice_id: str,
) -> VoiceProfile | None:

    normalized = (
        voice_id
        .strip()
        .lower()
    )

    return VOICE_PROFILES.get(
        normalized
    )


def get_configured_voices(
) -> list[VoiceProfile]:

    return [
        voice
        for voice in VOICE_PROFILES.values()
        if voice.configured
    ]


def get_available_accents(
    *,
    language: str | None = None,
) -> set[str]:

    voices = get_configured_voices()

    if language is not None:
        normalized_language = (
            language
            .strip()
            .lower()
        )

        voices = [
            voice
            for voice in voices
            if normalized_language
            in voice.languages
        ]

    return {
        voice.accent
        for voice in voices
    }


def filter_voices(
    *,
    language: str | None = None,
    accent: str | None = None,
    gender: str | None = None,
) -> list[VoiceProfile]:

    voices = get_configured_voices()

    if language is not None:
        normalized_language = (
            language
            .strip()
            .lower()
        )

        voices = [
            voice
            for voice in voices
            if normalized_language
            in voice.languages
        ]

    if accent is not None:
        normalized_accent = (
            accent
            .strip()
            .lower()
        )

        voices = [
            voice
            for voice in voices
            if voice.accent
            == normalized_accent
        ]

    if gender is not None:
        normalized_gender = (
            gender
            .strip()
            .lower()
        )

        voices = [
            voice
            for voice in voices
            if voice.gender
            == normalized_gender
        ]

    return voices