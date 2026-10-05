"""
Agni AI - Configuration Catalog API

Exposes frontend-selectable configuration supported by
the Agni backend.

Provider-specific voice identifiers are intentionally
never exposed through these endpoints.
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)
from pydantic import BaseModel

from app.services.agent_session_manager import (
    SUPPORTED_AGENT_LANGUAGES,
)
from app.voice.voice_registry import (
    SUPPORTED_VOICE_GENDERS,
    filter_voices,
    get_available_accents,
    get_configured_voices,
)


router = APIRouter(
    prefix="/catalog",
    tags=["catalog"],
)


LANGUAGE_LABELS = {
    "english": "English",
    "hindi": "Hindi",
    "hinglish": "Hinglish",
    "marathi": "Marathi",
}


class LanguageOptionResponse(BaseModel):
    id: str
    label: str


class LanguagesResponse(BaseModel):
    languages: list[LanguageOptionResponse]


class AccentOptionResponse(BaseModel):
    id: str
    label: str


class AccentsResponse(BaseModel):
    accents: list[AccentOptionResponse]


class VoiceOptionResponse(BaseModel):
    id: str
    name: str
    gender: str
    accent: str
    accent_label: str
    languages: list[str]


class VoicesResponse(BaseModel):
    voices: list[VoiceOptionResponse]


def _normalize_language(
    language: str | None,
) -> str | None:

    if language is None:
        return None

    normalized = (
        language
        .strip()
        .lower()
    )

    if (
        normalized
        not in SUPPORTED_AGENT_LANGUAGES
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "language must be one of: "
                "english, hindi, "
                "hinglish, marathi"
            ),
        )

    return normalized


def _normalize_gender(
    gender: str | None,
) -> str | None:

    if gender is None:
        return None

    normalized = (
        gender
        .strip()
        .lower()
    )

    if (
        normalized
        not in SUPPORTED_VOICE_GENDERS
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "gender must be one of: "
                "male, female"
            ),
        )

    return normalized


@router.get(
    "/languages",
    response_model=LanguagesResponse,
)
async def get_languages() -> LanguagesResponse:

    return LanguagesResponse(
        languages=[
            LanguageOptionResponse(
                id=language,
                label=LANGUAGE_LABELS[language],
            )
            for language in sorted(
                SUPPORTED_AGENT_LANGUAGES
            )
        ]
    )


@router.get(
    "/accents",
    response_model=AccentsResponse,
)
async def get_accents(
    language: str | None = Query(
        default=None,
    ),
) -> AccentsResponse:

    normalized_language = (
        _normalize_language(
            language
        )
    )

    available_accents = (
        get_available_accents(
            language=normalized_language,
        )
    )

    # Accent labels come from configured voice profiles.
    accent_labels = {
        voice.accent: voice.accent_label
        for voice in get_configured_voices()
        if (
            normalized_language is None
            or normalized_language
            in voice.languages
        )
    }

    return AccentsResponse(
        accents=[
            AccentOptionResponse(
                id=accent,
                label=accent_labels.get(
                    accent,
                    accent,
                ),
            )
            for accent in sorted(
                available_accents
            )
        ]
    )


@router.get(
    "/voices",
    response_model=VoicesResponse,
)
async def get_voices(
    language: str | None = Query(
        default=None,
    ),
    accent: str | None = Query(
        default=None,
    ),
    gender: str | None = Query(
        default=None,
    ),
) -> VoicesResponse:

    normalized_language = (
        _normalize_language(
            language
        )
    )

    normalized_gender = (
        _normalize_gender(
            gender
        )
    )

    normalized_accent = (
        accent.strip().lower()
        if accent is not None
        else None
    )

    voices = filter_voices(
        language=normalized_language,
        accent=normalized_accent,
        gender=normalized_gender,
    )

    return VoicesResponse(
        voices=[
            VoiceOptionResponse(
                id=voice.id,
                name=voice.name,
                gender=voice.gender,
                accent=voice.accent,
                accent_label=(
                    voice.accent_label
                ),
                languages=sorted(
                    voice.languages
                ),
            )
            for voice in sorted(
                voices,
                key=lambda item: item.id,
            )
        ]
    )
