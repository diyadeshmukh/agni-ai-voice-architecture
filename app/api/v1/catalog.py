"""
Agni AI - Configuration Catalog API

Exposes configuration options that are currently
supported by the Agni backend.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.agent_session_manager import (
    SUPPORTED_AGENT_LANGUAGES,
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
