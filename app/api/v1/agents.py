"""
Jeeva AI - Agents API

Frontend-facing API for the Agent Builder.

The Agent is the main resource for configuration such as:

- model
- voice
- system prompt
- welcome message
- memory
- emotion
- accent
- functions
- calendar
- CRM
- knowledge base
- speech settings
- call settings
- post-call data extraction
- webhook settings
- prompt variables

Individual Agent Builder sections should not receive separate APIs.
They belong inside the Agent resource.

Current implementation focus:

    Post-Call Data Extraction

The request structure in this file follows the frontend UI shown
in the Jeeva Agent Builder rather than introducing separate APIs
for each Agent subsection.
"""

from __future__ import annotations

from typing import Literal, Self

from fastapi import APIRouter
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


router = APIRouter(
    prefix="/agents",
    tags=["Agents"],
)


# ---------------------------------------------------------------------------
# Post-Call Data Extraction
# ---------------------------------------------------------------------------


PostCallFieldType = Literal[
    "text",
    "selector",
    "yes_no",
    "number",
]


class PostCallExtractionField(BaseModel):
    """
    One extraction field configured through the frontend.

    The Agent Builder currently supports four field types:

        text
        selector
        yes_no
        number

    Frontend behavior:

        text
            name
            description
            optional format_examples

        selector
            name
            description
            choices

        yes_no
            name
            description

        number
            name
            description
    """

    name: str

    description: str

    type: PostCallFieldType

    # --------------------------------------------------------------
    # Text field configuration
    # --------------------------------------------------------------
    #
    # The UI allows optional examples that describe how a text
    # extraction result should be formatted.
    # --------------------------------------------------------------

    format_examples: list[str] = Field(
        default_factory=list
    )

    # --------------------------------------------------------------
    # Selector field configuration
    # --------------------------------------------------------------
    #
    # Selector values must come from the choices configured by
    # the user in the Agent Builder.
    # --------------------------------------------------------------

    choices: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_field_configuration(
        self,
    ) -> Self:
        """
        Validate and normalize one extraction field.

        This follows the frontend field-type behavior:

        - every field requires a name
        - selector requires at least one choice
        - text may contain format examples
        - yes_no and number need no additional configuration
        """

        self.name = self.name.strip()

        self.description = (
            self.description.strip()
        )

        if not self.name:
            raise ValueError(
                "Post-call field name cannot be empty."
            )

        # ----------------------------------------------------------
        # Clean format examples
        # ----------------------------------------------------------

        self.format_examples = [
            example.strip()
            for example in self.format_examples
            if example.strip()
        ]

        # ----------------------------------------------------------
        # Clean selector choices
        # ----------------------------------------------------------

        cleaned_choices: list[str] = []

        for choice in self.choices:

            normalized_choice = (
                choice.strip()
            )

            if not normalized_choice:
                continue

            if (
                normalized_choice
                in cleaned_choices
            ):
                continue

            cleaned_choices.append(
                normalized_choice
            )

        self.choices = cleaned_choices

        # ----------------------------------------------------------
        # Selector-specific validation
        # ----------------------------------------------------------

        if self.type == "selector":

            if not self.choices:
                raise ValueError(
                    "Selector fields must contain "
                    "at least one choice."
                )

        # ----------------------------------------------------------
        # Remove configuration that does not belong to the
        # selected field type.
        #
        # This prevents stale frontend form values from affecting
        # the extraction configuration if the user changes a field
        # from one type to another.
        # ----------------------------------------------------------

        if self.type != "text":
            self.format_examples = []

        if self.type != "selector":
            self.choices = []

        return self


class PostCallDataExtraction(BaseModel):
    """
    Agent Builder Post-Call Data Extraction configuration.

    The frontend section contains:

        model
            GPT/model selector shown at the bottom of the
            Post-Call Data Retrieval section.

        fields
            Extraction fields created with the Add button.

    Copy/Paste/Add are frontend UI actions and therefore are not
    stored as backend configuration fields.

    There is intentionally no `enabled` field here because the
    supplied frontend UI does not show a separate enabled toggle.
    """

    model: str | None = None

    fields: list[
        PostCallExtractionField
    ] = Field(
        default_factory=list
    )

    @field_validator("model")
    @classmethod
    def normalize_model(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Normalize the model selected in the frontend.

        We keep this as a string rather than hardcoding the visible
        model list because the frontend/provider model catalogue can
        change without requiring the Agent schema to be redesigned.
        """

        if value is None:
            return None

        normalized = value.strip()

        return (
            normalized
            if normalized
            else None
        )

    @model_validator(mode="after")
    def validate_unique_field_names(
        self,
    ) -> Self:
        """
        Extraction field names must be unique inside one Agent.

        The extracted result is a JSON object keyed by field name,
        so duplicate names would otherwise overwrite one another.
        """

        seen_names: set[str] = set()

        for field in self.fields:

            if field.name in seen_names:
                raise ValueError(
                    "Duplicate post-call extraction "
                    f"field name: {field.name}"
                )

            seen_names.add(
                field.name
            )

        return self
