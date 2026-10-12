"""
Jeeva AI - Post-Call Data Extraction Interface

Defines the runtime contract for extracting structured information
from a completed call transcript.

This module is intentionally independent of:
- FastAPI
- database models
- Agent CRUD
- Call History persistence

That allows the voice/runtime layer to perform extraction first,
while the API/database layer can decide later where the result
should be stored.

Flow:

    Completed Call
         ↓
    Final Transcript
         ↓
    Agent Extraction Configuration
         ↓
    PostCallDataExtractor
         ↓
    Structured Values
         ↓
    Call record / report
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


class PostCallFieldType(str, Enum):
    """
    Field types supported by the Jeeva Agent configuration UI.
    """

    TEXT = "text"
    SELECTOR = "selector"
    YES_NO = "yes_no"
    NUMBER = "number"


@dataclass(slots=True)
class PostCallField:
    """
    Configuration for one value that should be extracted
    after a call finishes.

    Examples:

        customer_name
            type = text

        interested
            type = yes_no

        budget
            type = number

        preferred_plan
            type = selector
            choices = ["Basic", "Premium"]
    """

    name: str
    description: str
    type: PostCallFieldType

    # Used mainly for text fields when the Agent builder gives
    # the model examples of the expected output format.
    format_examples: list[str] = field(
        default_factory=list
    )

    # Used by selector fields.
    # The extracted value must be one of these choices.
    choices: list[str] = field(
        default_factory=list
    )


PostCallValue: TypeAlias = (
    str
    | bool
    | int
    | float
    | None
)

PostCallExtractionResult: TypeAlias = (
    dict[str, PostCallValue]
)


def build_post_call_fields(
    field_configs: list[dict[str, object]] | None,
) -> list[PostCallField]:
    """
    Convert the Post-Call Data Extraction configuration stored
    inside an Agent into the runtime PostCallField objects used
    by the extractor.

    Expected frontend/Agent field structure:

        {
            "name": "budget",
            "description": "Customer budget",
            "type": "number",
            "format_examples": [],
            "choices": []
        }

    This function intentionally accepts plain dictionaries because
    Pooja's Agent configuration is stored as JSON/dict data.

    That keeps the extraction runtime independent of:
    - FastAPI request models
    - Pydantic Agent schemas
    - database models
    """

    if not field_configs:
        return []

    fields: list[PostCallField] = []

    seen_names: set[str] = set()

    for config in field_configs:

        # ----------------------------------------------------------
        # Basic field information
        # ----------------------------------------------------------

        raw_name = config.get(
            "name",
            "",
        )

        raw_description = config.get(
            "description",
            "",
        )

        raw_type = config.get(
            "type",
            "",
        )

        if not isinstance(
            raw_name,
            str,
        ):
            raise ValueError(
                "Post-call field name must be a string."
            )

        if not isinstance(
            raw_description,
            str,
        ):
            raise ValueError(
                "Post-call field description must be a string."
            )

        if not isinstance(
            raw_type,
            str,
        ):
            raise ValueError(
                "Post-call field type must be a string."
            )

        name = raw_name.strip()

        description = (
            raw_description.strip()
        )

        if not name:
            raise ValueError(
                "Post-call field name cannot be empty."
            )

        # ----------------------------------------------------------
        # Convert the frontend field type into the runtime enum.
        # ----------------------------------------------------------

        try:
            field_type = PostCallFieldType(
                raw_type.strip().lower()
            )

        except ValueError as exc:
            raise ValueError(
                "Unsupported post-call field type: "
                f"{raw_type}"
            ) from exc

        # ----------------------------------------------------------
        # Field names become keys in the extraction result, so they
        # must be unique.
        # ----------------------------------------------------------

        if name in seen_names:
            raise ValueError(
                "Duplicate post-call field name: "
                f"{name}"
            )

        seen_names.add(
            name
        )

        # ----------------------------------------------------------
        # Text field format examples
        # ----------------------------------------------------------

        raw_format_examples = config.get(
            "format_examples",
            [],
        )

        if raw_format_examples is None:
            raw_format_examples = []

        if not isinstance(
            raw_format_examples,
            list,
        ):
            raise ValueError(
                "Post-call format_examples must be a list."
            )

        format_examples: list[str] = []

        if field_type == PostCallFieldType.TEXT:

            for example in raw_format_examples:

                if not isinstance(
                    example,
                    str,
                ):
                    raise ValueError(
                        "Post-call format examples "
                        "must contain strings."
                    )

                normalized_example = (
                    example.strip()
                )

                if normalized_example:
                    format_examples.append(
                        normalized_example
                    )

        # ----------------------------------------------------------
        # Selector choices
        # ----------------------------------------------------------

        raw_choices = config.get(
            "choices",
            [],
        )

        if raw_choices is None:
            raw_choices = []

        if not isinstance(
            raw_choices,
            list,
        ):
            raise ValueError(
                "Post-call choices must be a list."
            )

        choices: list[str] = []

        if (
            field_type
            == PostCallFieldType.SELECTOR
        ):

            for choice in raw_choices:

                if not isinstance(
                    choice,
                    str,
                ):
                    raise ValueError(
                        "Post-call selector choices "
                        "must contain strings."
                    )

                normalized_choice = (
                    choice.strip()
                )

                if (
                    normalized_choice
                    and normalized_choice
                    not in choices
                ):
                    choices.append(
                        normalized_choice
                    )

            if not choices:
                raise ValueError(
                    "Selector fields must contain "
                    "at least one choice."
                )

        # ----------------------------------------------------------
        # Create the runtime field used by the extractor.
        # ----------------------------------------------------------

        fields.append(
            PostCallField(
                name=name,
                description=description,
                type=field_type,
                format_examples=format_examples,
                choices=choices,
            )
        )

    return fields


def build_post_call_transcript(
    messages: list[dict[str, object]],
) -> str:
    """
    Convert Jeeva's stored session messages into the plain-text
    conversation supplied to the post-call extractor.

    Both sides of the conversation are preserved because short
    customer answers such as "yes" may only make sense when the
    preceding Jeeva question is included.

    Interrupted assistant messages are marked explicitly so the
    extraction model knows Jeeva did not necessarily finish saying
    that response to the caller.
    """

    transcript_lines: list[str] = []

    for message in messages:

        role = str(
            message.get("role", "")
        ).strip().lower()

        content = str(
            message.get("content", "")
        ).strip()

        if not content:
            continue

        if role == "user":
            speaker = "Customer"

        elif role == "assistant":
            speaker = "Jeeva"

        else:
            # Ignore unexpected runtime message types rather than
            # allowing unrelated internal data into extraction.
            continue

        interrupted = bool(
            message.get(
                "interrupted",
                False,
            )
        )

        if (
            role == "assistant"
            and interrupted
        ):
            transcript_lines.append(
                f"{speaker} [interrupted]: {content}"
            )
        else:
            transcript_lines.append(
                f"{speaker}: {content}"
            )

    return "\n".join(
        transcript_lines
    ).strip()


class PostCallDataExtractor(ABC):
    """
    Common interface for post-call extraction providers.

    An implementation may use OpenAI or another LLM provider,
    but the rest of Jeeva should depend only on this contract.
    """

    @abstractmethod
    async def extract(
        self,
        *,
        transcript: str,
        fields: list[PostCallField],
        model: str | None = None,
    ) -> PostCallExtractionResult:
        """
        Extract configured values from a completed transcript.

        Rules expected from every implementation:

        - text      -> string or None
        - selector  -> one configured choice or None
        - yes_no    -> bool or None
        - number    -> int/float or None

        If the transcript does not contain enough information,
        the value must be None.

        The extractor must never invent missing information.
        """

        raise NotImplementedError(
            "Post-call extractors must implement extract()"
        )