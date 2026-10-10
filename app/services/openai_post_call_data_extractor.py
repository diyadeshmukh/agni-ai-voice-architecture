"""
Jeeva AI - OpenAI Post-Call Data Extractor

Uses the OpenAI Responses API to extract configured structured
values from a completed call transcript.

This module does not know anything about:
- FastAPI routes
- database tables
- Agent persistence
- Call History persistence

It receives:
    transcript + configured fields

and returns:
    structured extracted values

Example:

    {
        "customer_name": "Rahul",
        "interested": True,
        "budget": 50000,
        "preferred_plan": "Premium",
    }
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.services.post_call_data_extractor import (
    PostCallDataExtractor,
    PostCallExtractionResult,
    PostCallField,
    PostCallFieldType,
)


load_dotenv(".env.local", override=True)


DEFAULT_EXTRACTION_MODEL = "gpt-5.6-luna"


class PostCallExtractionError(RuntimeError):
    """
    Raised when post-call extraction cannot be completed safely.
    """


class OpenAIPostCallDataExtractor(PostCallDataExtractor):
    """
    OpenAI implementation of Jeeva's post-call extraction contract.

    Important behavior:

    - Missing information becomes None.
    - The model is explicitly told not to guess.
    - Selector values must match configured choices.
    - yes_no values must be bool or None.
    - number values must be numeric or None.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> None:

        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
        )

        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing from .env.local"
            )

        # Keep the internal environment-variable naming convention
        # compatible with the existing Jeeva/Agni backend.
        self.model = (
            model
            or os.getenv("AGNI_POST_CALL_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_EXTRACTION_MODEL
        )

        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else int(
                os.getenv(
                    "AGNI_POST_CALL_MAX_OUTPUT_TOKENS",
                    "1000",
                )
            )
        )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
        )

    async def extract(
        self,
        *,
        transcript: str,
        fields: list[PostCallField],
        model: str | None = None,
    ) -> PostCallExtractionResult:
        """
        Extract all configured post-call fields from one transcript.

        No configured fields:
            returns {}

        Empty transcript:
            returns every configured field as None

        Otherwise:
            sends one extraction request to OpenAI and validates
            every returned value before exposing it to the runtime.
        """

        self._validate_fields(fields)

        if not fields:
            return {}

        transcript = transcript.strip()

        # There is nothing safe to extract from an empty transcript.
        if not transcript:
            return {
                field.name: None
                for field in fields
            }

        prompt = self._build_extraction_prompt(
            transcript=transcript,
            fields=fields,
        )

        try:
            response = await self.client.responses.create(
                model=model or self.model,
                instructions=(
                    "You are Jeeva AI's post-call data extraction engine. "
                    "Extract only information explicitly supported by the "
                    "call transcript. Never guess, assume, infer unsupported "
                    "facts, or invent missing information."
                ),
                input=prompt,

                # Structured Outputs forces the model response to match
                # the exact field names and value types configured by the Agent.
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "post_call_extraction",
                        "schema": self._build_output_schema(
                            fields
                        ),
                        "strict": True,
                    }
                },

                max_output_tokens=self.max_output_tokens,
            )
        except Exception as exc:
            raise PostCallExtractionError(
                "OpenAI post-call extraction request failed."
            ) from exc

        raw_output = (
            response.output_text or ""
        ).strip()

        if not raw_output:
            raise PostCallExtractionError(
                "OpenAI returned an empty post-call extraction result."
            )

        parsed = self._parse_json_output(raw_output)

        return self._validate_result(
            parsed=parsed,
            fields=fields,
        )

    @staticmethod
    def _validate_fields(
        fields: list[PostCallField],
    ) -> None:
        """
        Validate Agent extraction configuration before sending
        anything to the LLM.
        """

        seen_names: set[str] = set()

        for field in fields:

            field.name = field.name.strip()
            field.description = field.description.strip()

            if not field.name:
                raise ValueError(
                    "Post-call extraction field name cannot be empty."
                )

            if field.name in seen_names:
                raise ValueError(
                    "Duplicate post-call extraction field name: "
                    f"{field.name}"
                )

            seen_names.add(field.name)

            if (
                field.type == PostCallFieldType.SELECTOR
                and not field.choices
            ):
                raise ValueError(
                    "Selector field "
                    f"'{field.name}' must define at least one choice."
                )

    @staticmethod
    def _build_output_schema(
        fields: list[PostCallField],
    ) -> dict[str, Any]:
        """
        Build the strict JSON Schema used by OpenAI Structured Outputs.

        Every configured field is required in the JSON object.

        If the transcript does not provide enough information,
        the model must explicitly return null for that field.
        """

        properties: dict[
            str,
            dict[str, Any],
        ] = {}

        required: list[str] = []

        for field in fields:

            required.append(
                field.name
            )

            if field.type == PostCallFieldType.TEXT:

                properties[field.name] = {
                    "type": [
                        "string",
                        "null",
                    ],
                }

            elif field.type == PostCallFieldType.YES_NO:

                properties[field.name] = {
                    "type": [
                        "boolean",
                        "null",
                    ],
                }

            elif field.type == PostCallFieldType.NUMBER:

                properties[field.name] = {
                    "type": [
                        "number",
                        "null",
                    ],
                }

            elif field.type == PostCallFieldType.SELECTOR:

                properties[field.name] = {
                    "enum": [
                        *field.choices,
                        None,
                    ],
                }

            else:

                properties[field.name] = {
                    "type": "null",
                }

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    @staticmethod
    def _build_extraction_prompt(
        *,
        transcript: str,
        fields: list[PostCallField],
    ) -> str:
        """
        Build a clear extraction request containing the transcript
        and Agent-configured extraction fields.
        """

        field_definitions: list[dict[str, Any]] = []

        for field in fields:

            definition: dict[str, Any] = {
                "name": field.name,
                "type": field.type.value,
                "description": field.description,
            }

            if field.format_examples:
                definition["format_examples"] = (
                    field.format_examples
                )

            if field.choices:
                definition["choices"] = field.choices

            field_definitions.append(definition)

        fields_json = json.dumps(
            field_definitions,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
Extract the configured post-call fields from the transcript below.

STRICT RULES:

1. Return one JSON object only.
2. Use each configured field name exactly as the JSON key.
3. Do not add extra keys.
4. If the transcript does not clearly provide a value, return null.
5. Never guess or invent information.
6. text:
   - return a JSON string or null.
7. selector:
   - return exactly one configured choice or null.
   - never create a new choice.
8. yes_no:
   - return true, false, or null.
9. number:
   - return a JSON number or null.
   - do not return units or explanatory text.
10. format_examples describe the desired formatting only.
    They are not facts and must never be copied as extracted values
    unless the transcript itself supports them.

CONFIGURED FIELDS:

{fields_json}

CALL TRANSCRIPT:

--- BEGIN TRANSCRIPT ---
{transcript}
--- END TRANSCRIPT ---

Return only the final JSON object.
""".strip()

    @staticmethod
    def _parse_json_output(
        raw_output: str,
    ) -> dict[str, Any]:
        """
        Parse the model response.

        The model is instructed to return JSON only. We deliberately
        reject malformed output instead of trying to guess what the
        model intended.
        """

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise PostCallExtractionError(
                "OpenAI returned invalid JSON for post-call extraction."
            ) from exc

        if not isinstance(parsed, dict):
            raise PostCallExtractionError(
                "Post-call extraction result must be a JSON object."
            )

        return parsed

    @staticmethod
    def _validate_result(
        *,
        parsed: dict[str, Any],
        fields: list[PostCallField],
    ) -> PostCallExtractionResult:
        """
        Validate every LLM-produced value against its configured
        field type.

        Invalid or unsupported model values become None rather than
        leaking unreliable data into Call History.
        """

        result: PostCallExtractionResult = {}

        for field in fields:

            value = parsed.get(field.name)

            if value is None:
                result[field.name] = None
                continue

            if field.type == PostCallFieldType.TEXT:

                if isinstance(value, str):
                    cleaned = value.strip()
                    result[field.name] = (
                        cleaned if cleaned else None
                    )
                else:
                    result[field.name] = None

                continue

            if field.type == PostCallFieldType.SELECTOR:

                if (
                    isinstance(value, str)
                    and value in field.choices
                ):
                    result[field.name] = value
                else:
                    result[field.name] = None

                continue

            if field.type == PostCallFieldType.YES_NO:

                if isinstance(value, bool):
                    result[field.name] = value
                else:
                    result[field.name] = None

                continue

            if field.type == PostCallFieldType.NUMBER:

                # bool is a subclass of int in Python, so explicitly
                # reject it before accepting numeric values.
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    result[field.name] = value
                else:
                    result[field.name] = None

                continue

            # Defensive fallback for any future unsupported field type.
            result[field.name] = None

        return result