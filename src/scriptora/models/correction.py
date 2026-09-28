"""Correction models: the validated boundary between the AI and application state.

Nothing from an LLM is ever written into state directly. The LLM's reply is
parsed into `SubtitleCorrection` / `VocabularyAddition`, validated by
Pydantic *and* by a second semantic check (`is_valid_for`), and only then
applied. If validation fails the caller falls back to a deterministic rule
corrector, so a bad model response degrades quality instead of corrupting
the transcript.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator

# Tolerate models that wrap JSON in prose or markdown fences.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class CorrectionAction(str, Enum):
    CORRECT_SUBTITLE = "correct_subtitle"
    ADD_VOCABULARY = "add_vocabulary"
    NO_ACTION = "no_action"


class CorrectionOutcome(str, Enum):
    APPLIED = "applied"
    NO_ACTION = "no_action"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    ERROR = "error"


def extract_json_object(raw: str) -> dict[str, Any]:
    """Best-effort pull of a JSON object out of a model response.

    Raises ValueError when no JSON object can be recovered.
    """
    if not raw or not raw.strip():
        raise ValueError("empty model response")

    candidates: list[str] = []
    fenced = _FENCE.search(raw)
    if fenced:
        candidates.append(fenced.group(1))
    obj = _OBJECT.search(raw)
    if obj:
        candidates.append(obj.group(0))
    candidates.append(raw.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"could not parse a JSON object from model response: {raw[:200]!r}")


class SubtitleCorrection(BaseModel):
    """A validated request to replace one subtitle's text."""

    action: CorrectionAction = CorrectionAction.CORRECT_SUBTITLE
    target_subtitle_id: str | None = None
    replacement_text: str | None = None
    vocabulary_term: str | None = None
    reason: str = ""

    @field_validator("action", mode="before")
    @classmethod
    def _coerce_action(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    def is_valid_for(self, known_ids: set[str]) -> bool:
        """Semantic validation beyond the type system.

        Guards against the two failure modes seen in practice: a model
        inventing an id, and a model returning a non-substantive edit.
        """
        if self.action is not CorrectionAction.CORRECT_SUBTITLE:
            return False
        if not self.replacement_text or not self.replacement_text.strip():
            return False
        if len(self.replacement_text) > 2000:
            return False
        return bool(self.target_subtitle_id) and self.target_subtitle_id in known_ids


class VocabularyAddition(BaseModel):
    """A validated request to add a term to the project vocabulary."""

    action: CorrectionAction = CorrectionAction.ADD_VOCABULARY
    vocabulary_term: str | None = None
    reason: str = ""

    @field_validator("action", mode="before")
    @classmethod
    def _coerce_action(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    def is_valid_for(self) -> bool:
        if self.action is not CorrectionAction.ADD_VOCABULARY:
            return False
        if not self.vocabulary_term or not self.vocabulary_term.strip():
            return False
        return 0 < len(self.vocabulary_term.strip()) <= 80


class CorrectionResult(BaseModel):
    """What the correction engine did - rendered directly in the activity log."""

    outcome: CorrectionOutcome
    message: str
    backend: str = "unknown"
    action: CorrectionAction = CorrectionAction.NO_ACTION
    subtitle_id: str | None = None
    before: str | None = None
    after: str | None = None
    vocabulary_term: str | None = None
    reason: str = ""

    @classmethod
    def failure(
        cls, outcome: CorrectionOutcome, message: str, *, backend: str, reason: str = ""
    ) -> CorrectionResult:
        return cls(outcome=outcome, message=message, backend=backend, reason=reason)
