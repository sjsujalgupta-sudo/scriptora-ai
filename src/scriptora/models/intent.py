"""Intent models: the validated boundary between the repair interpreter and state.

The interpreter (an LLM) reads a spoken turn and the recent transcript, and
returns a *plan*, not an edit: which operation was meant, what to find, what to
replace it with, and how sure it is. The plan carries no subtitle ids - target
resolution is the application's job, on the same rules every other command
uses. The plan is validated twice before it can do anything: Pydantic here, and
a transcript lookup in the session (a `replace_text` whose `find` does not
resolve against the subtitles is discarded as content).

Anything that fails validation collapses to `None` in the interpreter, so the
turn simply stays in the transcript as ordinary speech.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator

# A "find" or "replacement" the model returns that is longer than this is not a
# repair of a subtitle word but the model rambling; both are bounded so a
# broken response cannot flow into the executor.
_MAX_FIND = 120
_MAX_REPLACEMENT = 400
_MAX_REASON = 300


class IntentKind(str, Enum):
    NONE = "none"
    CORRECT_SUBTITLE = "correct_subtitle"
    REPLACE_TEXT = "replace_text"
    REMEMBER_TERM = "remember_term"


class IntentTarget(str, Enum):
    LAST = "last"
    CURRENT = "current"
    PREVIOUS = "previous"
    ORDINAL = "ordinal"
    FIND_TEXT = "find_text"


def _clean(value: Any, *, limit: int) -> str | None:
    """Strip and bound a string, collapsing empties and over-long rambling."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > limit:
        return None
    return cleaned


class IntentResult(BaseModel):
    """A validated repair plan. `intent=none` means "leave the turn as content"."""

    intent: IntentKind
    target: IntentTarget | None = None
    ordinal: int | None = None
    find: str | None = None
    replacement: str | None = None
    confidence: float = 0.0
    reason: str = ""

    @field_validator("intent", "target", mode="before")
    @classmethod
    def _coerce_enum(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("ordinal", mode="before")
    @classmethod
    def _coerce_ordinal(cls, value: Any) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            ordinal = int(value)
        except (TypeError, ValueError):
            return None
        # 1-based line numbers only; a 0 or a 5-digit line is a model error.
        return ordinal if 1 <= ordinal <= 9999 else None

    @field_validator("find", mode="before")
    @classmethod
    def _coerce_find(cls, value: Any) -> str | None:
        return _clean(value, limit=_MAX_FIND)

    @field_validator("replacement", mode="before")
    @classmethod
    def _coerce_replacement(cls, value: Any) -> str | None:
        return _clean(value, limit=_MAX_REPLACEMENT)

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: Any) -> float:
        if isinstance(value, bool) or value is None:
            return 0.0
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

    @field_validator("reason", mode="before")
    @classmethod
    def _coerce_reason(cls, value: Any) -> str:
        # truncate, not drop: unlike find/replacement a long reason is garbage
        # in a display field, and keeping its beginning preserves the meaning
        if not isinstance(value, str):
            return ""
        return value.strip()[:_MAX_REASON]

    @property
    def is_repair(self) -> bool:
        return self.intent is not IntentKind.NONE
