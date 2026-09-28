"""Subtitle state model.

A Subtitle is one AssemblyAI turn as Scriptora sees it. Partial turns are
transient (a live in-progress utterance) and are replaced in place when the
same turn is finalized, so the UI never shows a duplicate line.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field


class SubtitleStatus(str, Enum):
    """Lifecycle of a subtitle line.

    partial   -> AssemblyAI is still revising this turn (text is provisional)
    final     -> AssemblyAI closed the turn; text is stable
    corrected -> a human/AI correction has been applied on top of `raw_text`
    """

    PARTIAL = "partial"
    FINAL = "final"
    CORRECTED = "corrected"


def _now_ms() -> int:
    return int(time.time() * 1000)


class Subtitle(BaseModel):
    """One subtitle line. Deliberately small - id, text, timing, status."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str
    status: SubtitleStatus = SubtitleStatus.PARTIAL

    # Milliseconds, monotonic within a session.
    start_time: int = Field(default_factory=_now_ms)
    end_time: int | None = None

    # AssemblyAI turn bookkeeping, useful for the activity log.
    turn_order: int | None = None
    confidence: float | None = None

    # Text before any correction. Never mutated, so the UI can always show
    # a raw-vs-corrected comparison.
    raw_text: str | None = None

    def finalize(self, text: str | None = None, end_time: int | None = None) -> None:
        """Promote a partial subtitle to final.

        When the finalized text differs from the raw text, `raw_text` is
        captured first so the original transcription is never lost.
        """
        if text is not None and text.strip() != self.text.strip():
            if self.raw_text is None:
                self.raw_text = self.text
            self.text = text.strip()
        self.status = SubtitleStatus.FINAL
        if end_time is not None:
            self.end_time = end_time
        elif self.end_time is None:
            self.end_time = _now_ms()

    def apply_correction(self, replacement_text: str) -> None:
        """Apply a validated correction.

        `raw_text` preserves the original transcript, enabling the RAW vs
        CORRECTED comparison panel.
        """
        replacement_text = replacement_text.strip()
        if not replacement_text:
            raise ValueError("replacement_text must not be empty")
        if self.raw_text is None:
            self.raw_text = self.text
        self.text = replacement_text
        self.status = SubtitleStatus.CORRECTED
        if self.end_time is None:
            self.end_time = _now_ms()

    @property
    def is_provisional(self) -> bool:
        return self.status is SubtitleStatus.PARTIAL

    @property
    def was_corrected(self) -> bool:
        return self.status is SubtitleStatus.CORRECTED and (
            self.raw_text is not None and self.raw_text.strip() != self.text.strip()
        )
