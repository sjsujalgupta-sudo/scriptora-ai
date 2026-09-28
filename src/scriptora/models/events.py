"""Server -> browser event envelope.

Every message the server pushes over the audio WebSocket is one of these, so
the browser has a single `type` switch instead of guessing at payload shapes.
"""

from __future__ import annotations

import time
from enum import Enum

from pydantic import BaseModel, Field

from .subtitle import Subtitle


class EventType(str, Enum):
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    STATUS = "status"
    SUBTITLE = "subtitle"
    SUBTITLE_REMOVED = "subtitle_removed"
    CORRECTION = "correction"
    CONTEXT = "context"
    ACTIVITY = "activity"
    ERROR = "error"


class ServerEvent(BaseModel):
    type: EventType
    ts: int = Field(default_factory=lambda: int(time.time() * 1000))
    message: str = ""
    subtitle: Subtitle | None = None
    payload: dict = Field(default_factory=dict)
