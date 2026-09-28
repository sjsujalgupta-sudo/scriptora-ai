"""One Scriptora listening session.

Owns everything scoped to a single browser connection:

* the AssemblyAI real-time session
* the subtitle store and project context
* the correction service
* the asyncio tasks pumping audio to AssemblyAI and events back to the browser

Subtitle lines are not stored in a database. A session is intentionally
ephemeral - when the tab closes, the transcript is gone.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ..config import Settings
from ..models.correction import CorrectionAction, CorrectionOutcome, CorrectionResult
from ..models.events import EventType, ServerEvent
from ..models.subtitle import Subtitle
from .assemblyai_service import AssemblyAIRealtimeService
from .command_service import CommandKind, ParsedCommand, parse_command
from .context_service import ContextService
from .correction_service import CorrectionService
from .subtitle_service import SubtitleService

logger = logging.getLogger(__name__)

# Vocabulary terms recognized as commands. Kept small on purpose: a wide net
# would misread ordinary dictation ("remember to lock the door") as a command.
_COMMANDS: tuple[tuple[CommandKind, tuple[str, ...]], ...] = (
    (CommandKind.REMEMBER, ("remember ", "remember\u00a0")),
    (
        CommandKind.CORRECT_PREVIOUS,
        ("fix the previous", "correct the previous", "fix previous", "correct previous"),
    ),
    (
        CommandKind.CORRECT_LAST,
        (
            "correct the last",
            "fix the last",
            "correct last",
            "fix last",
            "correct it",
        ),
    ),
    (CommandKind.REPLACE, ("change ", "replace ", "swap ")),
)


class ScriptoraSession:
    """A single listening session bound to one WebSocket."""

    def __init__(self, settings: Settings, send: Any) -> None:
        self._settings = settings
        self._send = send
        self.id = uuid.uuid4().hex[:8]

        self.subtitles = SubtitleService()
        self.context = ContextService()
        self.corrections = CorrectionService(settings, self.subtitles, self.context)
        self._assemblyai = AssemblyAIRealtimeService(settings)

        self.status = "idle"
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._stream_task: asyncio.Task[None] | None = None
        self._running = False
        self._turns_seen = 0
        self._correction_count = 0

    # ---------------------------------------------------------------- setup
    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Connect to AssemblyAI and begin pumping audio."""
        if self._running:
            return
        self._running = True
        self.status = "connecting"

        await self._emit(
            EventType.SESSION_STARTED,
            f"Scriptora session {self.id} started.",
            payload={
                "session_id": self.id,
                "config": self._settings.public_summary(),
                "vocabulary": self.context.terms,
            },
        )
        await self._emit_context()

        try:
            await self._assemblyai.connect(
                self.context.keyterms(),
                on_partial=self._handle_partial,
                on_final=self._handle_final,
                on_speech_start=self._handle_speech_started,
                on_status=self._handle_status,
                on_error=self._handle_error,
            )
        except Exception as exc:
            self._running = False
            self.status = "error"
            message = self._friendly_connect_error(exc)
            logger.warning("Could not connect to AssemblyAI: %s", type(exc).__name__)
            await self._emit(EventType.ERROR, message, payload={"fatal": True})
            await self._emit(EventType.STATUS, "Disconnected.")
            return

        # The SDK's `stream` consumes an async iterable once, so it runs as a
        # long-lived task that ends when `stop()` queues the sentinel.
        self._stream_task = asyncio.create_task(
            self._assemblyai.stream(self._frame_generator()),
            name=f"scriptora-stream-{self.id}",
        )

    async def stop(self) -> None:
        """Stop listening and release the AssemblyAI session."""
        if not self._running:
            return
        self._running = False
        self.status = "stopping"
        await self._emit(EventType.STATUS, "Stopping...")

        # End the frame generator, let the stream task finish, then terminate
        # the AssemblyAI session so the final turn is flushed and billing stops.
        await self._audio_queue.put(None)
        if self._stream_task is not None:
            try:
                await asyncio.wait_for(self._stream_task, timeout=5.0)
            except TimeoutError:
                self._stream_task.cancel()
            except Exception as exc:
                logger.info("Audio stream ended with %s", type(exc).__name__)
            self._stream_task = None

        await self._assemblyai.disconnect()
        self.status = "idle"

        await self._emit(
            EventType.SESSION_ENDED,
            f"Stopped. {self._turns_seen} turn(s), {self._correction_count} correction(s).",
            payload={
                "turns": self._turns_seen,
                "corrections": self._correction_count,
                "subtitle_count": len(self.subtitles),
            },
        )

    # ----------------------------------------------------------------- audio
    async def send_audio(self, chunk: bytes) -> None:
        """Queue one audio frame from the browser."""
        if not self._running:
            return
        await self._audio_queue.put(chunk)

    async def _frame_generator(self) -> AsyncIterator[bytes]:
        """Yield AssemblyAI-sized frames pulled from the browser queue.

        Frames are resized to `frame_bytes`. The browser is asked to send exact
        frames, but jitter at the WebSocket layer can produce a short final
        frame, and AssemblyAI rejects anything outside 50-1000 ms with error
        3007. Normalising here removes that whole class of failure.

        The generator ends when `None` is queued, which is how `stop()` drains
        the session cleanly.
        """
        target = self._settings.frame_bytes
        carry = b""

        while True:
            frame = await self._audio_queue.get()
            if frame is None:
                break
            if not frame:
                continue

            carry += frame
            while len(carry) >= target:
                yield carry[:target]
                carry = carry[target:]

        # Never send a short trailing frame; drop it rather than risk error 3007.
        if carry:
            logger.debug("Dropped %d byte trailing audio frame", len(carry))

    # -------------------------------------------------- AssemblyAI handlers
    async def _handle_partial(
        self, text: str, turn_order: int | None, confidence: float | None
    ) -> None:
        subtitle = self.subtitles.upsert_partial(text, turn_order=turn_order, confidence=confidence)
        await self._emit(
            EventType.SUBTITLE,
            text,
            subtitle=subtitle,
            payload={"partial": True},
        )

    async def _handle_final(
        self, text: str, turn_order: int | None, confidence: float | None
    ) -> None:
        self._turns_seen += 1
        subtitle = self.subtitles.finalize(text, turn_order=turn_order, end_time=None)
        if subtitle is not None:
            subtitle.confidence = confidence
        await self._emit(
            EventType.SUBTITLE,
            text,
            subtitle=subtitle,
            payload={"partial": False},
        )
        # A finished turn is the most natural moment to evaluate a spoken
        # command, so try it before the next sentence begins.
        await self._maybe_run_command(text)

    async def _handle_speech_started(self) -> None:
        self.status = "listening"
        await self._emit(EventType.STATUS, "Listening...", payload={"speaking": True})

    async def _handle_status(self, status: str, message: str) -> None:
        self.status = status
        await self._emit(EventType.STATUS, message)

    async def _handle_error(self, message: str, fatal: bool) -> None:
        self.status = "error" if fatal else self.status
        await self._emit(EventType.ERROR, message, payload={"fatal": fatal})
        if fatal:
            self._running = False

    # ------------------------------------------------------------- commands
    async def handle_command(self, raw: str) -> CorrectionResult:
        """Run a spoken (or typed) command. Always returns a result."""
        command = parse_command(raw)
        result = self.corrections.correct(command)
        await self._publish_result(command, result)
        return result

    async def _maybe_run_command(self, turn_text: str) -> None:
        """Treat a finalized turn as a command when it clearly is one."""
        lowered = turn_text.strip().lower()
        if not any(prefix in lowered for _kind, prefixes in _COMMANDS for prefix in prefixes):
            return

        command = parse_command(turn_text)
        if not command.is_supported:
            # Only report "unsupported" when it really looked like a command.
            return

        # A command is an instruction, not a subtitle, so it must not become a
        # transcript line. The subtitle event has already gone out by now, so
        # tell the browser to drop the row too - otherwise the command lingers
        # on screen forever even though it is gone from state.
        last = self.subtitles.last()
        if last is not None and last.text.strip() == turn_text.strip():
            self.subtitles.remove(last.id)
            await self._emit(
                EventType.SUBTITLE_REMOVED,
                "",
                subtitle=last,
                payload={"reason": "voice_command"},
            )

        result = self.corrections.correct(command)
        await self._publish_result(command, result)

    async def _publish_result(self, command: ParsedCommand, result: CorrectionResult) -> None:
        if result.outcome is CorrectionOutcome.APPLIED:
            self._correction_count += 1

        # Vocabulary changes must reach both the UI and AssemblyAI.
        if result.action is CorrectionAction.ADD_VOCABULARY and result.vocabulary_term:
            await self._emit_context()
            pushed = await self._assemblyai.update_keyterms(self.context.keyterms())
            await self._emit(
                EventType.ACTIVITY,
                (
                    f'AssemblyAI keyterms updated with "{result.vocabulary_term}"'
                    if pushed
                    else f'"{result.vocabulary_term}" will apply to the next session'
                ),
            )

        await self._emit(
            EventType.CORRECTION,
            result.message,
            subtitle=self.subtitles.find(result.subtitle_id) if result.subtitle_id else None,
            payload=result.model_dump(mode="json"),
        )

    # ---------------------------------------------------------------- output
    async def emit_error(self, message: str) -> None:
        """Report a user-facing, non-fatal error."""
        await self._emit(EventType.ERROR, message, payload={"fatal": False})

    async def add_term(self, term: str) -> CorrectionResult:
        """Add a vocabulary term from the UI (same path as the voice command)."""
        command = ParsedCommand(kind=CommandKind.REMEMBER, raw=f"Remember {term}", find=term)
        result = self.corrections.remember(command)
        await self._publish_result(command, result)
        return result

    async def broadcast_transcript(self) -> None:
        """Re-send every subtitle, so a reconnecting client resyncs."""
        for subtitle in self.subtitles:
            await self._emit(
                EventType.SUBTITLE, subtitle.text, subtitle=subtitle, payload={"partial": False}
            )
        await self._emit_context()

    async def _emit_context(self) -> None:
        await self._emit(
            EventType.CONTEXT,
            f"{len(self.context.terms)} term(s) in project context.",
            payload={
                "vocabulary": self.context.terms,
                "project_name": self.context.context.project_name,
            },
        )

    async def _emit(
        self,
        event_type: EventType,
        message: str = "",
        *,
        subtitle: Subtitle | None = None,
        payload: dict | None = None,
    ) -> None:
        event = ServerEvent(
            type=event_type,
            message=message,
            subtitle=subtitle,
            payload=payload or {},
        )
        try:
            await self._send(event.model_dump(mode="json"))
        except Exception as exc:
            logger.info("Could not send %s event: %s", event_type.value, type(exc).__name__)

    # --------------------------------------------------------------- errors
    @staticmethod
    def _friendly_connect_error(exc: Exception) -> str:
        """Turn an AssemblyAI connection failure into actionable text.

        Never interpolates credentials or a raw SDK payload.
        """
        text = str(exc).lower()
        if "api key" in text or "unauthorized" in text or "401" in text:
            return (
                "AssemblyAI rejected the API key. Check ASSEMBLYAI_API_KEY in your "
                ".env file and restart Scriptora."
            )
        if "connect" in text or "timed out" in text or "timeout" in text:
            return (
                "Could not reach AssemblyAI's streaming endpoint. Check your network "
                "connection, then try Start Listening again."
            )
        return (
            "Could not start the AssemblyAI streaming session. See the server log for "
            f"details ({type(exc).__name__})."
        )
