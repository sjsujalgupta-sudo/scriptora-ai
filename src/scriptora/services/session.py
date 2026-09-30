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
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from ..config import Settings
from ..models.correction import CorrectionAction, CorrectionOutcome, CorrectionResult
from ..models.events import EventType, ServerEvent
from ..models.subtitle import Subtitle
from .assemblyai_service import AssemblyAIRealtimeService
from .command_service import (
    CommandKind,
    ParsedCommand,
    names_a_line_to_fix,
    needs_model,
    parse_command,
)
from .context_service import ContextService
from .correction_service import CorrectionService
from .intent_service import IntentInterpreter, command_from_intent, looks_like_repair
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
    (CommandKind.REPLACE, ("change ", "replace ", "swap ", "substitute ")),
    # Natural "change this / sentence 3 to <text>" phrasing. The REPLACE row
    # above already covers "change " and "replace "; these add the verbs that
    # only ever mean a whole-line rewrite, so a dictated "rewrite the third
    # sentence to ..." is recognised as a command at all.
    (
        CommandKind.SET_TEXT,
        ("rewrite ", "rewrite\u00a0", "make ", "set ", "put "),
    ),
)

# Politeness allowed *before* the command verb. Kept in step with the parser,
# which also tolerates exactly one leading "please", so the voice gate can never
# accept a phrase the parser then rejects.
_LEADING_POLITENESS = re.compile(r"^please\s+", re.IGNORECASE)


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
        # The repair interpreter (JOB 1) is only worth having when the same
        # gateway the corrector relies on is reachable. Without it, Stage B of
        # the voice gate never fires and natural repairs are simply left in the
        # transcript - which is honest, not an error.
        self._intent = IntentInterpreter(settings) if settings.llm_enabled else None

        self.status = "idle"
        # Bounded: see `send_audio`. Sized as ~1 s of audio at the configured
        # frame duration, which is enough to absorb WebSocket jitter without
        # letting latency run away on a slow link.
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=max(2, int(1000 / self._settings.frame_duration_ms) + 1)
        )
        self._dropped_frames = 0
        self._stopping = False
        self._stream_task: asyncio.Task[None] | None = None
        # Fire-and-forget emits that must outlive the callback that started them.
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._running = False
        self._turns_seen = 0
        self._correction_count = 0
        # Subtitle ids with a correction in flight, so a second command aimed at
        # the same line is refused instead of racing the first one.
        self._pending: set[str] = set()

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
        self._stream_task.add_done_callback(self._on_stream_finished)

    async def stop(self) -> None:
        """Stop listening and release the AssemblyAI session.

        Ordering matters. The sentinel is queued first so every buffered frame
        is still handed to AssemblyAI, then the stream is allowed to finish
        before the session is terminated. Terminating first, or cancelling the
        stream, discards the final words of the session.
        """
        if not self._running:
            return
        self._running = False
        self._stopping = True
        self.status = "stopping"
        try:
            await self._emit(EventType.STATUS, "Stopping...")

            # End the frame generator, let the stream task finish, then
            # terminate so AssemblyAI flushes the final turn and billing stops.
            await self._audio_queue.put(None)
            if self._stream_task is not None:
                try:
                    await asyncio.wait_for(self._stream_task, timeout=10.0)
                except TimeoutError:
                    logger.warning(
                        "Stream did not finish within 10s; %d frame(s) undrained",
                        self._audio_queue.qsize(),
                    )
                    self._stream_task.cancel()
                except Exception as exc:
                    logger.info("Audio stream ended with %s", type(exc).__name__)
                self._stream_task = None

            await self._assemblyai.disconnect()
            self.status = "idle"

            if self._dropped_frames:
                logger.info("Dropped %d audio frame(s) during this session", self._dropped_frames)

            await self._emit(
                EventType.SESSION_ENDED,
                f"Stopped. {self._turns_seen} turn(s), {self._correction_count} correction(s).",
                payload={
                    "turns": self._turns_seen,
                    "corrections": self._correction_count,
                    "subtitle_count": len(self.subtitles),
                },
            )
        finally:
            self._stopping = False

    # ----------------------------------------------------------------- audio
    async def send_audio(self, chunk: bytes) -> None:
        """Queue one audio frame from the browser.

        The queue is bounded and drops its oldest frame when full. An unbounded
        queue does not apply backpressure here - it just relocates the backlog,
        and the live transcript drifts further behind the speaker for the rest
        of the session. Dropping the oldest frame keeps latency near the cap:
        during live speech the newest audio is the only audio that still matters.
        """
        # Audio is accepted until the stop control arrives. The browser (see
        # app.js flushAndStop) places its final audio before the stop control,
        # and WebSocket delivery is ordered, so every frame that belongs to the
        # last sentence is already captured here - and drained by stop(). This
        # guard only drops frames that arrive after a genuine stop request.
        if not self._running:
            return
        try:
            self._audio_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            with suppress(asyncio.QueueEmpty):
                self._audio_queue.get_nowait()
            self._dropped_frames += 1
            logger.debug("Audio queue full; dropped a frame (%d total)", self._dropped_frames)
            with suppress(asyncio.QueueFull):
                self._audio_queue.put_nowait(chunk)

    async def _frame_generator(self) -> AsyncIterator[bytes]:
        """Yield AssemblyAI-sized frames pulled from the browser queue.

        Frames are resized to `frame_bytes`. The browser is asked to send exact
        frames, but jitter at the WebSocket layer can produce a short final
        frame, and AssemblyAI rejects anything outside 50-1000 ms with error
        3007. Normalising here removes that whole class of failure.

        The generator ends when `None` is queued, which is how `stop()` drains
        the session cleanly.

        Frames are handed over at exactly one per frame duration. The SDK's
        `stream()` pushes each frame into its own unbounded write queue as fast
        as this generator yields, so an unpaced generator lets the socket fall
        behind and then dumps the accumulated backlog as a burst on the way out.
        Real-time pacing keeps AssemblyAI receiving audio at 1x, which is what
        the server expects.
        """
        target = self._settings.frame_bytes
        frame_seconds = self._settings.frame_duration_ms / 1000
        carry = b""
        due = time.perf_counter()

        while True:
            frame = await self._audio_queue.get()
            if frame is None:
                break
            if not frame:
                continue

            carry += frame
            while len(carry) >= target:
                chunk = carry[:target]
                carry = carry[target:]
                yield chunk
                # Pace to wall-clock, not to a sleep per frame, so a late
                # frame does not permanently shift the cadence.
                due += frame_seconds
                delay = due - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)

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
        # A connection-closed error arriving while `stop()` is shutting the
        # session down is the expected consequence of closing it, not a
        # failure the user needs to act on. SESSION_ENDED already reports a
        # clean stop, and surfacing an error here left the UI stuck on "Error"
        # after a perfectly normal Stop.
        if self._stopping:
            logger.info("Connection closed during shutdown: %s", message)
            return
        self.status = "error" if fatal else self.status
        await self._emit(EventType.ERROR, message, payload={"fatal": fatal})
        if fatal:
            self._running = False

    def _on_stream_finished(self, task: asyncio.Task[None]) -> None:
        """Report a stream that died on its own, while the session looked live.

        A dropped connection or a suspended laptop ends the SDK task without
        ever reaching `_handle_error`. Without this the browser would keep
        showing "Listening" against a socket that is no longer recording, so
        the user would keep talking to a session that is already gone.
        """
        # A normal `stop()` drains and ends the task itself; that is not a fault.
        if not self._running or task.cancelled():
            return
        self._running = False
        self.status = "error"
        logger.warning(
            "AssemblyAI stream ended unexpectedly: %s",
            "error" if task.exception() else "clean exit",
        )
        # The callback cannot await, so the notification is scheduled. The task
        # always has a loop because it was created from one. Holding a reference
        # keeps it from being garbage collected mid-flight.
        notify = asyncio.create_task(
            self._emit(EventType.ERROR, self._friendly_stream_error(), payload={"fatal": True})
        )
        self._background_tasks.add(notify)
        notify.add_done_callback(self._background_tasks.discard)

    # ------------------------------------------------------------- commands
    async def handle_command(self, raw: str) -> CorrectionResult:
        """Run a spoken (or typed) command. Always returns a result."""
        return await self._run_command(parse_command(raw))

    async def _run_command(self, command: ParsedCommand) -> CorrectionResult:
        """Execute one command, announcing and guarding model-backed work.

        The gateway call takes real time, so two things have to be true while it
        is in flight: the UI must already show that something is happening, and
        a second command aimed at the same line must not race it. Both are keyed
        on the resolved subtitle id rather than the command text, so "correct the
        last subtitle" and "fix this" colliding on the same line is caught too.
        """
        if not command.is_supported:
            result = await self.corrections.correct(command)
            await self._publish_result(command, result)
            return result

        target = self.corrections.resolve(command)
        if target is not None and target.id in self._pending:
            result = CorrectionResult.failure(
                CorrectionOutcome.INVALID,
                f'"{target.text[:60]}" is still being corrected. Wait for it to finish.',
                backend="session",
                subtitle_id=target.id,
                reason=command.reason,
            )
            await self._publish_result(command, result)
            return result

        token = target.id if needs_model(command) and target is not None else None
        if token is not None:
            self._pending.add(token)
            await self._emit(
                EventType.CORRECTION_PENDING,
                f"Correcting {command.describe_target()}…",
                subtitle=target,
                payload={
                    "subtitle_id": target.id,
                    "text": target.text,
                    "target": command.describe_target(),
                    "command": command.raw,
                },
            )
        try:
            result = await self.corrections.correct(command)
        finally:
            if token is not None:
                self._pending.discard(token)

        await self._publish_result(command, result)
        return result

    async def _maybe_run_command(self, turn_text: str) -> None:
        """Treat a finalized turn as a command when it clearly is one."""
        # The gate is loose about wording but strict about shape: a command is an
        # instruction, so it *begins* with the verb. Matching the keyword
        # anywhere in the sentence made ordinary dictation ("I need to remember
        # to lock the door") look like a command, and quietly swallowing a
        # transcript line is far worse than the false negatives this avoids.
        # Only the leading "please" is tolerated, because that is the sole piece
        # of politeness the parser also accepts - so the gate can never fire on
        # something the parser would then reject as a command.
        lowered = _LEADING_POLITENESS.sub("", turn_text.strip().lower())
        if not any(
            lowered.startswith(prefix) for _kind, prefixes in _COMMANDS for prefix in prefixes
        ) and not names_a_line_to_fix(lowered):
            # Stage B: the turn did not open with a command verb, but it may
            # still be a repair in ordinary words ("No, I said Symphony."). Only
            # the interpreter may decide that - in ordinary speech it stays a
            # subtitle, in a confirmed repair it becomes a command.
            await self._maybe_interpret_repair(turn_text)
            return

        command = parse_command(turn_text)

        # A command is an instruction, not a subtitle, so it must not become a
        # transcript line. The subtitle event has already gone out by now, so
        # tell the browser to drop the row too - otherwise the command lingers
        # on screen forever even though it is gone from state.
        #
        # This runs for an unparseable command too, and deliberately so. Text
        # only reaches here once it has cleared the command gate above, so the
        # speaker was plainly trying to issue a command; an unparsed one is
        # still an instruction that was not carried out, and leaving it in the
        # transcript as though it were content is the worst possible reading of
        # it. `_run_command` answers UNSUPPORTED with a result and no model call,
        # so the user is told what happened instead of hearing nothing.
        await self._withdraw_turn_line(turn_text)

        await self._run_command(command)

    async def _maybe_interpret_repair(self, turn_text: str) -> None:
        """Stage B of the voice gate: ask the interpreter about a repair.

        Runs only when Stage A declined. The cheap cue check keeps ordinary
        dictation away from the gateway; the interpreter then has to confirm
        the turn refers to the transcript, or the subtitle is left alone. The
        utterance is withdrawn from the transcript only once a repair is
        confirmed, never before - an ambiguous turn costs nothing but the
        round trip.
        """
        if self._intent is None or not looks_like_repair(turn_text):
            return

        result = await self._intent.interpret(turn_text, self.subtitles, self.context)
        logger.debug("Stage B interpreted %r as %s", turn_text, result)
        if result is None or not result.is_repair:
            return
        if result.confidence < self._settings.intent_confidence_threshold:
            logger.debug(
                "Stage B declined %r: confidence %s below %s",
                turn_text,
                result.confidence,
                self._settings.intent_confidence_threshold,
            )
            return

        command = command_from_intent(result, turn_text)
        logger.debug("Stage B mapped %r to %r", turn_text, command)
        if command is None:
            return

        # Safety interlock: a replace needs a `find` that is actually in the
        # transcript. The interpreter is told this, but it is checked here too,
        # so a confident hallucination cannot withdraw a line and then fail.
        # Unlike Stage A - where the speaker clearly issued a command - a Stage
        # B miss means the turn was probably not a repair at all, so it is kept
        # as content rather than announced as an error.
        if command.kind is CommandKind.REPLACE and self.corrections.resolve(command) is None:
            logger.debug("Stage B kept %r: replacement find did not resolve", turn_text)
            return

        await self._withdraw_turn_line(turn_text)
        await self._run_command(command)

    async def _withdraw_turn_line(self, turn_text: str) -> None:
        """Drop the finalized turn from the transcript and the browser's view."""
        last = self.subtitles.last()
        if last is not None and last.text.strip() == turn_text.strip():
            self.subtitles.remove(last.id)
            await self._emit(
                EventType.SUBTITLE_REMOVED,
                "",
                subtitle=last,
                payload={"reason": "voice_command"},
            )

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

        if result.outcome is CorrectionOutcome.APPLIED and result.corrected_subtitle_id:
            # The corrected wording is a new line in the transcript, so the
            # client has to be sent it. The correction event alone carries only
            # the original, which would leave the screen showing the untouched
            # line with nothing under it.
            corrected = self.subtitles.find(result.corrected_subtitle_id)
            if corrected is not None:
                await self._emit(
                    EventType.SUBTITLE,
                    corrected.text,
                    subtitle=corrected,
                    payload={"partial": False},
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

    @staticmethod
    def _friendly_stream_error() -> str:
        """Explain a stream that ended mid-session, and how to recover.

        Deliberately not derived from the exception: by this point the text
        reaches the browser, so it must be safe and actionable rather than
        specific.
        """
        return (
            "Lost the connection to AssemblyAI while listening. Check your network, "
            "then press Start Listening again to reconnect."
        )
