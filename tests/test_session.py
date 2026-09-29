"""Session orchestration and the WebSocket transport.

AssemblyAI is stubbed at the service boundary, so the routing, event flow and
error handling are exercised without any network access.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from scriptora.config import Settings
from scriptora.main import create_app
from scriptora.models.correction import CorrectionOutcome
from scriptora.models.events import EventType
from scriptora.services.session import ScriptoraSession

FAKE_KEY = "test-key-not-a-credential"


class StubAssemblyAI:
    """Stands in for AssemblyAIRealtimeService.

    Records calls and lets a test drive Turn events by hand.
    """

    def __init__(self, connect_error: Exception | None = None) -> None:
        self.connect_error = connect_error
        self.connected = False
        self.keyterms: list[list[str]] = []
        self.streams: list[bytes] = []
        self.disconnected = False
        # When set, `stream()` waits on this instead of consuming frames, which
        # simulates a socket that has stopped draining.
        self.stall: asyncio.Event | None = None
        # Raised through the on_error callback when disconnecting, to model a
        # server that closes the socket as we terminate.
        self.error_on_disconnect: str | None = None
        self._handlers: dict = {}

    async def connect(
        self, keyterms, *, on_partial, on_final, on_speech_start, on_status, on_error
    ):
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True
        self._handlers = {
            "partial": on_partial,
            "final": on_final,
            "speech": on_speech_start,
            "status": on_status,
            "error": on_error,
        }
        await on_status("listening", "Connected to AssemblyAI. Start speaking.")

    async def stream(self, frames):
        """Consume the frame generator, mirroring the SDK's one-shot contract."""
        if self.stall is not None:
            await self.stall.wait()
        async for chunk in frames:
            self.streams.append(chunk)

    async def update_keyterms(self, keyterms: list[str]) -> bool:
        self.keyterms.append(list(keyterms))
        return True

    async def disconnect(self) -> None:
        self.disconnected = True
        self.connected = False
        if self.error_on_disconnect is not None:
            await self._handlers["error"](self.error_on_disconnect, True)

    # Test drivers
    async def emit_partial(self, text, turn_order=None, confidence=None):
        await self._handlers["partial"](text, turn_order, confidence)

    async def emit_final(self, text, turn_order=None, confidence=None):
        await self._handlers["final"](text, turn_order, confidence)


@pytest.fixture
def stub(monkeypatch):
    """Install StubAssemblyAI for every session built during the test."""
    created: list[StubAssemblyAI] = []
    error: list[Exception | None] = [None]

    def factory(settings):
        stub_instance = StubAssemblyAI(connect_error=error[0])
        created.append(stub_instance)
        return stub_instance

    monkeypatch.setattr("scriptora.services.session.AssemblyAIRealtimeService", factory)

    class Control:
        @property
        def all(self):
            return created

        @property
        def last(self):
            return created[-1]

        def fail_connect_with(self, exc):
            error[0] = exc

    return Control()


def make_session(settings=None, sink=None):
    events: list[dict] = []

    async def send(event: dict) -> None:
        events.append(event)

    return ScriptoraSession(settings or Settings(assemblyai_api_key=FAKE_KEY), send), events


def events_of(events, event_type: EventType):
    return [e for e in events if e["type"] == event_type.value]


# ============================================================== lifecycle
async def test_start_opens_a_session(stub):
    session, events = make_session()
    await session.start()

    assert stub.last.connected is True
    assert session.status == "listening"
    assert len(events_of(events, EventType.SESSION_STARTED)) == 1


async def test_stop_disconnects(stub):
    session, events = make_session()
    await session.start()
    await session.stop()

    assert stub.last.disconnected is True
    assert len(events_of(events, EventType.SESSION_ENDED)) == 1


async def test_start_twice_is_a_no_op(stub):
    session, _ = make_session()
    await session.start()
    await session.start()

    assert len(stub.all) == 1


async def test_stop_without_start_is_a_no_op(stub):
    session, _ = make_session()
    await session.stop()  # must not raise


# ============================================================ transcript
async def test_partial_then_final_produces_one_line(stub):
    session, events = make_session()
    await session.start()

    await stub.last.emit_partial("today we are")
    await stub.last.emit_partial("today we are building")
    await stub.last.emit_final("Today we are building the backend.")

    assert len(session.subtitles) == 1
    assert session.subtitles.last().status.value == "final"
    assert len(events_of(events, EventType.SUBTITLE)) == 3


async def test_empty_transcripts_produce_nothing(stub):
    session, _ = make_session()
    await session.start()
    await stub.last.emit_final("")

    assert len(session.subtitles) == 0


# =============================================================== commands
def corrected_text(session, original_id: str) -> str:
    """The corrected wording, which lives in a child line after the original.

    A correction never overwrites what AssemblyAI heard, so assertions about
    the *new* text have to read the child, and assertions that the original
    survived read the parent. Mixing the two up is the easiest way to write a
    test that passes for the wrong reason.
    """
    children = session.subtitles.corrections_of(original_id)
    assert len(children) == 1, f"expected exactly one correction, got {len(children)}"
    return children[0].text


async def test_voice_command_is_not_kept_as_a_subtitle(stub):
    """A spoken instruction must not also appear in the transcript."""
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("Today we are building the backend using fast API.")
    sub.finalize()

    await stub.last.emit_final("Correct the last subtitle. It's FastAPI.")

    # The command text is gone; what remains is the line that was spoken plus
    # the correction inserted after it.
    texts = [s.text for s in session.subtitles]
    assert texts == [
        "Today we are building the backend using fast API.",
        "Today we are building the backend using FastAPI.",
    ]
    assert len(events_of(events, EventType.CORRECTION)) == 1


async def test_voice_command_emits_a_removal_so_the_ui_drops_the_row(stub):
    """The command reaches the transcript before we recognise it as one.

    Without an explicit removal event the browser keeps rendering a line the
    server has already deleted, leaving a ghost instruction in the transcript.
    """
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("Today we are building the backend using fast API.")
    sub.finalize()

    await stub.last.emit_final("Correct the last subtitle. It's FastAPI.")

    removals = events_of(events, EventType.SUBTITLE_REMOVED)
    assert removals, "the browser must be told to drop the command line"
    assert removals[-1]["payload"]["reason"] == "voice_command"

    # The row that is withdrawn is the command's own line, and it is gone from
    # server state too. The real subtitle survives.
    withdrawn = removals[-1]["subtitle"]["id"]
    assert "Correct the last subtitle" in removals[-1]["subtitle"]["text"]
    assert withdrawn not in {s.id for s in session.subtitles}
    assert sub.id in {s.id for s in session.subtitles}
    assert corrected_text(session, sub.id).endswith("using FastAPI.")

    """A bare "correct the last subtitle" works off project context alone.

    AssemblyAI is seeded, so hearing "assembly AI" is a known spelling error
    that needs no term named in the command.
    """
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("We deploy on assembly AI every day.")
    sub.finalize()

    await stub.last.emit_final("Correct the last subtitle.")

    assert corrected_text(session, sub.id) == "We deploy on AssemblyAI every day."
    assert session.subtitles.get(sub.id).text == "We deploy on assembly AI every day."
    assert len(events_of(events, EventType.CORRECTION)) == 1


async def test_named_term_in_the_command_corrects_before_being_remembered(stub):
    """The primary demo: "it's FastAPI" fixes the line with no prior context."""
    session, _ = make_session()
    await session.start()
    sub = session.subtitles.add("Today we are building the backend using fast API.")
    sub.finalize()
    assert "FastAPI" not in session.context.terms

    await stub.last.emit_final("Correct the last subtitle. It's FastAPI.")

    assert corrected_text(session, sub.id).endswith("using FastAPI.")


async def test_remember_command_updates_vocabulary_and_assemblyai(stub):
    """The headline demo: memory must reach both the UI and AssemblyAI."""
    session, events = make_session()
    await session.start()

    await stub.last.emit_final("Remember FastAPI as a technical term.")

    assert "FastAPI" in session.context.terms
    # The vocabulary is pushed into the live AssemblyAI session.
    assert stub.last.keyterms
    assert "FastAPI" in stub.last.keyterms[-1]
    context_events = events_of(events, EventType.CONTEXT)
    assert context_events, "vocabulary change must be broadcast"
    assert "FastAPI" in context_events[-1]["payload"]["vocabulary"]


async def test_ordinary_dictation_is_not_treated_as_a_command(stub):
    session, _ = make_session()
    await session.start()

    await stub.last.emit_final("I need to remember to lock the door before I leave.")

    # It stays a transcript line and no correction was attempted.
    assert len(session.subtitles) == 1
    assert session.subtitles.last().text.startswith("I need to remember")


# ================================================================== audio
async def test_audio_is_forwarded_to_assemblyai(stub):
    session, _ = make_session()
    await session.start()

    # One whole frame: 100 ms of 16 kHz mono s16le = 3200 bytes.
    await session.send_audio(b"\x01\x02" * 1600)
    await asyncio_sleep()

    assert len(stub.last.streams) == 1
    assert len(stub.last.streams[0]) == 3200


async def test_short_frames_are_buffered_until_valid(stub):
    """A partial frame must not be forwarded - AssemblyAI rejects <50 ms."""
    session, _ = make_session()
    await session.start()

    # Half a frame (1600 bytes at 100 ms / 16 kHz).
    await session.send_audio(b"\x01\x02" * 800)
    await asyncio_sleep()
    assert stub.last.streams == []

    # Completing the frame makes exactly one valid frame.
    await session.send_audio(b"\x01\x02" * 800)
    await asyncio_sleep()
    assert len(stub.last.streams) == 1


# ====================================================== audio backpressure
def _frame(settings: Settings) -> bytes:
    """One frame of exactly the size the session will forward."""
    return b"\x00" * settings.frame_bytes


def _fast_settings() -> Settings:
    """Settings with a 1 ms frame so pacing is exercised but tests stay quick."""
    return Settings(assemblyai_api_key=FAKE_KEY, frame_duration_ms=1)


async def test_audio_queue_is_bounded_when_the_socket_stalls(stub):
    """A stalled socket must not grow the backlog without limit.

    The browser has no way to slow down, so an unbounded queue just relocates
    the lag: the transcript falls further behind the speaker for the rest of the
    session. Frames have to be dropped instead.
    """
    session, _ = make_session(_fast_settings())
    settings = _fast_settings()
    await session.start()
    # The stub never drains, exactly like a socket that stopped reading.
    stub.last.stall = asyncio.Event()

    for _ in range(2000):
        await session.send_audio(_frame(settings))

    depth = session._audio_queue.qsize()
    assert depth <= session._audio_queue.maxsize, f"queue grew unbounded: {depth}"
    assert session._dropped_frames > 0, "frames should be dropped once the queue is full"

    stub.last.stall.set()
    await session.stop()


async def test_audio_is_fed_to_assemblyai_at_real_time(stub):
    """Frames must reach AssemblyAI at 1x, not as an unpaced firehose.

    `stream()` pushes whatever the generator yields into an unbounded write
    queue. Without pacing here, audio outpacing the socket is buffered and then
    flushed as a burst, which is what a rate-limited server rejects.
    """
    import time

    frame_ms = 50
    frames = 10
    settings = Settings(assemblyai_api_key=FAKE_KEY, frame_duration_ms=frame_ms)
    session, _ = make_session(settings)
    await session.start()

    # Offer the audio all at once, as a backlog would arrive.
    for _ in range(frames):
        await session.send_audio(_frame(settings))

    started = time.perf_counter()
    await session.stop()
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert len(stub.last.streams) == frames
    # Pacing means the last frame can only leave after (frames - 1) durations.
    # Without it every frame leaves immediately and this is ~0 ms.
    assert elapsed_ms >= (frames - 1) * frame_ms * 0.8, (
        f"frames were flushed in {elapsed_ms:.0f} ms; expected real-time pacing"
    )


async def test_stop_flushes_buffered_audio(stub):
    """Whatever was queued must still be handed over when Stop is pressed.

    Cancelling the stream before the queue drains would silently discard the
    last words of the session.
    """
    session, _ = make_session(_fast_settings())
    settings = _fast_settings()
    await session.start()
    for _ in range(20):
        await session.send_audio(_frame(settings))
    await session.stop()
    assert len(stub.last.streams) == 20


async def test_a_stalled_socket_still_stops_cleanly(stub):
    """Stop must not hang forever when the consumer is not reading."""
    settings = _fast_settings()
    session, events = make_session(settings)
    await session.start()
    stub.last.stall = asyncio.Event()
    for _ in range(50):
        await session.send_audio(_frame(settings))

    stopper = asyncio.create_task(session.stop())
    await asyncio.wait_for(stopper, timeout=20)
    assert events_of(events, EventType.SESSION_ENDED), "Stop never completed"


async def test_error_raised_while_stopping_is_not_reported_to_the_user(stub):
    """A socket closing as we terminate it is expected, not a failure.

    Reporting it left the UI stuck on "Error" after a completely normal Stop.
    """
    session, events = make_session(_fast_settings())
    await session.start()
    stub.last.error_on_disconnect = "The AssemblyAI connection was dropped unexpectedly."
    await session.stop()

    assert events_of(events, EventType.ERROR) == []
    assert events_of(events, EventType.SESSION_ENDED), "the clean stop must still be reported"
    assert session.status == "idle"


async def test_error_outside_stopping_is_still_reported(stub):
    """Suppression must not swallow genuine mid-session failures."""
    message = "The AssemblyAI connection was dropped unexpectedly."
    session, events = make_session(_fast_settings())
    await session.start()
    await stub.last._handlers["error"](message, True)

    errors = events_of(events, EventType.ERROR)
    assert errors, "a real mid-session failure must still reach the browser"
    assert errors[0]["message"] == message
    assert errors[0]["payload"]["fatal"] is True
    assert session.status == "error"


async def test_long_session_does_not_accumulate_tasks(stub):
    """Many turns must not leave background tasks or duplicate subtitles behind."""
    session, _ = make_session(_fast_settings())
    await session.start()
    for index in range(200):
        await stub.last.emit_partial(f"partial {index}", turn_order=index)
        await stub.last.emit_final(f"final {index}", turn_order=index)

    assert not session._background_tasks, "background emits piled up"
    # One line per turn: partials update in place rather than appending.
    assert len(session.subtitles) == 200
    await session.stop()
    assert not session._background_tasks


# ================================================================= errors
async def test_connect_failure_reports_a_friendly_error(stub):
    stub.fail_connect_with(RuntimeError("unauthorized api key"))
    session, events = make_session()
    await session.start()

    errors = events_of(events, EventType.ERROR)
    assert errors
    assert "API key" in errors[0]["message"]
    assert errors[0]["payload"]["fatal"] is True
    assert session.running is False


async def test_connect_failure_never_leaks_the_key(stub):
    stub.fail_connect_with(RuntimeError(f"failed for key {FAKE_KEY}"))
    session, events = make_session()
    await session.start()

    serialised = json.dumps(events)
    assert FAKE_KEY not in serialised


async def test_stream_dying_mid_session_tells_the_user(stub):
    """A dropped connection must not leave the UI claiming it is still live.

    When the socket dies the SDK task raises on its own, which never reaches
    `_handle_error`, so the browser would otherwise sit on "Listening" against
    a session that is already gone.
    """
    session, events = make_session()

    async def dying_stream(frames):
        raise ConnectionError("connection reset by peer")

    await session.start()
    assert session.running is True
    session._assemblyai.stream = dying_stream

    # Restart the stream with the failing implementation, as a real drop would.
    session._stream_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await session._stream_task
    session._stream_task = asyncio.create_task(dying_stream(session._frame_generator()))
    session._stream_task.add_done_callback(session._on_stream_finished)
    # The callback cannot await, so it schedules the emit; let the loop run it.
    await asyncio.sleep(0.05)

    errors = events_of(events, EventType.ERROR)
    assert errors, "a dead stream must be reported to the browser"
    assert errors[0]["payload"]["fatal"] is True
    assert "Start Listening again" in errors[0]["message"]
    assert session.running is False
    assert FAKE_KEY not in json.dumps(events)


async def test_stream_dying_does_not_report_anything_during_a_normal_stop(stub):
    """`stop()` ends the task deliberately; that must stay a clean shutdown."""
    session, events = make_session()
    await session.start()
    await session.stop()
    await asyncio.sleep(0)

    assert events_of(events, EventType.ERROR) == []


# ============================================================== transport
def test_websocket_hello_frame_has_no_credentials(stub):
    app = create_app(Settings(assemblyai_api_key=FAKE_KEY))
    with TestClient(app) as client, client.websocket_connect("/ws/audio") as ws:
        hello = ws.receive_json()

    assert hello["type"] == "hello"
    assert FAKE_KEY not in json.dumps(hello)
    assert hello["payload"]["config"]["api_key_configured"] is True


def test_websocket_without_a_key_closes_with_a_clear_message(stub):
    app = create_app(Settings(assemblyai_api_key=None))
    with TestClient(app) as client, client.websocket_connect("/ws/audio") as ws:
        message = ws.receive_json()

    assert message["type"] == "error"
    assert "ASSEMBLYAI_API_KEY" in message["message"]


def test_health_endpoint_is_credential_free(stub):
    app = create_app(Settings(assemblyai_api_key=FAKE_KEY))
    with TestClient(app) as client:
        body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert FAKE_KEY not in json.dumps(body)


def test_index_page_renders(stub):
    app = create_app(Settings(assemblyai_api_key=FAKE_KEY))
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "SCRIPTORA" in response.text
    assert FAKE_KEY not in response.text


def test_typed_command_over_the_socket_reaches_the_corrector(stub):
    app = create_app(Settings(assemblyai_api_key=FAKE_KEY, corrector_backend="rules"))
    with TestClient(app) as client, client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()  # hello
        ws.send_json({"action": "start"})
        # session_started, context, status
        for _ in range(3):
            ws.receive_json()

        ws.send_json({"action": "add_term", "term": "Kubernetes"})
        seen_add = None
        for _ in range(6):
            event = ws.receive_json()
            if event["type"] == "correction":
                seen_add = event
                break

    assert seen_add is not None
    assert seen_add["payload"]["outcome"] == "applied"
    assert "Kubernetes" in seen_add["payload"]["vocabulary_term"]


def test_a_typed_natural_command_works_over_the_socket(stub):
    """The demo phrase, through the real transport rather than the service.

    With no spoken lines the command cannot succeed, which is the point: the
    reply proves it was *parsed* as a whole-line rewrite naming "this", rather
    than falling through to find-and-replace. Populating a transcript needs a
    microphone, and is covered at the service level instead.
    """
    app = create_app(Settings(assemblyai_api_key=FAKE_KEY, corrector_backend="rules"))
    with TestClient(app) as client, client.websocket_connect("/ws/audio") as ws:
        ws.receive_json()  # hello
        ws.send_json({"action": "command", "text": 'Change this to "To deployed."'})

        seen = None
        for _ in range(8):
            event = ws.receive_json()
            if event["type"] == "correction":
                seen = event
                break

    assert seen is not None, "no correction event arrived for the typed command"
    assert seen["payload"]["outcome"] == "invalid"
    assert "no subtitles" in seen["payload"]["message"].lower()
    # The reason is the parser's own, so it proves SET_TEXT matched and named
    # the target. A find-and-replace would have reported a different reason.
    assert seen["payload"]["reason"] == 'Setting this sentence to "To deployed."'


def test_the_typed_command_route_does_not_block_the_socket_read_loop(stub):
    """A slow correction must not stall the loop that drains audio frames.

    `handle_command` is awaited on the same receive loop that reads the
    browser's audio, so awaiting it inline would hold every subsequent frame
    for the length of the gateway round trip. The bounded queue downstream
    would then absorb that backlog as drops, which looks exactly like the
    audio bug this app already had.
    """
    app = create_app(Settings(assemblyai_api_key=FAKE_KEY, corrector_backend="rules"))
    captured: list = []

    class BlockingSession(ScriptoraSession):
        async def handle_command(self, raw):
            captured.append(raw)
            # Yield control the way a real gateway await does, long enough for
            # the receive loop to come back around for more frames.
            for _ in range(20):
                await asyncio.sleep(0)
            return await super().handle_command(raw)

    with (
        patch.object(ScriptoraSession, "handle_command", BlockingSession.handle_command),
        TestClient(app) as client,
        client.websocket_connect("/ws/audio") as ws,
    ):
        ws.receive_json()  # hello
        ws.send_json({"action": "command", "text": 'Change this to "X"'})
        # A frame sent *after* the command must still be received while the
        # correction is outstanding; if the loop were blocked it would wait.
        ws.send_bytes(b"\x01\x02" * 1600)
        ws.send_json({"action": "add_term", "term": "Zzz"})

        types = []
        for _ in range(14):
            try:
                event = ws.receive_json()
            except Exception:
                break
            types.append(event["type"])
            if "correction" in types and "activity" in types:
                break

    assert captured == ['Change this to "X"'], "the command was not dispatched"
    # The term round trip completing at all proves the loop kept reading.
    assert "activity" in types, f"socket stalled behind the correction: {types}"


async def asyncio_sleep():
    """Let the audio pump task drain the queue."""
    import asyncio

    await asyncio.sleep(0.01)


# ============================================ correction through the async path
def _gateway_response(content: str, status: int = 200) -> httpx.Response:
    """A gateway envelope around raw model output, for no_action replies."""
    return httpx.Response(
        status,
        json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]},
    )


def _gateway_says(subtitle_id: str, replacement: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"action": "correct_subtitle", "target_subtitle_id": "'
                            + subtitle_id
                            + '", "replacement_text": "'
                            + replacement
                            + '", "vocabulary_term": null, "reason": "canonical spelling"}'
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    )


async def test_a_typed_command_still_corrects_the_intended_subtitle(stub):
    """The HTTP command route must survive the async conversion intact.

    `handle_command` is the path the /command endpoint awaits, so this covers
    the real entry point rather than the service in isolation: the awaited
    correction still lands on the intended subtitle, and the UI still receives
    exactly one correction event carrying the same fields as before.
    """
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("we use fast api here")
    sub.finalize()

    with patch(
        "httpx.AsyncClient.post", return_value=_gateway_says(sub.id, "we use FastAPI here")
    ) as post:
        result = await session.handle_command("Correct the last subtitle")

    assert post.call_count == 1
    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.backend == "llm"
    assert result.subtitle_id == sub.id
    assert corrected_text(session, sub.id) == "we use FastAPI here"
    assert session.subtitles.get(sub.id).text == "we use fast api here"

    corrections = events_of(events, EventType.CORRECTION)
    assert len(corrections) == 1
    assert corrections[0]["payload"]["outcome"] == "applied"
    assert corrections[0]["payload"]["backend"] == "llm"
    assert corrections[0]["payload"]["subtitle_id"] == sub.id


# ==================== typed and spoken commands reach the same correction
# The live demo tried "Correct the last subtitle" both ways and neither fixed
# anything. Both routes converge on `corrections.correct()`, so these assert the
# shared behaviour is actually shared rather than assuming it.


async def _correct_quen(stub, *, spoken: bool) -> tuple:
    """Run the reported command one way and return the session and subtitle."""
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("Quen.")
    sub.finalize()

    # "Qwen" is deliberately absent from the project context.
    assert "Qwen" not in session.context.terms

    with patch("httpx.AsyncClient.post", return_value=_gateway_says(sub.id, "Qwen.")) as post:
        if spoken:
            await session._maybe_run_command("Correct the last subtitle")
            result = None
        else:
            result = await session.handle_command("Correct the last subtitle")
        assert post.call_count == 1, "the model must actually be consulted"

    corrections = events_of(events, EventType.CORRECTION)
    assert len(corrections) == 1
    return session, sub, result, corrections[0]["payload"]


async def test_the_typed_command_repairs_a_misheard_term(stub):
    session, sub, result, payload = await _correct_quen(stub, spoken=False)

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.backend == "llm"
    assert corrected_text(session, sub.id) == "Qwen."
    assert session.subtitles.get(sub.id).text == "Quen."
    assert payload["outcome"] == "applied"
    assert payload["after"] == "Qwen."


async def test_the_spoken_command_repairs_the_same_misheard_term(stub):
    session, sub, _result, payload = await _correct_quen(stub, spoken=True)

    assert corrected_text(session, sub.id) == "Qwen."
    assert session.subtitles.get(sub.id).text == "Quen."
    assert payload["outcome"] == "applied"
    assert payload["after"] == "Qwen."


async def test_a_declined_correction_tells_the_browser_why(stub):
    """The browser must receive the reason, not just "No correction was needed."

    Without this the UI shows a no-op with no explanation, which is what made
    the live failure look like the command was ignored.
    """
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("Quen.")
    sub.finalize()

    declined = (
        '{"action": "no_action", "target_subtitle_id": null, "replacement_text": null, '
        '"vocabulary_term": null, "reason": "no confident correction for this word"}'
    )
    with patch(
        "httpx.AsyncClient.post",
        return_value=_gateway_response(declined),
    ):
        await session.handle_command("Correct the last subtitle")

    payload = events_of(events, EventType.CORRECTION)[0]["payload"]
    assert payload["outcome"] == "no_action"
    assert payload["reason"] == "no confident correction for this word"
    assert session.subtitles.get(sub.id).text == "Quen."
    assert session.subtitles.corrections_of(sub.id) == [], (
        "a declined correction must insert nothing"
    )


async def test_a_spoken_command_does_not_block_audio_forwarding_while_pending(stub):
    """A pending LLM answer must not stop audio reaching AssemblyAI.

    The gateway parks until the test releases it, so the overlap between an
    in-flight correction and a forwarded audio frame is a fact, not a race. If
    the correction blocked the loop, the frame could not be delivered until the
    gateway replied - and the gateway never replies until released.
    """
    import asyncio

    session, _ = make_session()
    await session.start()
    sub = session.subtitles.add("we use fast api here")
    sub.finalize()

    request_started = asyncio.Event()
    release = asyncio.Event()

    async def parked_post(*_args, **_kwargs):
        request_started.set()
        await release.wait()
        return _gateway_says(sub.id, "we use FastAPI here")

    with patch("httpx.AsyncClient.post", new=parked_post):
        pending = asyncio.create_task(session.handle_command("Correct the last subtitle"))
        await asyncio.wait_for(request_started.wait(), timeout=5)

        # Audio keeps flowing while the correction is still outstanding.
        # One whole frame: 100 ms of 16 kHz mono s16le = 3200 bytes.
        await asyncio.wait_for(session.send_audio(b"\x01\x02" * 1600), timeout=5)
        await asyncio_sleep()
        assert stub.last.streams, "no audio reached AssemblyAI while the LLM was pending"
        assert not pending.done(), "correction completed before the gateway replied"

        release.set()
        result = await asyncio.wait_for(pending, timeout=5)

    assert result.outcome is CorrectionOutcome.APPLIED
    assert corrected_text(session, sub.id) == "we use FastAPI here"
    assert session.subtitles.get(sub.id).text == "we use fast api here"


# ============================================== pending / duplicate guard
def _parked_gateway(target_id: str, replacement: str):
    """A gateway that parks its *first* request until released.

    Only the first request blocks. Later ones answer immediately, so a test can
    prove that a second command is genuinely free to proceed while the first is
    still outstanding - parking every call would just deadlock the test.
    """
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[int] = []

    async def post(_self, *_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            started.set()
            await release.wait()
        return _gateway_says(target_id, replacement)

    return post, started, release


async def test_a_pending_event_is_sent_before_the_gateway_is_called(stub):
    """The user must see something happen before the slow part starts.

    Emitted after the fact, the spinner would appear together with the answer
    and convey nothing; the whole point is to cover the wait.
    """
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("Quen.")
    sub.finalize()

    post, started, release = _parked_gateway(sub.id, "Qwen.")
    with patch("httpx.AsyncClient.post", new=post):
        pending = asyncio.create_task(session.handle_command("Correct the last subtitle"))
        await asyncio.wait_for(started.wait(), timeout=5)

        # The gateway is mid-request and the pending event has already landed.
        assert not pending.done()
        announced = events_of(events, EventType.CORRECTION_PENDING)
        assert len(announced) == 1
        assert announced[0]["payload"]["subtitle_id"] == sub.id
        assert announced[0]["payload"]["text"] == "Quen."
        assert "the last sentence" in announced[0]["payload"]["target"]
        assert events_of(events, EventType.CORRECTION) == []

        release.set()
        await asyncio.wait_for(pending, timeout=5)

    assert len(events_of(events, EventType.CORRECTION)) == 1


async def test_a_deterministic_command_sends_no_pending_event(stub):
    """A pending state for an instant command is just a flash of noise."""
    session, events = make_session()
    await session.start()
    session.subtitles.add("we should changed the config").finalize()

    await session.handle_command('Change the last sentence to "To deployed."')

    assert events_of(events, EventType.CORRECTION_PENDING) == []
    assert len(events_of(events, EventType.CORRECTION)) == 1


async def test_a_second_command_on_a_pending_line_is_refused(stub):
    """Two corrections racing on one line must not both be applied.

    The second would resolve the same subtitle and overwrite or duplicate the
    first one's result, so it is refused while the first is in flight. The rule
    is keyed on the resolved line, not the wording, so "correct the last
    subtitle" and "fix this" collide correctly.
    """
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("Quen.")
    sub.finalize()

    post, started, release = _parked_gateway(sub.id, "Qwen.")
    with patch("httpx.AsyncClient.post", new=post):
        first = asyncio.create_task(session.handle_command("Correct the last subtitle"))
        await asyncio.wait_for(started.wait(), timeout=5)

        # Same line, different wording - still the same target.
        second = await session.handle_command("Correct the last subtitle")
        assert second.outcome is CorrectionOutcome.INVALID
        assert "still being corrected" in second.message

        release.set()
        await asyncio.wait_for(first, timeout=5)

    # Exactly one correction landed, and it is the one that got there first.
    # The refusal is also reported, so the user is told why nothing happened.
    applied = [
        e for e in events_of(events, EventType.CORRECTION) if e["payload"]["outcome"] == "applied"
    ]
    assert len(applied) == 1
    assert len(session.subtitles.corrections_of(sub.id)) == 1
    assert "still being corrected" in events_of(events, EventType.CORRECTION)[0]["message"]


async def test_the_pending_guard_is_released_when_the_correction_ends(stub):
    """A refusal that never clears would strand the line permanently."""
    session, _events = make_session()
    await session.start()
    sub = session.subtitles.add("Quen.")
    sub.finalize()

    with patch("httpx.AsyncClient.post", return_value=_gateway_says(sub.id, "Qwen.")):
        await session.handle_command("Correct the last subtitle")

    assert session._pending == set()
    # And the same line is immediately correctable again.
    with patch("httpx.AsyncClient.post", return_value=_gateway_says(sub.id, "Qwen2.")):
        again = await session.handle_command("Correct the last subtitle")
    assert again.outcome is CorrectionOutcome.APPLIED


async def test_a_pending_guard_does_not_block_a_different_line(stub):
    """The guard is per line. One slow correction must not freeze the session.

    The busy line is the *last* one; the command issued meanwhile targets the
    *previous* line, so it has to go through immediately.
    """
    session, _events = make_session()
    await session.start()
    first = session.subtitles.add("Quen.")
    first.finalize()
    second = session.subtitles.add("we use fast api here")
    second.finalize()

    post, started, release = _parked_gateway(second.id, "we use FastAPI here")
    with patch("httpx.AsyncClient.post", new=post):
        slow = asyncio.create_task(session.handle_command("Correct the last subtitle"))
        await asyncio.wait_for(started.wait(), timeout=5)
        assert second.id in session._pending

        other = await session.handle_command("Correct the previous subtitle")
        assert other.outcome is CorrectionOutcome.APPLIED
        assert other.subtitle_id == first.id

        release.set()
        await asyncio.wait_for(slow, timeout=5)

    assert session._pending == set()


async def test_a_typed_natural_command_produces_the_same_result_as_a_spoken_one(stub):
    """One parser, one resolver, two input paths.

    The correction itself is not a command and must not linger in the
    transcript, exactly as a spoken command does not.
    """
    session, events = make_session()
    await session.start()
    for text in ("First line.", "Second line.", "we should changed the config"):
        session.subtitles.add(text).finalize()
    third = session.subtitles.originals()[2]

    result = await session.handle_command('Change the 3rd sentence to "To deployed."')

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.subtitle_id == third.id
    assert session.subtitles.get(third.id).text == "we should changed the config"
    assert corrected_text(session, third.id) == "To deployed."
    assert len(events_of(events, EventType.CORRECTION)) == 1
    assert events_of(events, EventType.CORRECTION_PENDING) == []


async def test_a_correction_sends_the_new_child_line_to_the_browser(stub):
    """The corrected wording must be sent as a subtitle, not just logged.

    The correction event carries the original, so the browser can redraw that
    line but has never heard about the child. Without this event the transcript
    still shows only what AssemblyAI heard, and the correction is invisible.
    """
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("we should changed the config")
    sub.finalize()

    await session.handle_command('Change the last sentence to "To deployed."')

    correction = events_of(events, EventType.CORRECTION)[0]
    child_id = correction["payload"]["corrected_subtitle_id"]
    assert child_id != sub.id

    sent = [e for e in events_of(events, EventType.SUBTITLE) if e["subtitle"]["id"] == child_id]
    assert len(sent) == 1, "the inserted child must be broadcast exactly once"
    assert sent[0]["subtitle"]["text"] == "To deployed."
    assert sent[0]["subtitle"]["corrects_id"] == sub.id
    assert sent[0]["subtitle"]["raw_text"] == "we should changed the config"
    assert sent[0]["payload"]["partial"] is False

    # It has to arrive before the correction event, so the UI is not told a
    # line is finished before the finished line exists.
    order = [e["type"] for e in events]
    assert order.index("subtitle") < order.index("correction")

    # The original is still broadcast nowhere new: it was never modified.
    assert not [e for e in events_of(events, EventType.SUBTITLE) if e["subtitle"]["id"] == sub.id]


async def test_repeated_corrections_broadcast_children_in_order(stub):
    """Each child must arrive in the order it was made.

    The browser places a child under its original by reference, so an
    out-of-order event would render the edits stacked backwards.
    """
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("we should changed the config")
    sub.finalize()

    await session.handle_command('Change the last sentence to "To deployed."')
    await session.handle_command('Change the last sentence to "Deployed."')

    texts = [
        e["subtitle"]["text"]
        for e in events_of(events, EventType.SUBTITLE)
        if (e["subtitle"] or {}).get("id") not in (None, sub.id)
    ]
    assert texts == ["To deployed.", "Deployed."]


async def test_a_spoken_natural_command_is_removed_from_the_transcript(stub):
    """Spoken commands must not become subtitle rows."""
    session, events = make_session()
    await session.start()
    session.subtitles.add("we should changed the config").finalize()

    await stub.last.emit_final('Change the last sentence to "To deployed."')

    removals = events_of(events, EventType.SUBTITLE_REMOVED)
    assert removals, "the command line must be withdrawn from the UI"
    assert removals[-1]["payload"]["reason"] == "voice_command"
    assert "Change the last sentence" not in " ".join(s.text for s in session.subtitles)


async def test_a_spoken_ordinal_command_targets_the_same_line_as_a_typed_one(stub):
    session, _events = make_session()
    await session.start()
    for text in ("First line.", "Second line.", "Third line."):
        session.subtitles.add(text).finalize()
    third = session.subtitles.originals()[2]

    await stub.last.emit_final("Change sentence 3 to Rewritten three.")

    assert session.subtitles.get(third.id).text == "Third line."
    assert corrected_text(session, third.id) == "Rewritten three."


async def test_the_transcript_count_ignores_corrections(stub):
    """A correction is not a sentence the user spoke.

    If the UI counted it, the count would climb on every edit and any spoken
    "the third sentence" would slowly start addressing the wrong line.
    """
    session, _events = make_session()
    await session.start()
    for text in ("First line.", "Second line.", "Third line."):
        session.subtitles.add(text).finalize()

    assert len(session.subtitles.originals()) == 3
    await session.handle_command('Change sentence 1 to "One, revised."')
    assert len(session.subtitles) == 4
    assert len(session.subtitles.originals()) == 3


# ================================================ unsupported spoken commands
#
# The rehearsal's worst finding: "Correct the last sentence." passed the voice
# gate, failed to parse, and was then dropped on the floor. The instruction
# stayed on screen as though it were a transcript line and the user got silence.
# An uncarried instruction must never be indistinguishable from content.


async def test_an_unsupported_spoken_command_is_withdrawn_and_explained(stub):
    """Gate says command, parser says no: say so, do not stay silent."""
    session, events = make_session()
    await session.start()
    session.subtitles.add("we deployed the application on Quen clusters").finalize()

    # Passes the gate ("change ...") but names a line we cannot resolve.
    await stub.last.emit_final("Change the last thing to hello.")

    removals = events_of(events, EventType.SUBTITLE_REMOVED)
    assert removals, "an uncarried command must be pulled off the transcript"
    assert removals[-1]["payload"]["reason"] == "voice_command"

    # The command must not linger as a subtitle.
    assert all("hello" not in s.text for s in session.subtitles)

    # And the user must be told, not left guessing.
    corrections = events_of(events, EventType.CORRECTION)
    assert corrections, "an unsupported spoken command must produce feedback"
    assert corrections[-1]["payload"]["outcome"] == CorrectionOutcome.UNSUPPORTED.value
    assert corrections[-1]["payload"]["message"]


async def test_an_unsupported_spoken_command_changes_nothing(stub):
    """No model call, and above all no mutation of the line it aimed at."""
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("we deployed the application on Quen clusters")
    sub.finalize()

    with patch("httpx.AsyncClient.post") as post:
        await stub.last.emit_final("Change the last thing to hello.")

    assert post.call_count == 0, "an unsupported command must not reach the model"
    assert session.subtitles.get(sub.id).text == sub.text
    assert len(session.subtitles) == 1
    assert not events_of(events, EventType.CORRECTION_PENDING)


async def test_ordinary_dictation_containing_a_command_word_is_left_alone(stub):
    """The gate must not fire on prose that merely contains a keyword.

    Anchoring the gate to the start of the utterance is what makes reporting an
    unparsed command safe: without it, "I need to change the config" would be
    withdrawn from the transcript.
    """
    session, events = make_session()
    await session.start()

    for spoken in (
        "I need to remember to lock the door before I leave.",
        "We should change the configuration before shipping.",
    ):
        await stub.last.emit_final(spoken)

    assert len(session.subtitles) == 2
    assert not events_of(events, EventType.SUBTITLE_REMOVED)
    assert not events_of(events, EventType.CORRECTION_PENDING)


# ================================================ natural spoken corrections
#
# The two exact phrases that failed live, verified end to end through the real
# voice path: the sentence/subtitle synonym, and a comma the transcriber put
# after "to".


async def test_a_spoken_sentence_correction_reaches_the_model(stub):
    session, _events = make_session()
    await session.start()
    sub = session.subtitles.add("we deployed the application on Quen clusters")
    sub.finalize()

    with patch("httpx.AsyncClient.post", return_value=_gateway_says(sub.id, "Qwen.")) as post:
        await stub.last.emit_final("Correct the last sentence.")

    assert post.call_count == 1, "the model must actually be consulted"
    assert corrected_text(session, sub.id) == "Qwen."
    assert session.subtitles.get(sub.id).text == "we deployed the application on Quen clusters"


async def test_a_spoken_command_with_a_comma_after_to_still_rewrites_the_line(stub):
    """AssemblyAI punctuated this as "...to, we deployed it". It must still work."""
    session, _events = make_session()
    await session.start()
    sub = session.subtitles.add("we deployed the application on Qwen clusters")
    sub.finalize()

    with patch("httpx.AsyncClient.post") as post:
        await stub.last.emit_final("Change the last sentence to, we deployed it to Qwen clusters.")

    assert post.call_count == 0
    assert corrected_text(session, sub.id) == "we deployed it to Qwen clusters."


async def test_a_spoken_command_does_not_call_the_model(stub):
    """A dictated rewrite is already the answer, so the gateway is pointless."""
    session, _events = make_session()
    await session.start()
    sub = session.subtitles.add("we deployed the application on Qwen clusters")
    sub.finalize()

    with patch("httpx.AsyncClient.post") as post:
        await stub.last.emit_final('Change the last sentence to "We deploy it on Friday."')

    assert post.call_count == 0, "a literal rewrite must stay deterministic"
    assert corrected_text(session, sub.id) == "We deploy it on Friday."


async def test_ordinals_still_count_originals_after_two_corrections(stub):
    """Corrections are not sentences the user spoke, so they must not renumber.

    "the third sentence" has to keep meaning the third thing the user said even
    once two children have been inserted underneath the second.
    """
    session, _events = make_session()
    await session.start()
    for text in ("First line.", "Second line.", "Third line."):
        session.subtitles.add(text).finalize()
    second, third = session.subtitles.originals()[1:3]

    await session.handle_command("Change sentence 2 to Second, revised once.")
    await session.handle_command("Change sentence 2 to Second, revised twice.")

    assert len(session.subtitles) == 5
    assert len(session.subtitles.originals()) == 3

    await stub.last.emit_final("Change the third sentence to Rewritten three.")

    # `corrected_text` insists on exactly one child, so read the newest directly:
    # a second revision must not displace the first in the ordering.
    assert [c.text for c in session.subtitles.corrections_of(third.id)] == ["Rewritten three."]
    assert [c.text for c in session.subtitles.corrections_of(second.id)] == [
        "Second, revised once.",
        "Second, revised twice.",
    ]
    assert session.subtitles.get(second.id).text == "Second line."


# ================ spoken "correct <term> in <line>" ========================
# The voice gate is a list of fixed verb+target prefixes, and a command with a
# word in the middle matched none of them: the instruction was spoken, cleared
# the gate as ordinary dictation, and was left in the transcript as a subtitle.


async def test_a_spoken_named_term_command_is_withdrawn_from_the_transcript(stub):
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("Can you type Symfpony?")
    sub.finalize()

    with patch(
        "httpx.AsyncClient.post", return_value=_gateway_says(sub.id, "Can you type Symphony?")
    ):
        await stub.last.emit_final("Correct Symfpony in the last sentence.")

    removals = events_of(events, EventType.SUBTITLE_REMOVED)
    assert removals, "the command line must be removed, not left on screen"
    assert removals[-1]["payload"]["reason"] == "voice_command"
    assert "Correct Symfpony" in removals[-1]["subtitle"]["text"]

    # The instruction is gone from state; the spoken line it referred to is not.
    texts = [s.text for s in session.subtitles]
    assert not any("Correct Symfpony" in t for t in texts)
    assert session.subtitles.get(sub.id).text == "Can you type Symfpony?"


async def test_a_spoken_named_term_command_reaches_the_model(stub):
    session, _events = make_session()
    await session.start()
    sub = session.subtitles.add("Can you type Symfpony?")
    sub.finalize()

    with patch(
        "httpx.AsyncClient.post", return_value=_gateway_says(sub.id, "Can you type Symphony?")
    ) as post:
        await stub.last.emit_final("Correct Symfpony in the last sentence.")

    assert post.call_count == 1, "the model decides the replacement, not the grammar"
    assert corrected_text(session, sub.id) == "Can you type Symphony?"
    assert session.subtitles.get(sub.id).text == "Can you type Symfpony?"


async def test_a_named_term_command_shows_a_pending_state_while_the_model_thinks(stub):
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("Can you type Symfpony?")
    sub.finalize()

    with patch(
        "httpx.AsyncClient.post", return_value=_gateway_says(sub.id, "Can you type Symphony?")
    ):
        await stub.last.emit_final("Correct Symfpony in the last sentence.")

    pending = events_of(events, EventType.CORRECTION_PENDING)
    assert pending, "the user must see that the correction is still in flight"


async def test_a_named_term_command_in_the_previous_line_targets_the_previous_line(stub):
    session, _events = make_session()
    await session.start()
    first = session.subtitles.add("Can you type Symfpony?")
    first.finalize()
    second = session.subtitles.add("Moving on to the next topic.")
    second.finalize()

    with patch(
        "httpx.AsyncClient.post", return_value=_gateway_says(first.id, "Can you type Symphony?")
    ):
        await stub.last.emit_final("Fix Symfpony in the previous sentence.")

    assert corrected_text(session, first.id) == "Can you type Symphony?"
    assert session.subtitles.get(second.id).text == "Moving on to the next topic."


async def test_prose_mentioning_the_phrase_is_left_in_the_transcript(stub):
    """The safety case: a report of having corrected something is dictation."""
    session, _events = make_session()
    await session.start()

    spoken = "I corrected Symfpony yesterday in the last sentence of my report."
    await stub.last.emit_final(spoken)

    assert [s.text for s in session.subtitles] == [spoken]


async def test_prose_about_fixing_something_is_left_in_the_transcript(stub):
    session, _events = make_session()
    await session.start()

    spoken = "I need to fix the error in the last line of the report."
    await stub.last.emit_final(spoken)

    assert [s.text for s in session.subtitles] == [spoken]
