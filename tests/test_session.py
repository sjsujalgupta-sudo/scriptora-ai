"""Session orchestration and the WebSocket transport.

AssemblyAI is stubbed at the service boundary, so the routing, event flow and
error handling are exercised without any network access.
"""

from __future__ import annotations

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
        async for chunk in frames:
            self.streams.append(chunk)

    async def update_keyterms(self, keyterms: list[str]) -> bool:
        self.keyterms.append(list(keyterms))
        return True

    async def disconnect(self) -> None:
        self.disconnected = True
        self.connected = False

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
async def test_voice_command_is_not_kept_as_a_subtitle(stub):
    """A spoken instruction must not also appear in the transcript."""
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("Today we are building the backend using fast API.")
    sub.finalize()

    await stub.last.emit_final("Correct the last subtitle. It's FastAPI.")

    texts = [s.text for s in session.subtitles]
    assert texts == ["Today we are building the backend using FastAPI."]
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
    assert session.subtitles.get(sub.id).text.endswith("using FastAPI.")

    """A bare "correct the last subtitle" works off project context alone.

    AssemblyAI is seeded, so hearing "assembly AI" is a known spelling error
    that needs no term named in the command.
    """
    session, events = make_session()
    await session.start()
    sub = session.subtitles.add("We deploy on assembly AI every day.")
    sub.finalize()

    await stub.last.emit_final("Correct the last subtitle.")

    assert session.subtitles.get(sub.id).text == "We deploy on AssemblyAI every day."
    assert len(events_of(events, EventType.CORRECTION)) == 1


async def test_named_term_in_the_command_corrects_before_being_remembered(stub):
    """The primary demo: "it's FastAPI" fixes the line with no prior context."""
    session, _ = make_session()
    await session.start()
    sub = session.subtitles.add("Today we are building the backend using fast API.")
    sub.finalize()
    assert "FastAPI" not in session.context.terms

    await stub.last.emit_final("Correct the last subtitle. It's FastAPI.")

    assert session.subtitles.get(sub.id).text.endswith("using FastAPI.")


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


async def asyncio_sleep():
    """Let the audio pump task drain the queue."""
    import asyncio

    await asyncio.sleep(0.01)


# ============================================ correction through the async path
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
    assert session.subtitles.get(sub.id).text == "we use FastAPI here"

    corrections = events_of(events, EventType.CORRECTION)
    assert len(corrections) == 1
    assert corrections[0]["payload"]["outcome"] == "applied"
    assert corrections[0]["payload"]["backend"] == "llm"
    assert corrections[0]["payload"]["subtitle_id"] == sub.id


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
    assert session.subtitles.get(sub.id).text == "we use FastAPI here"
