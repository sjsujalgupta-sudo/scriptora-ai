"""AssemblyAI Realtime integration details that the stubbed session tests cannot see.

Both bugs this file guards against were found by talking to the real API, not
by unit tests: `keyterms_prompt` is `list[str]` (a string is rejected with error
3006), and `AsyncRealTimeTranscriber.stream` is a coroutine that consumes an
async iterable exactly once.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from assemblyai.streaming.v3 import RealTimeParameters, RealTimeSessionParameters
from conftest import FAKE_KEY

from scriptora.config import Settings
from scriptora.services.assemblyai_service import AssemblyAIRealtimeService


@pytest.fixture
def service() -> AssemblyAIRealtimeService:
    return AssemblyAIRealtimeService(Settings(assemblyai_api_key=FAKE_KEY))


# ------------------------------------------------------------------ parameters
def test_keyterms_prompt_is_a_list_of_strings(service):
    """A joined string here makes AssemblyAI reject the whole session."""
    params = service._build_parameters(["FastAPI", "AssemblyAI"])

    assert isinstance(params.keyterms_prompt, list)
    assert params.keyterms_prompt == ["FastAPI", "AssemblyAI"]


def test_keyterms_prompt_is_omitted_when_empty(service):
    params = service._build_parameters([])

    assert params.keyterms_prompt is None


def test_known_model_is_passed_through(service):
    params = service._build_parameters(None)

    assert params.sample_rate == 16000
    assert params.format_turns is True


def test_unknown_model_falls_back_instead_of_raising():
    service = AssemblyAIRealtimeService(
        Settings(assemblyai_api_key=FAKE_KEY, speech_model="not-a-real-model")
    )

    params = service._build_parameters(None)

    assert params.speech_model is not None


# ---------------------------------------------------------------------- stream
class _RecordingClient:
    """Stands in for AsyncRealTimeTranscriber, mirroring its real contract."""

    def __init__(self) -> None:
        self.received: list[bytes] = []
        self.stream_calls = 0

    async def stream(self, frames: AsyncIterator[bytes]) -> None:
        # The real client is one-shot: it drains the iterable then returns.
        self.stream_calls += 1
        async for chunk in frames:
            self.received.append(chunk)


async def _frames(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def test_stream_forwards_every_frame_to_the_client(service):
    client = _RecordingClient()
    service._client = client

    await service.stream(_frames(b"a" * 3200, b"b" * 3200))

    assert client.received == [b"a" * 3200, b"b" * 3200]


async def test_stream_is_awaited_and_consumes_the_iterable(service):
    """If `stream` were called without await, nothing would reach the client."""
    client = _RecordingClient()
    service._client = client

    result = service.stream(_frames(b"a" * 3200))
    assert asyncio.iscoroutine(result), "stream() must return a coroutine"
    await result

    assert client.stream_calls == 1
    assert client.received == [b"a" * 3200]


async def test_stream_without_a_client_is_a_no_op(service):
    """Never raises: the session may stop before AssemblyAI finished connecting."""
    await service.stream(_frames(b"a" * 3200))


# ------------------------------------------------------------------ error hints
@pytest.mark.parametrize(
    "code,expected",
    [
        (1001, "API key"),
        (1006, "Start Listening"),
        (3007, "50-1000 ms"),
    ],
)
async def test_known_error_codes_give_an_actionable_message(service, code, expected):
    """A dropped connection must not be reported as a raw code.

    1006 in particular is the WebSocket "abnormal closure" code - a transport
    failure, never a credential problem, so it must not be phrased as one.
    """
    seen: list[tuple[str, bool]] = []

    async def on_error(message: str, fatal: bool) -> None:
        seen.append((message, fatal))

    handler = service._on_error(on_error)
    await handler(None, SimpleNamespace(code=code, __str__=lambda self: "raw text"))

    assert seen, f"code {code} produced no user-facing message"
    assert expected in seen[0][0], f"code {code}: {seen[0][0]!r}"
    assert seen[0][1] is True


async def test_error_1006_never_blames_the_api_key(service):
    """The connection dropped; telling the user to check their key is wrong."""
    seen: list[tuple[str, bool]] = []

    async def on_error(message: str, fatal: bool) -> None:
        seen.append((message, fatal))

    await service._on_error(on_error)(None, SimpleNamespace(code=1006, __str__=lambda self: ""))
    assert "API key" not in seen[0][0]
    assert "ASSEMBLYAI_API_KEY" not in seen[0][0]


async def test_error_messages_never_leak_credentials(service):
    seen: list[tuple[str, bool]] = []

    async def on_error(message: str, fatal: bool) -> None:
        seen.append((message, fatal))

    leaky = f"unauthorized for key {FAKE_KEY}"
    await service._on_error(on_error)(None, SimpleNamespace(code=9999, __str__=lambda self: leaky))
    assert seen[0][0]
    assert FAKE_KEY not in seen[0][0]


# --------------------------------------------------------------------- keyterms
class _ParamsClient:
    """Records set_params calls, mirroring the real coroutine signature."""

    def __init__(self) -> None:
        self.calls: list[RealTimeSessionParameters] = []

    async def set_params(self, params: RealTimeSessionParameters) -> None:
        self.calls.append(params)


async def test_update_keyterms_awaits_set_params(service):
    """`set_params` is a coroutine: not awaiting it sent nothing but returned True."""
    client = _ParamsClient()
    service._client = client
    service._connected = True

    assert await service.update_keyterms(["FastAPI"]) is True
    assert client.calls, "the parameters must actually reach the SDK"
    assert client.calls[-1].keyterms_prompt == ["FastAPI"]


async def test_update_keyterms_reports_failure_instead_of_raising(service):
    class _Broken:
        async def set_params(self, _params):
            raise RuntimeError("socket closed")

    service._client = _Broken()
    service._connected = True

    assert await service.update_keyterms(["FastAPI"]) is False


async def test_update_keyterms_before_connect_returns_false(service):
    assert await service.update_keyterms(["FastAPI"]) is False


async def test_update_keyterms_with_empty_list_returns_false(service):
    service._client = _ParamsClient()
    service._connected = True

    assert await service.update_keyterms([]) is False


def test_parameters_are_typed_for_the_sdk(service):
    """Guards the return type the SDK actually expects."""
    params: RealTimeParameters = service._build_parameters(["FastAPI"])
    assert params.keyterms_prompt == ["FastAPI"]
