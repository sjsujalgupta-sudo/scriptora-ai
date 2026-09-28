"""Correction engine.

The LLM is mocked at the HTTP boundary, so these tests exercise the real
validation and fallback logic without touching the network.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from unittest.mock import patch

import httpx
import pytest

from scriptora.config import Settings
from scriptora.models.correction import (
    CorrectionOutcome,
    SubtitleCorrection,
    VocabularyAddition,
    extract_json_object,
)
from scriptora.services import correction_service
from scriptora.services.command_service import parse_command
from scriptora.services.context_service import ContextService
from scriptora.services.correction_service import CorrectionService, LLMCorrector
from scriptora.services.subtitle_service import SubtitleService


def build(subtitles: SubtitleService, context: ContextService, **overrides):
    settings = Settings(
        assemblyai_api_key="test-key-not-a-credential",
        corrector_backend=overrides.pop("backend", "rules"),
        **overrides,
    )
    return CorrectionService(settings, subtitles, context)


# ============================================================ extract_json
def test_extract_plain_json_object():
    assert extract_json_object('{"action": "no_action"}') == {"action": "no_action"}


def test_extract_json_from_markdown_fence():
    raw = '```json\n{"action": "correct_subtitle", "replacement_text": "hi"}\n```'
    assert extract_json_object(raw)["action"] == "correct_subtitle"


def test_extract_json_embedded_in_prose():
    raw = 'Sure! Here you go: {"action": "no_action", "reason": "fine"} Hope that helps.'
    assert extract_json_object(raw)["action"] == "no_action"


@pytest.mark.parametrize("raw", ["", "   ", "no json here at all", "{broken json"])
def test_extract_rejects_unparseable(raw):
    with pytest.raises(ValueError):
        extract_json_object(raw)


# ================================================== structured-output schema
def test_correction_model_accepts_well_formed_output():
    parsed = SubtitleCorrection.model_validate(
        {
            "action": "correct_subtitle",
            "target_subtitle_id": "abc123",
            "replacement_text": "Using FastAPI.",
            "vocabulary_term": None,
            "reason": "Casing fix",
        }
    )
    assert parsed.is_valid_for({"abc123"}) is True


def test_correction_model_rejects_invented_subtitle_id():
    """A model that invents an id must not be allowed to mutate state."""
    parsed = SubtitleCorrection.model_validate(
        {
            "action": "correct_subtitle",
            "target_subtitle_id": "made-up-id",
            "replacement_text": "Using FastAPI.",
            "reason": "",
        }
    )
    assert parsed.is_valid_for({"abc123"}) is False


def test_correction_model_rejects_empty_replacement():
    parsed = SubtitleCorrection.model_validate(
        {"action": "correct_subtitle", "target_subtitle_id": "abc123", "replacement_text": "  "}
    )
    assert parsed.is_valid_for({"abc123"}) is False


def test_correction_model_rejects_oversized_replacement():
    parsed = SubtitleCorrection.model_validate(
        {
            "action": "correct_subtitle",
            "target_subtitle_id": "abc123",
            "replacement_text": "x" * 5000,
        }
    )
    assert parsed.is_valid_for({"abc123"}) is False


def test_correction_model_coerces_action_casing():
    parsed = SubtitleCorrection.model_validate(
        {"action": "  Correct_Subtitle ", "target_subtitle_id": "a", "replacement_text": "b"}
    )
    assert parsed.action.value == "correct_subtitle"


def test_vocabulary_addition_validation():
    assert VocabularyAddition.model_validate(
        {"action": "add_vocabulary", "vocabulary_term": "FastAPI"}
    ).is_valid_for()
    assert not VocabularyAddition.model_validate(
        {"action": "add_vocabulary", "vocabulary_term": "  "}
    ).is_valid_for()


# ======================================================== rules corrector
async def test_rules_fixes_the_primary_demo_case(context_with_fastapi):
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context)

    sub = subtitles.add("Today we're building the backend using fast API.")
    sub.finalize()

    result = await service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.subtitle_id == sub.id
    assert subtitles.get(sub.id).text == "Today we're building the backend using FastAPI."


async def test_rules_honours_an_explicit_replace(context_with_fastapi):
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    sub = subtitles.add("We are working with Alice on this.")
    sub.finalize()

    await service.correct(parse_command("Replace Alice with Atlas."))

    assert subtitles.get(sub.id).text == "We are working with Atlas on this."


async def test_rules_reports_no_change_when_nothing_is_wrong(context_with_fastapi):
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    sub = subtitles.add("We are using FastAPI today.")
    sub.finalize()

    result = await service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.NO_ACTION
    assert result.before == result.after


async def test_correction_with_no_subtitles_is_reported_not_crashed(context_with_fastapi):
    service = build(SubtitleService(), context_with_fastapi)

    result = await service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.INVALID
    assert "no subtitles" in result.message.lower()


async def test_unsupported_command_is_reported(context_with_fastapi):
    subtitles = SubtitleService()
    subtitles.add("some text").finalize()
    service = build(subtitles, context_with_fastapi)

    result = await service.correct(parse_command("Delete the last subtitle"))

    assert result.outcome is CorrectionOutcome.UNSUPPORTED


async def test_correct_previous_targets_the_earlier_line(context_with_fastapi):
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    first = subtitles.add("We deploy with fast api.")
    first.finalize()
    subtitles.finalize("This is the second line.")

    result = await service.correct(parse_command("Fix the previous subtitle."))

    assert result.subtitle_id == first.id
    assert "FastAPI" in subtitles.get(first.id).text


# ============================================================== remember
async def test_remember_adds_to_vocabulary(context_with_fastapi):
    context = context_with_fastapi
    service = build(SubtitleService(), context)

    result = await service.correct(parse_command("Remember Kubernetes as a technical term."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert "Kubernetes" in context.terms
    assert result.vocabulary_term == "Kubernetes"


async def test_remember_is_idempotent(context_with_fastapi):
    context = context_with_fastapi
    service = build(SubtitleService(), context)
    await service.correct(parse_command("Remember Kubernetes."))

    result = await service.correct(parse_command("Remember Kubernetes."))

    assert "already" in result.message.lower()
    assert context.terms.count("Kubernetes") == 1


async def test_remembered_term_is_used_by_the_next_correction(context_with_fastapi):
    """The memory feature must actually affect later corrections."""
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context)

    await service.correct(parse_command("Remember FastAPI as a technical term."))
    sub = subtitles.add("We rely on fast api for the service.")
    sub.finalize()

    await service.correct(parse_command("Correct the last subtitle."))

    assert "FastAPI" in subtitles.get(sub.id).text


# ============================================================== LLM paths
def _llm_response(content: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]},
    )


async def test_llm_result_is_applied_when_valid(context_with_fastapi):
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("Today we're building the backend using fast API.")
    sub.finalize()

    good = (
        '{"action": "correct_subtitle", "target_subtitle_id": "' + sub.id + '", '
        '"replacement_text": "Today we are building the backend using FastAPI.", '
        '"vocabulary_term": null, "reason": "Known term"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(good)):
        result = await service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.backend == "llm"
    assert subtitles.get(sub.id).text == "Today we are building the backend using FastAPI."


async def test_llm_response_with_invented_id_is_discarded(context_with_fastapi):
    """Even a confident wrong answer must not corrupt the transcript."""
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("Using fast API.")
    sub.finalize()

    bad = (
        '{"action": "correct_subtitle", "target_subtitle_id": "totally-made-up", '
        '"replacement_text": "HACKED", "reason": "x"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(bad)):
        await service.correct(parse_command("Correct the last subtitle."))

    # Falls back to the deterministic result; "HACKED" never reaches state.
    assert subtitles.get(sub.id).text == "Using FastAPI."
    assert "HACKED" not in subtitles.get(sub.id).text


@pytest.mark.parametrize(
    "bad_content",
    [
        "not json at all",
        '{"action": "correct_subtitle"}',  # missing required fields
        '{"action": "explode", "target_subtitle_id": "a", "replacement_text": "b"}',
        "",
    ],
)
async def test_invalid_llm_output_falls_back_to_rules(bad_content, context_with_fastapi):
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("Using fast API.")
    sub.finalize()

    with patch("httpx.AsyncClient.post", return_value=_llm_response(bad_content)):
        result = await service.correct(parse_command("Correct the last subtitle."))

    assert subtitles.get(sub.id).text == "Using FastAPI."
    assert result.outcome is CorrectionOutcome.APPLIED


async def test_llm_http_error_falls_back_to_rules(context_with_fastapi):
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("Using fast API.")
    sub.finalize()

    with patch("httpx.AsyncClient.post", return_value=httpx.Response(500, text="boom")):
        result = await service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.backend == "rules"
    assert "AI unavailable" in result.reason


async def test_llm_network_error_falls_back_to_rules(context_with_fastapi):
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("Using fast API.")
    sub.finalize()

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("no route")):
        result = await service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert "AI unavailable" in result.reason


async def test_llm_malformed_envelope_falls_back_to_rules(context_with_fastapi):
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("Using fast API.")
    sub.finalize()

    with patch(
        "httpx.AsyncClient.post", return_value=httpx.Response(200, json={"unexpected": True})
    ):
        result = await service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.backend == "rules"


async def test_explicit_replace_wins_over_llm_rewrite(context_with_fastapi):
    """A literal user instruction must not be overridden by the model.

    The model is not consulted at all: the deterministic path implements a
    literal edit exactly, so the gateway call could only produce an answer that
    gets discarded. Asserting the call count is the point of the test - it is
    the regression guard against paying (and blocking on) a dead round-trip.
    """
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("We work with Alice.")
    sub.finalize()

    llm_says = (
        '{"action": "correct_subtitle", "target_subtitle_id": "' + sub.id + '", '
        '"replacement_text": "We work with Bob and the team.", "reason": "x"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(llm_says)) as post:
        result = await service.correct(parse_command("Replace Alice with Atlas."))

    assert subtitles.get(sub.id).text == "We work with Atlas."
    # No HTTP call: the answer could not have changed the outcome.
    assert post.call_count == 0
    # And it is never misattributed to the model.
    assert result.backend == "rules"


async def test_replace_skips_the_llm_even_when_the_edit_is_impossible(context_with_fastapi):
    """A replace whose target text is absent still costs no gateway call.

    The literal edit produced nothing, so the result is a no_action - the same
    outcome the previous code produced after paying for a discarded answer.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi, backend="llm")
    sub = subtitles.add("Nothing relevant here.")
    sub.finalize()

    llm_says = (
        '{"action": "correct_subtitle", "target_subtitle_id": "' + sub.id + '", '
        '"replacement_text": "Invented rewrite.", "reason": "x"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(llm_says)) as post:
        result = await service.correct(parse_command("Replace missing with present."))

    assert post.call_count == 0
    assert result.outcome is CorrectionOutcome.NO_ACTION
    assert subtitles.get(sub.id).text == "Nothing relevant here."


async def test_correction_commands_still_consult_the_llm(context_with_fastapi):
    """The short-circuit must not disable the model for real corrections."""
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("we use fast api here")
    sub.finalize()

    llm_says = (
        '{"action": "correct_subtitle", "target_subtitle_id": "' + sub.id + '", '
        '"replacement_text": "we use FastAPI here", "reason": "canonical spelling"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(llm_says)) as post:
        result = await service.correct(parse_command("Correct the last subtitle"))

    assert post.call_count == 1
    assert result.backend == "llm"
    assert subtitles.get(sub.id).text == "we use FastAPI here"


async def test_no_api_key_disables_the_llm_backend(context_with_fastapi):
    settings = Settings(assemblyai_api_key=None, corrector_backend="auto")
    subtitles = SubtitleService()
    service = CorrectionService(settings, subtitles, context_with_fastapi)

    assert service.active_backend == "rules"
    sub = subtitles.add("Using fast API.")
    sub.finalize()
    result = await service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED


# ====================================================== non-blocking gateway
def test_the_gateway_call_path_is_genuinely_coroutine():
    """Structural guard: no blocking call can hide inside the awaited path.

    The behavioural test below can only observe that the event loop was free
    while the request was outstanding. If a *synchronous* HTTP call were
    reintroduced inside the coroutine, that test would hang rather than fail,
    because a blocked thread cannot be woken by any asyncio construct. These
    assertions fail immediately instead, naming the exact frame that regressed.
    """
    assert inspect.iscoroutinefunction(CorrectionService.correct)
    assert inspect.iscoroutinefunction(LLMCorrector.correct_subtitle)
    assert inspect.iscoroutinefunction(LLMCorrector._complete)


def test_the_blocking_httpx_post_is_not_called_anywhere_in_the_module():
    """The synchronous `httpx.post` this task removed must stay gone.

    Checked at the source level on purpose. A reintroduction is invisible to the
    behavioural test below - a thread stuck inside a blocking call cannot be
    woken by any asyncio construct, so that test would hang rather than fail -
    and the coroutine assertions would still pass, because `correct()` would be
    awaiting *something*. Parsing the AST names the actual regression instead of
    a proxy for it, and ignores the prose in comments that explains it.
    """
    blocking = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(correction_service)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "post"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "httpx"
    ]
    assert blocking == [], "blocking httpx.post reintroduced in correction_service"


async def test_a_pending_gateway_request_does_not_block_the_event_loop(context_with_fastapi):
    """A slow LLM must not stall the realtime session's event loop.

    The fake gateway parks on an event owned by the test, so "the request is
    still in flight" is an established fact, not a timing guess - there is no
    sleeping and no dependence on how loaded the machine is. The timeouts below
    are deadlock guards, not the assertion. What is asserted is that unrelated
    async work ran while the request was outstanding, and that the correction
    only completed once the gateway was released.

    Before the async conversion this path held the event loop for the whole
    request, so the watchdog could not run at all and audio streaming to
    AssemblyAI stalled behind the LLM.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi, backend="llm")
    sub = subtitles.add("we use fast api here")
    sub.finalize()

    llm_says = (
        '{"action": "correct_subtitle", "target_subtitle_id": "' + sub.id + '", '
        '"replacement_text": "we use FastAPI here", "reason": "canonical spelling"}'
    )

    request_started = asyncio.Event()
    release = asyncio.Event()

    async def parked_post(*_args, **_kwargs):
        request_started.set()
        await release.wait()
        return _llm_response(llm_says)

    async def unrelated_work() -> str:
        return "responsive"

    with patch("httpx.AsyncClient.post", new=parked_post):
        pending = asyncio.create_task(service.correct(parse_command("Correct the last subtitle")))
        await asyncio.wait_for(request_started.wait(), timeout=5)

        # The request is outstanding. Unrelated work must still be schedulable.
        watchdog = asyncio.create_task(unrelated_work())
        done, _ = await asyncio.wait({watchdog}, timeout=5)
        assert watchdog in done, "event loop was blocked while the gateway request was pending"
        assert not pending.done(), "correction completed before the gateway replied"

        release.set()
        result = await asyncio.wait_for(pending, timeout=5)

    # And the model answer is still adopted, unchanged.
    assert result.backend == "llm"
    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.subtitle_id == sub.id
    assert subtitles.get(sub.id).text == "we use FastAPI here"


async def test_a_literal_replace_costs_no_gateway_request_while_others_wait(context_with_fastapi):
    """The literal-edit short-circuit holds under the async conversion.

    Same guarantee as the synchronous version, restated on the async path: a
    request that cannot change the outcome is never sent, so it cannot occupy
    the event loop either.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi, backend="llm")
    sub = subtitles.add("We are working with Alice on this.")
    sub.finalize()

    llm_says = (
        '{"action": "correct_subtitle", "target_subtitle_id": "' + sub.id + '", '
        '"replacement_text": "Invented rewrite.", "reason": "x"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(llm_says)) as post:
        result = await service.correct(parse_command("Replace Alice with Atlas."))

    assert post.call_count == 0
    assert result.backend == "rules"
    assert result.outcome is CorrectionOutcome.APPLIED
    assert subtitles.get(sub.id).text == "We are working with Atlas on this."


async def test_the_gateway_timeout_is_preserved(context_with_fastapi):
    """The 25 s budget must survive the move to an async client.

    Changing it silently would either hang sessions or give up on slow-but-valid
    corrections, so it is pinned by a test rather than left to the default.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi, backend="llm")
    sub = subtitles.add("we use fast api here")
    sub.finalize()

    seen: dict = {}

    async def capture_post(*_args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return _llm_response(
            '{"action": "correct_subtitle", "target_subtitle_id": "' + sub.id + '", '
            '"replacement_text": "we use FastAPI here", "reason": "canonical"}'
        )

    with patch("httpx.AsyncClient.post", new=capture_post):
        await service.correct(parse_command("Correct the last subtitle."))

    assert seen["timeout"] == 25.0


async def test_an_unreachable_gateway_still_falls_back_without_raising(context_with_fastapi):
    """Failure handling is unchanged by the async conversion.

    A transport error on the awaited call must collapse to the same rules-based
    result and the same "AI unavailable" reason the synchronous version gave.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi, backend="llm")
    sub = subtitles.add("Using fast API.")
    sub.finalize()

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("no route")):
        result = await service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.backend == "rules"
    assert "AI unavailable" in result.reason
    assert subtitles.get(sub.id).text == "Using FastAPI."


async def test_the_async_client_is_closed_after_each_request(context_with_fastapi):
    """A per-call client must not leak an unclosed connection pool.

    Corrections are frequent enough over a long session for a leaked pool to
    matter, and an unclosed AsyncClient is a ResourceWarning the test suite
    would otherwise hide.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi, backend="llm")
    sub = subtitles.add("Using fast API.")
    sub.finalize()

    closed: list[bool] = []

    class RecordingClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc) -> bool:
            closed.append(True)
            return False

        async def post(self, *_args, **_kwargs):
            return _llm_response(
                '{"action": "correct_subtitle", "target_subtitle_id": "' + sub.id + '", '
                '"replacement_text": "Using FastAPI.", "reason": "canonical"}'
            )

    with patch("httpx.AsyncClient", RecordingClient):
        result = await service.correct(parse_command("Correct the last subtitle."))

    assert closed == [True]
    assert result.backend == "llm"
