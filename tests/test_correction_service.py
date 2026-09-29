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
from scriptora.services.command_service import CommandKind, parse_command
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
def assert_corrected_as_child(
    subtitles: SubtitleService, original_id: str, heard: str, expected: str
):
    """Assert the whole correction contract in one place.

    The original line must survive untouched, the corrected wording must be a
    separate child that references it, and that child must sit *immediately*
    after the original. Every correction test funnels through here so a change
    to any one of those three properties cannot pass silently in the other 20.
    """
    original = subtitles.get(original_id)
    assert original.text == heard, "the original line must never be overwritten"
    assert original.raw_text is None, "an original is never corrected in place"

    children = subtitles.corrections_of(original_id)
    assert len(children) == 1, f"expected exactly one correction, got {len(children)}"
    child = children[0]
    assert child.text == expected
    assert child.corrects_id == original_id
    assert child.is_correction

    order = subtitles.items
    assert order.index(child) == order.index(original) + 1, (
        "the corrected line must appear immediately after the line it corrects"
    )
    return child


def assert_uncorrected(subtitles: SubtitleService, original_id: str, heard: str):
    """Assert nothing was written: the line is still exactly what was heard.

    The counterpart to `assert_corrected_as_child`, for the paths that must
    decline to act. Without it, a silent no-op and a correction that somehow
    failed to insert a child would look identical.
    """
    original = subtitles.get(original_id)
    assert original.text == heard, "a line that was not corrected must be untouched"
    assert subtitles.corrections_of(original_id) == [], "no correction should have been inserted"


async def test_rules_fixes_the_primary_demo_case(context_with_fastapi):
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context)

    sub = subtitles.add("Today we're building the backend using fast API.")
    sub.finalize()

    result = await service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.subtitle_id == sub.id
    assert_corrected_as_child(
        subtitles,
        sub.id,
        "Today we're building the backend using fast API.",
        "Today we're building the backend using FastAPI.",
    )
    # The result must point at the child so the UI can render exactly the line
    # that changed, while `subtitle_id` keeps pointing at the original.
    assert result.after == "Today we're building the backend using FastAPI."
    assert result.corrected_subtitle_id == subtitles.last().id


async def test_rules_honours_an_explicit_replace(context_with_fastapi):
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    sub = subtitles.add("We are working with Alice on this.")
    sub.finalize()

    await service.correct(parse_command("Replace Alice with Atlas."))

    assert_corrected_as_child(
        subtitles,
        sub.id,
        "We are working with Alice on this.",
        "We are working with Atlas on this.",
    )


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
    second = subtitles.add("This is the second line.")
    second.finalize()

    result = await service.correct(parse_command("Fix the previous subtitle."))

    assert result.subtitle_id == first.id
    assert_corrected_as_child(
        subtitles,
        first.id,
        "We deploy with fast api.",
        "We deploy with FastAPI.",
    )
    # The second line is untouched and is still the last *original*, so a
    # follow-up "correct the last subtitle" lands on it rather than on the
    # correction we just inserted after the first line.
    assert "second line" in subtitles.last_original().text
    assert subtitles.last_original().id == second.id


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

    assert_corrected_as_child(
        subtitles,
        sub.id,
        "We rely on fast api for the service.",
        "We rely on FastAPI for the service.",
    )


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
    assert_corrected_as_child(
        subtitles,
        sub.id,
        "Today we're building the backend using fast API.",
        "Today we are building the backend using FastAPI.",
    )


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
    assert_corrected_as_child(subtitles, sub.id, "Using fast API.", "Using FastAPI.")
    assert "HACKED" not in "".join(item.text for item in subtitles)


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

    assert_corrected_as_child(subtitles, sub.id, "Using fast API.", "Using FastAPI.")
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

    assert_corrected_as_child(subtitles, sub.id, "We work with Alice.", "We work with Atlas.")
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
    assert_uncorrected(subtitles, sub.id, "Nothing relevant here.")


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
    assert_corrected_as_child(subtitles, sub.id, "we use fast api here", "we use FastAPI here")


# ============================== mishearing repaired without vocabulary support
# Regression cover for the live demo failure: AssemblyAI heard "Quen", the user
# said "Correct the last subtitle", and nothing happened. The model declined
# because the prompt only ever anchored corrections to `project_vocabulary`, and
# "Qwen" was not in it. The fix is a general prompt tier, NOT a Quen->Qwen rule,
# so these tests use several different mishearings and assert the rules engine
# cannot do it alone.


def test_the_rules_engine_cannot_repair_a_wrong_letter():
    """Why the model tier exists at all.

    `make_lookup_pattern` tolerates whitespace and casing between characters but
    not a substituted letter, so "Quen" can never match "Qwen". This is asserted
    deliberately: if someone ever hard-codes Quen->Qwen into the rules path, this
    test is what notices.
    """
    context = ContextService()
    context.add_term("Qwen")
    # A wrong letter is not a spacing/casing variant, so the deterministic path
    # leaves it alone in both directions. Only the model tier repairs these.
    assert context.restore_terms("Quen.") == "Quen."
    assert context.restore_terms("Quwen.") == "Quwen."
    # ...whereas the variants it *is* built for are repaired offline, no network.
    assert context.restore_terms("q wen.") == "Qwen."
    assert context.restore_terms("Q W E N.") == "Qwen."


async def test_a_mishearing_is_repaired_when_the_term_is_not_in_the_vocabulary():
    """The reported bug, end to end through the real service.

    "Qwen" is deliberately absent from the project context, so this only passes
    if the correction came from the model reasoning about the mishearing rather
    than from a vocabulary lookup.
    """
    subtitles = SubtitleService()
    context = ContextService()
    assert "Qwen" not in context.terms
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("Quen.")
    sub.finalize()

    model_says = (
        '{"action": "correct_subtitle", "target_subtitle_id": "' + sub.id + '", '
        '"replacement_text": "Qwen.", "reason": "clear mishearing of a known tool"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(model_says)):
        result = await service.correct(parse_command("Correct the last subtitle"))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.backend == "llm"
    assert_corrected_as_child(subtitles, sub.id, "Quen.", "Qwen.")


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        ("Quen.", "Qwen."),
        ("kubernetties are hard.", "Kubernetes are hard."),
        ("we deploy with kubernets.", "we deploy with Kubernetes."),
    ],
)
async def test_the_repair_is_general_not_keyword_specific(heard, expected):
    """Several unrelated mishearings, none of them known to the codebase."""
    subtitles = SubtitleService()
    service = build(subtitles, ContextService(), backend="llm")
    sub = subtitles.add(heard)
    sub.finalize()

    model_says = (
        '{"action": "correct_subtitle", "target_subtitle_id": "' + sub.id + '", '
        '"replacement_text": "' + expected + '", "reason": "mishearing"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(model_says)):
        result = await service.correct(parse_command("Correct the last subtitle"))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert_corrected_as_child(subtitles, sub.id, heard, expected)


async def test_an_unfamiliar_word_is_left_alone(context_with_fastapi):
    """The guard against the opposite failure: inventing a word to fill a gap.

    The model returns `no_action`, so the subtitle is untouched.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi, backend="llm")
    sub = subtitles.add("blargh")
    sub.finalize()

    declined = (
        '{"action": "no_action", "target_subtitle_id": null, "replacement_text": null, '
        '"vocabulary_term": null, "reason": "term is unfamiliar; no confident correction"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(declined)):
        result = await service.correct(parse_command("Correct the last subtitle"))

    assert result.outcome is CorrectionOutcome.NO_ACTION
    assert_uncorrected(subtitles, sub.id, "blargh")


async def test_a_declined_correction_explains_itself(context_with_fastapi):
    """The user must learn *why* nothing changed.

    This was the second half of the live bug: the model said no_action with a
    clear reason, the service discarded that reason, and the UI logged only
    "No correction was needed." - which reads as the subtitle being correct.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi, backend="llm")
    sub = subtitles.add("Quen.")
    sub.finalize()

    declined = (
        '{"action": "no_action", "target_subtitle_id": null, "replacement_text": null, '
        '"vocabulary_term": null, "reason": "no confident correction for this word"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(declined)):
        result = await service.correct(parse_command("Correct the last subtitle"))

    assert result.outcome is CorrectionOutcome.NO_ACTION
    assert result.reason == "no confident correction for this word"


def test_the_prompt_allows_vocabulary_free_correction_and_forbids_inventing():
    """Guard the prompt contract, since a regression here is invisible offline.

    Verified against the live gateway, where this tier turns "Quen" into "Qwen"
    while leaving "blargh", plain sentences and real proper nouns untouched.
    """
    prompt = correction_service.SYSTEM_PROMPT
    assert "NOT in project_vocabulary" in prompt
    assert "Never invent a replacement word" in prompt
    assert "merely unfamiliar is NOT an error" in prompt


async def test_a_vocabulary_term_split_by_speech_is_rejoined(context_with_fastapi):
    """The vocabulary tier still works after the prompt change."""
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi, backend="llm")
    sub = subtitles.add("we use fast API here")
    sub.finalize()

    model_says = (
        '{"action": "correct_subtitle", "target_subtitle_id": "' + sub.id + '", '
        '"replacement_text": "we use FastAPI here", "reason": "canonical spelling"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(model_says)):
        result = await service.correct(parse_command("Correct the last subtitle"))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert_corrected_as_child(subtitles, sub.id, "we use fast API here", "we use FastAPI here")


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
    assert_corrected_as_child(subtitles, sub.id, "we use fast api here", "we use FastAPI here")


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
    assert_corrected_as_child(
        subtitles,
        sub.id,
        "We are working with Alice on this.",
        "We are working with Atlas on this.",
    )


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
    assert_corrected_as_child(subtitles, sub.id, "Using fast API.", "Using FastAPI.")


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


# ===================================================== natural set-text
async def test_set_text_replaces_the_named_ordinal(context_with_fastapi):
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    for text in ("First line.", "Second line.", "Third line."):
        subtitles.add(text).finalize()

    result = await service.correct(parse_command("Change sentence 3 to To deployed."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert_corrected_as_child(subtitles, subtitles.originals()[2].id, "Third line.", "To deployed.")
    # The lines either side of the target must be completely untouched.
    assert_uncorrected(subtitles, subtitles.originals()[0].id, "First line.")
    assert_uncorrected(subtitles, subtitles.originals()[1].id, "Second line.")


@pytest.mark.parametrize(
    "phrase", ["Change the last sentence to X", "Change this to X", "Change the last subtitle to X"]
)
async def test_this_and_last_resolve_to_the_same_line(phrase, context_with_fastapi):
    """Two words for one intent. If they ever diverged, one would be a lie."""
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    first = subtitles.add("First line.")
    first.finalize()
    second = subtitles.add("Second line.")
    second.finalize()

    result = await service.correct(parse_command(phrase))

    assert result.subtitle_id == second.id
    assert first.id not in {c.corrects_id for c in subtitles.corrections_of(second.id)}


async def test_previous_resolves_to_the_line_before_the_last(context_with_fastapi):
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    first = subtitles.add("First line.")
    first.finalize()
    second = subtitles.add("Second line.")
    second.finalize()

    result = await service.correct(parse_command("Change the previous sentence to X"))

    assert result.subtitle_id == first.id
    assert_uncorrected(subtitles, second.id, "Second line.")


async def test_a_correction_does_not_shift_later_ordinals(context_with_fastapi):
    """The property that makes "sentence 3" trustworthy over a long session.

    Inserting a child line would, if ordinals counted raw list positions, make
    every later sentence address the wrong text. Corrections are excluded from
    the numbering for exactly this reason.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    lines = [f"Line {n}." for n in range(1, 6)]
    for text in lines:
        subtitles.add(text).finalize()

    await service.correct(parse_command("Change sentence 2 to Second, fixed."))
    # A second correction, to prove it is not a one-off.
    await service.correct(parse_command("Change sentence 1 to First, fixed."))

    # The originals are, by design, still exactly what was spoken. What the
    # corrections changed is recorded against them, in order.
    assert [item.text for item in subtitles.originals()] == lines
    corrected = [
        c.text for item in subtitles.originals() for c in subtitles.corrections_of(item.id)
    ]
    assert corrected == ["First, fixed.", "Second, fixed."]
    # And the children really are interleaved in the raw list, which is the
    # condition the ordinal exclusion exists to survive.
    assert len(subtitles.items) == 7


async def test_repeated_corrections_of_one_line_are_listed_in_order(context_with_fastapi):
    """Correcting the same line twice must not reverse its two children.

    Both children sit directly after their original, so the only thing
    separating them is the insertion point. Using the original's index every
    time puts the newest correction first, which makes the edits read
    backwards and drifts them away from the line they belong to.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    for text in ("First line.", "Second line."):
        subtitles.add(text).finalize()
    first = subtitles.original_at(1)

    await service.correct(parse_command("Change sentence 1 to First, once."))
    await service.correct(parse_command("Change sentence 1 to First, twice."))

    assert [c.text for c in subtitles.corrections_of(first.id)] == [
        "First, once.",
        "First, twice.",
    ]
    # The newest edit is the one that follows the original, not the oldest.
    assert [item.text for item in subtitles.items] == [
        "First line.",
        "First, once.",
        "First, twice.",
        "Second line.",
    ]
    # The original is still untouched, and a third correction lands after both.
    assert subtitles.get(first.id).text == "First line."
    await service.correct(parse_command("Change sentence 1 to First, thrice."))
    assert [c.text for c in subtitles.corrections_of(first.id)] == [
        "First, once.",
        "First, twice.",
        "First, thrice.",
    ]


async def test_previous_skips_over_corrections(context_with_fastapi):
    """After correcting line 1, "previous" must still mean line 2.

    Counting raw positions would make it return the child under line 1.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    for text in ("First line.", "Second line.", "Third line."):
        subtitles.add(text).finalize()

    await service.correct(parse_command("Change sentence 1 to First, fixed."))
    result = await service.correct(parse_command("Change the previous sentence to X"))

    assert result.subtitle_id == subtitles.originals()[1].id


async def test_an_out_of_range_ordinal_is_refused_with_a_useful_count(context_with_fastapi):
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    for text in ("First line.", "Second line."):
        subtitles.add(text).finalize()

    result = await service.correct(parse_command("Change sentence 9 to X"))

    assert result.outcome is CorrectionOutcome.INVALID
    # Not "speak first": two lines do exist, the third just does not.
    assert "no sentence 9" in result.message.lower()
    assert "2 spoken lines" in result.message.lower()


async def test_previous_with_only_one_line_says_so(context_with_fastapi):
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    subtitles.add("Only line.").finalize()

    result = await service.correct(parse_command("Change the previous sentence to X"))

    assert result.outcome is CorrectionOutcome.INVALID
    assert "no previous" in result.message.lower()


async def test_set_text_makes_no_gateway_call(context_with_fastapi):
    """The whole point of SET_TEXT: the answer is already in the command.

    A gateway round trip here would add seconds of latency to the single most
    common correction the demo performs, and could not change the result.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi, backend="llm")
    subtitles.add("we use fast api here").finalize()

    with patch("httpx.AsyncClient.post") as post:
        result = await service.correct(parse_command("Change the last sentence to To deployed."))

    assert post.call_count == 0
    assert result.backend == "rules"
    assert result.outcome is CorrectionOutcome.APPLIED


async def test_set_text_reports_no_action_when_the_text_already_matches(context_with_fastapi):
    """Asking for what is already there must do nothing, not add a duplicate."""
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    sub = subtitles.add("To deployed.")
    sub.finalize()

    result = await service.correct(parse_command('Change the last sentence to "To deployed."'))

    assert result.outcome is CorrectionOutcome.NO_ACTION
    assert_uncorrected(subtitles, sub.id, "To deployed.")


async def test_set_text_keeps_the_terminal_punctuation_of_a_dictated_replacement(
    context_with_fastapi,
):
    """The replacement is the user's wording, not dictation noise.

    Every other command strips a trailing full stop before matching, because
    there the punctuation is an artefact of speech-to-text. Here the text is
    the thing being spoken into the transcript, so stripping it would change
    the sentence the user asked for - and would make a rewrite of an identical
    line look like a real change.
    """
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    sub = subtitles.add("to developed")
    sub.finalize()

    result = await service.correct(parse_command("Change the last sentence to To deployed."))

    assert result.after == "To deployed."


async def test_set_text_lands_the_replacement_verbatim(context_with_fastapi):
    """Punctuation and capitalisation must survive, including inner quotes."""
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    sub = subtitles.add("to developed")
    sub.finalize()

    result = await service.correct(
        parse_command("Change the last sentence to \"He said 'to be', then stopped.\"")
    )

    assert result.after == "He said 'to be', then stopped."


async def test_a_quoted_set_text_target_works_end_to_end(context_with_fastapi):
    """The exact phrasing from the demo script, verified through the engine."""
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    for text in ("First line.", "Second line.", "we should changed the config"):
        subtitles.add(text).finalize()
    third = subtitles.originals()[2]

    result = await service.correct(parse_command('Change the 3rd sentence to "To deployed."'))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert_corrected_as_child(subtitles, third.id, "we should changed the config", "To deployed.")


# ================= correct <term> in <target>, end to end ==================
# Proves the term the user pointed at actually reaches the correction pipeline
# and that repairing it needs no hard-coded knowledge of the word pair.


async def test_the_named_term_reaches_the_model_as_a_hint():
    """`find` must be handed to the LLM, which cannot otherwise see the pointer."""
    subtitles = SubtitleService()
    context = ContextService()
    command = parse_command("Correct Symfpony in the last sentence.")
    sub = subtitles.add("Can you type Symfpony?")
    sub.finalize()

    captured: dict = {}

    async def _capture(self, *args, **kwargs):
        payload = self._payload(sub, command, context, subtitles)
        captured.update(payload["instructions"])
        return SubtitleCorrection(
            outcome=CorrectionOutcome.NO_CHANGE,
            backend="llm",
            reason="captured",
        )

    with patch.object(LLMCorrector, "correct_subtitle", _capture):
        await build(subtitles, context, backend="llm").correct(command)

    assert captured["user_named_term"] == "Symfpony"
    assert captured["suggested_target_subtitle_id"] == sub.id


async def test_the_named_term_is_not_sent_as_an_explicit_edit():
    """A named mistake is a hint, never an instruction to substitute it."""
    command = parse_command("Correct Symfpony in the last sentence.")

    assert command.kind is CommandKind.CORRECT_LAST
    assert command.replace is None


async def test_a_model_backed_correction_can_change_only_the_named_word():
    subtitles = SubtitleService()
    context = ContextService()
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("Can you type Symfpony?")
    sub.finalize()

    good = (
        '{"action": "correct_subtitle", "target_subtitle_id": "' + sub.id + '", '
        '"replacement_text": "Can you type Symphony?", '
        '"vocabulary_term": null, "reason": "clear mishearing"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(good)):
        result = await service.correct(parse_command("Correct Symfpony in the last sentence."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.backend == "llm"
    assert_corrected_as_child(subtitles, sub.id, "Can you type Symfpony?", "Can you type Symphony?")


async def test_naming_a_word_alone_never_lets_the_rules_engine_invent_a_fix():
    """The deterministic backend has no opinion about an unknown word."""
    subtitles = SubtitleService()
    context = ContextService()
    service = build(subtitles, context, backend="rules")
    sub = subtitles.add("Can you type Symfpony?")
    sub.finalize()

    result = await service.correct(parse_command("Correct Symfpony in the last sentence."))

    # Either it declines or it passes the line through unchanged; what it must
    # never do is substitute a word of its own invention.
    assert result.backend == "rules"
    assert_uncorrected(subtitles, sub.id, "Can you type Symfpony?")


async def test_a_named_term_in_the_previous_line_corrects_the_previous_line():
    subtitles = SubtitleService()
    context = ContextService()
    service = build(subtitles, context, backend="llm")
    first = subtitles.add("Can you type Symfpony?")
    first.finalize()
    second = subtitles.add("Moving on to the next topic.")
    second.finalize()

    good = (
        '{"action": "correct_subtitle", "target_subtitle_id": "' + first.id + '", '
        '"replacement_text": "Can you type Symphony?", '
        '"vocabulary_term": null, "reason": "clear mishearing"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(good)):
        result = await service.correct(parse_command("Correct Symfpony in the previous sentence."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert_corrected_as_child(
        subtitles, first.id, "Can you type Symfpony?", "Can you type Symphony?"
    )
    assert subtitles.get(second.id).text == "Moving on to the next topic."


async def test_a_named_term_with_an_ordinal_target_corrects_that_line():
    subtitles = SubtitleService()
    context = ContextService()
    service = build(subtitles, context, backend="llm")
    first = subtitles.add("First line of the demo.")
    first.finalize()
    second = subtitles.add("Second line of the demo.")
    second.finalize()
    third = subtitles.add("Can you type Symfpony?")
    third.finalize()

    good = (
        '{"action": "correct_subtitle", "target_subtitle_id": "' + third.id + '", '
        '"replacement_text": "Can you type Symphony?", '
        '"vocabulary_term": null, "reason": "clear mishearing"}'
    )
    with patch("httpx.AsyncClient.post", return_value=_llm_response(good)):
        result = await service.correct(parse_command("Correct Symfpony in sentence 3."))

    assert result.outcome is CorrectionOutcome.APPLIED
    # The target the model was pointed at is the third original, and no other
    # line moved.
    assert_corrected_as_child(
        subtitles, third.id, "Can you type Symfpony?", "Can you type Symphony?"
    )
    assert subtitles.get(first.id).text == "First line of the demo."
    assert subtitles.get(second.id).text == "Second line of the demo."


def test_the_prompt_explains_the_two_meanings_of_a_named_term():
    """The pointer is ambiguous across phrasings, so the model is told both."""
    prompt = correction_service.SYSTEM_PROMPT

    assert "user_named_term" in prompt
    assert "believe is wrong" in prompt
    assert "believe is right" in prompt
