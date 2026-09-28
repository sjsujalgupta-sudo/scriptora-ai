"""Correction engine.

The LLM is mocked at the HTTP boundary, so these tests exercise the real
validation and fallback logic without touching the network.
"""

from __future__ import annotations

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
from scriptora.services.command_service import parse_command
from scriptora.services.context_service import ContextService
from scriptora.services.correction_service import CorrectionService
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
def test_rules_fixes_the_primary_demo_case(context_with_fastapi):
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context)

    sub = subtitles.add("Today we're building the backend using fast API.")
    sub.finalize()

    result = service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.subtitle_id == sub.id
    assert subtitles.get(sub.id).text == "Today we're building the backend using FastAPI."


def test_rules_honours_an_explicit_replace(context_with_fastapi):
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    sub = subtitles.add("We are working with Alice on this.")
    sub.finalize()

    service.correct(parse_command("Replace Alice with Atlas."))

    assert subtitles.get(sub.id).text == "We are working with Atlas on this."


def test_rules_reports_no_change_when_nothing_is_wrong(context_with_fastapi):
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    sub = subtitles.add("We are using FastAPI today.")
    sub.finalize()

    result = service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.NO_ACTION
    assert result.before == result.after


def test_correction_with_no_subtitles_is_reported_not_crashed(context_with_fastapi):
    service = build(SubtitleService(), context_with_fastapi)

    result = service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.INVALID
    assert "no subtitles" in result.message.lower()


def test_unsupported_command_is_reported(context_with_fastapi):
    subtitles = SubtitleService()
    subtitles.add("some text").finalize()
    service = build(subtitles, context_with_fastapi)

    result = service.correct(parse_command("Delete the last subtitle"))

    assert result.outcome is CorrectionOutcome.UNSUPPORTED


def test_correct_previous_targets_the_earlier_line(context_with_fastapi):
    subtitles = SubtitleService()
    service = build(subtitles, context_with_fastapi)
    first = subtitles.add("We deploy with fast api.")
    first.finalize()
    subtitles.finalize("This is the second line.")

    result = service.correct(parse_command("Fix the previous subtitle."))

    assert result.subtitle_id == first.id
    assert "FastAPI" in subtitles.get(first.id).text


# ============================================================== remember
def test_remember_adds_to_vocabulary(context_with_fastapi):
    context = context_with_fastapi
    service = build(SubtitleService(), context)

    result = service.correct(parse_command("Remember Kubernetes as a technical term."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert "Kubernetes" in context.terms
    assert result.vocabulary_term == "Kubernetes"


def test_remember_is_idempotent(context_with_fastapi):
    context = context_with_fastapi
    service = build(SubtitleService(), context)
    service.correct(parse_command("Remember Kubernetes."))

    result = service.correct(parse_command("Remember Kubernetes."))

    assert "already" in result.message.lower()
    assert context.terms.count("Kubernetes") == 1


def test_remembered_term_is_used_by_the_next_correction(context_with_fastapi):
    """The memory feature must actually affect later corrections."""
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context)

    service.correct(parse_command("Remember FastAPI as a technical term."))
    sub = subtitles.add("We rely on fast api for the service.")
    sub.finalize()

    service.correct(parse_command("Correct the last subtitle."))

    assert "FastAPI" in subtitles.get(sub.id).text


# ============================================================== LLM paths
def _llm_response(content: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]},
    )


def test_llm_result_is_applied_when_valid(context_with_fastapi):
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
    with patch("httpx.post", return_value=_llm_response(good)):
        result = service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.backend == "llm"
    assert subtitles.get(sub.id).text == "Today we are building the backend using FastAPI."


def test_llm_response_with_invented_id_is_discarded(context_with_fastapi):
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
    with patch("httpx.post", return_value=_llm_response(bad)):
        service.correct(parse_command("Correct the last subtitle."))

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
def test_invalid_llm_output_falls_back_to_rules(bad_content, context_with_fastapi):
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("Using fast API.")
    sub.finalize()

    with patch("httpx.post", return_value=_llm_response(bad_content)):
        result = service.correct(parse_command("Correct the last subtitle."))

    assert subtitles.get(sub.id).text == "Using FastAPI."
    assert result.outcome is CorrectionOutcome.APPLIED


def test_llm_http_error_falls_back_to_rules(context_with_fastapi):
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("Using fast API.")
    sub.finalize()

    with patch("httpx.post", return_value=httpx.Response(500, text="boom")):
        result = service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.backend == "rules"
    assert "AI unavailable" in result.reason


def test_llm_network_error_falls_back_to_rules(context_with_fastapi):
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("Using fast API.")
    sub.finalize()

    with patch("httpx.post", side_effect=httpx.ConnectError("no route")):
        result = service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert "AI unavailable" in result.reason


def test_llm_malformed_envelope_falls_back_to_rules(context_with_fastapi):
    subtitles = SubtitleService()
    context = context_with_fastapi
    service = build(subtitles, context, backend="llm")
    sub = subtitles.add("Using fast API.")
    sub.finalize()

    with patch("httpx.post", return_value=httpx.Response(200, json={"unexpected": True})):
        result = service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
    assert result.backend == "rules"


def test_explicit_replace_wins_over_llm_rewrite(context_with_fastapi):
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
    with patch("httpx.post", return_value=_llm_response(llm_says)) as post:
        result = service.correct(parse_command("Replace Alice with Atlas."))

    assert subtitles.get(sub.id).text == "We work with Atlas."
    # No HTTP call: the answer could not have changed the outcome.
    assert post.call_count == 0
    # And it is never misattributed to the model.
    assert result.backend == "rules"


def test_replace_skips_the_llm_even_when_the_edit_is_impossible(context_with_fastapi):
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
    with patch("httpx.post", return_value=_llm_response(llm_says)) as post:
        result = service.correct(parse_command("Replace missing with present."))

    assert post.call_count == 0
    assert result.outcome is CorrectionOutcome.NO_ACTION
    assert subtitles.get(sub.id).text == "Nothing relevant here."


def test_correction_commands_still_consult_the_llm(context_with_fastapi):
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
    with patch("httpx.post", return_value=_llm_response(llm_says)) as post:
        result = service.correct(parse_command("Correct the last subtitle"))

    assert post.call_count == 1
    assert result.backend == "llm"
    assert subtitles.get(sub.id).text == "we use FastAPI here"


def test_no_api_key_disables_the_llm_backend(context_with_fastapi):
    settings = Settings(assemblyai_api_key=None, corrector_backend="auto")
    subtitles = SubtitleService()
    service = CorrectionService(settings, subtitles, context_with_fastapi)

    assert service.active_backend == "rules"
    sub = subtitles.add("Using fast API.")
    sub.finalize()
    result = service.correct(parse_command("Correct the last subtitle."))

    assert result.outcome is CorrectionOutcome.APPLIED
